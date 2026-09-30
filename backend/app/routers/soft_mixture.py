from typing import Optional, Dict
import numpy as np
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.schemas.soft_mixture import SoftMixtureResponse
from app.services.preprocessing import (
    load_image,
    image_to_tensor,
    tensor_to_image,
    image_to_base64_png,
)
from app.services.corruption import apply_corruption_to_image, VALID_CORRUPTIONS, SEVERITY_LEVELS
from app.services.postprocessing import softmax, format_probabilities
from app.services.onnx_runtime_manager import (
    has_model,
    run_model,
    fallback_classify_degradation,
    fallback_denoise_median,
    fallback_unsharp_mask,
    fallback_inpaint_occlusion,
)
from app.utils.timing import timer

router = APIRouter(tags=["Mixture of Experts"])

EXPERT_KEYS = ["identity", "salt_pepper", "blur", "occlusion"]


@router.post("/soft-mixture", response_model=SoftMixtureResponse)
async def soft_mixture(
    image: UploadFile = File(..., description="Uploaded image file"),
    corruption_type: Optional[str] = Form(None, description="clean, salt_pepper, blur, occlusion"),
    severity: Optional[str] = Form(None, description="low, medium, high"),
):
    # 1. Read & validate uploaded image
    file_bytes = await image.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty")

    try:
        clean_img = load_image(file_bytes)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {str(exc)}")

    # 2. Validate corruption parameters
    c_type = (corruption_type or "clean").strip().lower()
    if c_type not in VALID_CORRUPTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid corruption_type '{corruption_type}'. Must be one of: {list(VALID_CORRUPTIONS)}",
        )

    if c_type != "clean":
        if not severity:
            raise HTTPException(
                status_code=400,
                detail=f"Field 'severity' is required when corruption_type is '{c_type}'. Must be one of: {list(SEVERITY_LEVELS)}",
            )
        sev = severity.strip().lower()
        if sev not in SEVERITY_LEVELS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid severity '{severity}'. Must be one of: {list(SEVERITY_LEVELS)}",
            )
    else:
        sev = None

    # 3. Apply corruption
    try:
        corrupted_img, _ = apply_corruption_to_image(clean_img, c_type, sev)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Error applying corruption: {str(exc)}")

    # 4. Execute Soft MoE continuous blend with timer
    x = image_to_tensor(corrupted_img)

    with timer() as t:
        # A. Determine routing weights (softmax attention distribution)
        if has_model("classifier"):
            c_logits = run_model("classifier", x)[0]
            weights_arr = softmax(c_logits).flatten()
        else:
            weights_arr, _ = fallback_classify_degradation(x)
            weights_arr = weights_arr.flatten()

        routing_weights: Dict[str, float] = format_probabilities(weights_arr, EXPERT_KEYS)
        dominant_expert = max(routing_weights.items(), key=lambda kv: kv[1])[0]

        # B. Blend expert latent outputs continuously
        if has_model("soft_moe"):
            moe_outs = run_model("soft_moe", x)
            y = moe_outs[0]
        else:
            w_ident = routing_weights.get("identity", 0.25)
            w_salt = routing_weights.get("salt_pepper", 0.25)
            w_blur = routing_weights.get("blur", 0.25)
            w_occl = routing_weights.get("occlusion", 0.25)

            y_salt = (
                run_model("specialist_salt", x)[0]
                if has_model("specialist_salt")
                else fallback_denoise_median(x)
            )
            y_blur = (
                run_model("specialist_blur", x)[0]
                if has_model("specialist_blur")
                else fallback_unsharp_mask(x)
            )
            y_occl = (
                run_model("specialist_occlusion", x)[0]
                if has_model("specialist_occlusion")
                else fallback_inpaint_occlusion(x)
            )

            y = (w_ident * x) + (w_salt * y_salt) + (w_blur * y_blur) + (w_occl * y_occl)

    inference_ms = round(t["ms"], 2)

    # 5. Format outputs
    restored_img = tensor_to_image(y)

    return SoftMixtureResponse(
        input_image=image_to_base64_png(clean_img),
        corrupted_image=image_to_base64_png(corrupted_img),
        output_image=image_to_base64_png(restored_img),
        routing_weights=routing_weights,
        dominant_expert=dominant_expert,
        inference_time_ms=inference_ms,
    )

