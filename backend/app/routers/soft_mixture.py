from typing import Optional, Dict
import numpy as np
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.schemas.soft_mixture import SoftMixtureResponse
from app.services.preprocessing import (
    load_image,
    image_to_base64_png,
    image_to_unit_tensor,
    unit_tensor_to_image,
)
from app.services.corruption import apply_corruption_to_image, VALID_CORRUPTIONS, SEVERITY_LEVELS
from app.services.postprocessing import softmax, format_probabilities
from app.services.onnx_runtime_manager import (
    has_model,
    run_model,
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
    # Tasks 2a/2b/3 were trained on [0, 1] tensors.
    x = image_to_unit_tensor(corrupted_img)

    with timer() as t:
        if has_model("soft_moe"):
            moe_outs = run_model("soft_moe", x)
            y = moe_outs[0]
            if len(moe_outs) > 1 and moe_outs[1] is not None:
                weights_arr = np.asarray(moe_outs[1], dtype=np.float32).flatten()
            elif has_model("classifier"):
                c_logits = run_model("classifier", x)[0]
                weights_arr = softmax(c_logits).flatten()
            else:
                weights_arr = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        elif (
            has_model("classifier")
            and has_model("specialist_salt")
            and has_model("specialist_blur")
            and has_model("specialist_occlusion")
        ):
            c_logits = run_model("classifier", x)[0]
            weights_arr = softmax(c_logits).flatten()
            routing_weights = format_probabilities(weights_arr, EXPERT_KEYS)
            w_ident = routing_weights.get("identity", 0.25)
            w_salt = routing_weights.get("salt_pepper", 0.25)
            w_blur = routing_weights.get("blur", 0.25)
            w_occl = routing_weights.get("occlusion", 0.25)

            y_salt = run_model("specialist_salt", x)[0]
            y_blur = run_model("specialist_blur", x)[0]
            y_occl = run_model("specialist_occlusion", x)[0]

            y = (w_ident * x) + (w_salt * y_salt) + (w_blur * y_blur) + (w_occl * y_occl)
        else:
            raise HTTPException(
                status_code=503,
                detail="Model file not available: task3_soft_moe.onnx. Please place the model in backend/models.",
            )

        routing_weights: Dict[str, float] = format_probabilities(weights_arr, EXPERT_KEYS)
        dominant_expert = max(routing_weights.items(), key=lambda kv: kv[1])[0]

    inference_ms = round(t["ms"], 2)

    # 5. Format outputs
    restored_img = unit_tensor_to_image(y)

    return SoftMixtureResponse(
        input_image=image_to_base64_png(clean_img),
        corrupted_image=image_to_base64_png(corrupted_img),
        output_image=image_to_base64_png(restored_img),
        routing_weights=routing_weights,
        dominant_expert=dominant_expert,
        inference_time_ms=inference_ms,
    )
