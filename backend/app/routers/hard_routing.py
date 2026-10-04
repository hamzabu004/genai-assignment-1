from typing import Optional, Dict
import numpy as np
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.schemas.hard_routing import HardRoutingResponse
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

router = APIRouter(tags=["Routing"])

CLASSES = ["clean", "salt_pepper", "blur", "occlusion"]
EXPERT_NAMES = {
    "clean": "identity_pass",
    "salt_pepper": "salt_specialist",
    "blur": "blur_specialist",
    "occlusion": "occlusion_specialist",
}


@router.post("/hard-routing", response_model=HardRoutingResponse)
async def hard_routing(
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

    # 4. Execute Pipeline with timer: Classifier gating -> Argmax -> Expert dispatch
    # Tasks 2a/2b were trained on [0, 1] tensors.
    x = image_to_unit_tensor(corrupted_img)

    with timer() as t:
        if not has_model("classifier"):
            raise HTTPException(
                status_code=503,
                detail="Model file not available: task2_classifier.onnx. Please place the model in backend/models.",
            )

        logits = run_model("classifier", x)[0]
        probs_arr = softmax(logits)

        pred_idx = int(np.argmax(probs_arr))
        predicted_class = CLASSES[pred_idx]
        selected_expert = EXPERT_NAMES[predicted_class]

        # Expert Dispatch
        if predicted_class == "clean":
            y = x
        elif predicted_class == "salt_pepper":
            if not has_model("specialist_salt"):
                raise HTTPException(
                    status_code=503,
                    detail="Model file not available: task2_specialist_salt.onnx. Please place the model in backend/models.",
                )
            y = run_model("specialist_salt", x)[0]
        elif predicted_class == "blur":
            if not has_model("specialist_blur"):
                raise HTTPException(
                    status_code=503,
                    detail="Model file not available: task2_specialist_blur.onnx. Please place the model in backend/models.",
                )
            y = run_model("specialist_blur", x)[0]
        elif predicted_class == "occlusion":
            if not has_model("specialist_occlusion"):
                raise HTTPException(
                    status_code=503,
                    detail="Model file not available: task2_specialist_occlusion.onnx. Please place the model in backend/models.",
                )
            y = run_model("specialist_occlusion", x)[0]
        else:
            y = x

    inference_ms = round(t["ms"], 2)

    # 5. Format outputs
    class_probs: Dict[str, float] = format_probabilities(probs_arr, CLASSES)
    restored_img = unit_tensor_to_image(y)

    return HardRoutingResponse(
        input_image=image_to_base64_png(clean_img),
        corrupted_image=image_to_base64_png(corrupted_img),
        output_image=image_to_base64_png(restored_img),
        class_probabilities=class_probs,
        predicted_class=predicted_class,
        selected_expert=selected_expert,
        inference_time_ms=inference_ms,
    )
