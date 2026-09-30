from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.schemas.universal_restoration import UniversalRestorationResponse, CorruptionApplied
from app.services.preprocessing import (
    load_image,
    image_to_tensor,
    tensor_to_image,
    image_to_base64_png,
)
from app.services.corruption import apply_corruption_to_image, VALID_CORRUPTIONS, SEVERITY_LEVELS
from app.services.postprocessing import error_map_to_base64_png
from app.services.onnx_runtime_manager import has_model, run_model, fallback_universal_ae
from app.utils.timing import timer

router = APIRouter(tags=["Restoration"])


@router.post("/universal-restoration", response_model=UniversalRestorationResponse)
async def universal_restoration(
    image: UploadFile = File(..., description="Uploaded image file (JPEG, PNG, etc.)"),
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
        corrupted_img, applied_meta = apply_corruption_to_image(clean_img, c_type, sev)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Error applying corruption: {str(exc)}")

    # 4. Run Model Inference with timing
    x = image_to_tensor(corrupted_img)

    with timer() as t:
        if has_model("universal_ae"):
            out_tensors = run_model("universal_ae", x)
            y = out_tensors[0]
        else:
            y = fallback_universal_ae(x)

    inference_ms = round(t["ms"], 2)

    # 5. Format outputs
    restored_img = tensor_to_image(y)
    error_map_b64 = error_map_to_base64_png(clean_img, restored_img)

    return UniversalRestorationResponse(
        input_image=image_to_base64_png(clean_img),
        corrupted_image=image_to_base64_png(corrupted_img),
        output_image=image_to_base64_png(restored_img),
        error_map_image=error_map_b64,
        corruption_applied=CorruptionApplied(**applied_meta),
        inference_time_ms=inference_ms,
    )

