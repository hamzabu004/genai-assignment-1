from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.schemas.face_to_sketch import FaceToSketchResponse
from app.services.preprocessing import (
    load_image,
    image_to_tensor,
    tensor_to_image,
    image_to_base64_png,
)
from app.services.onnx_runtime_manager import (
    has_model,
    run_model,
    fallback_face_to_sketch,
)
from app.utils.timing import timer

router = APIRouter(tags=["Face to Sketch"])

VALID_STYLES = ("style_1", "style_2", "style_3")


@router.post("/face-to-sketch", response_model=FaceToSketchResponse)
async def face_to_sketch(
    image: UploadFile = File(..., description="Facial portrait image"),
    style: str = Form(..., description="Artistic sketch style: style_1, style_2, style_3"),
):
    # 1. Read & validate uploaded image
    file_bytes = await image.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty")

    try:
        face_img = load_image(file_bytes)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {str(exc)}")

    # 2. Validate style parameter
    clean_style = (style or "").strip().lower()
    if clean_style not in VALID_STYLES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid style '{style}'. Must be one of: {list(VALID_STYLES)}",
        )

    # 3. Model Inference with timer
    x = image_to_tensor(face_img)

    with timer() as t:
        if has_model("generator"):
            try:
                y = fallback_face_to_sketch(x, clean_style)
            except Exception:
                out = run_model("generator", x)
                y = out[0]
        else:
            y = fallback_face_to_sketch(x, clean_style)

    inference_ms = round(t["ms"], 2)

    # 4. Format output
    sketch_img = tensor_to_image(y)

    return FaceToSketchResponse(
        input_image=image_to_base64_png(face_img),
        sketch_image=image_to_base64_png(sketch_img),
        style_used=clean_style,
        inference_time_ms=inference_ms,
    )

