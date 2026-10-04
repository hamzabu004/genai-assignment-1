import numpy as np
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
    run_model_with_inputs,
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

    # Task 4 was trained on [-1, 1] images and takes an integer style ID.
    x = image_to_tensor(face_img)
    style_idx = VALID_STYLES.index(clean_style)

    with timer() as t:
        if not has_model("generator"):
            raise HTTPException(
                status_code=503,
                detail="Model file not available: task4_generator.onnx. Please place the model in backend/models.",
            )
        try:
            outputs = run_model_with_inputs(
                "generator",
                {"photo": x, "style_idx": np.asarray([style_idx], dtype=np.int64)},
            )
            if not outputs:
                raise RuntimeError("Generator returned no outputs")
            y = outputs[0]
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Face-to-sketch ONNX inference failed: {exc}")

    inference_ms = round(t["ms"], 2)

    # 4. Format output
    sketch_img = tensor_to_image(y)

    return FaceToSketchResponse(
        input_image=image_to_base64_png(face_img),
        sketch_image=image_to_base64_png(sketch_img),
        style_used=clean_style,
        inference_time_ms=inference_ms,
    )
