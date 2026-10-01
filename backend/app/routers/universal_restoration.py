import json
import math
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from skimage.metrics import structural_similarity

from app.core.config import settings
from app.schemas.universal_restoration import (
    CorruptionApplied,
    UniversalRestorationResponse,
    ValidationSample,
)
from app.services.corruption import (
    SEVERITY_LEVELS,
    VALID_CORRUPTIONS,
    apply_corruption_to_image,
    apply_manifest_corruption,
)
from app.services.onnx_runtime_manager import has_model, run_model
from app.services.postprocessing import error_map_to_base64_png
from app.services.preprocessing import (
    image_to_base64_png,
    image_to_unit_tensor,
    load_image,
    unit_tensor_to_image,
)
from app.utils.timing import timer

router = APIRouter(tags=["Restoration"])


def _read_validation_manifest() -> Dict[str, Dict[str, Any]]:
    manifest_path = Path(settings.resolved_validation_manifest_path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail=f"Validation manifest is unavailable: {manifest_path}")
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=503, detail=f"Could not read validation manifest: {exc}")
    if not isinstance(manifest, dict):
        raise HTTPException(status_code=503, detail="Validation manifest must contain an image mapping")
    return manifest


def _validation_sample(filename: str) -> tuple[Path, Dict[str, Any]]:
    if Path(filename).name != filename or filename in ("", ".", ".."):
        raise HTTPException(status_code=400, detail="Invalid validation sample filename")
    manifest = _read_validation_manifest()
    if filename not in manifest:
        raise HTTPException(status_code=404, detail="Validation sample is not listed in the official manifest")
    image_dir = Path(settings.resolved_validation_image_dir).resolve()
    image_path = (image_dir / filename).resolve()
    if image_path.parent != image_dir:
        raise HTTPException(status_code=400, detail="Invalid validation sample path")
    if not image_path.is_file():
        raise HTTPException(status_code=404, detail=f"Validation image is missing: {filename}")
    return image_path, manifest[filename]


def _quality_metrics(reference, prediction) -> Dict[str, float]:
    reference_arr = np.asarray(reference.convert("RGB"), dtype=np.float32) / 255.0
    prediction_arr = np.asarray(prediction.convert("RGB"), dtype=np.float32) / 255.0
    mse = float(np.mean(np.square(reference_arr - prediction_arr)))
    psnr = 10.0 * math.log10(1.0 / max(mse, 1e-12))
    ssim = float(structural_similarity(reference_arr, prediction_arr, channel_axis=2, data_range=1.0))
    return {"psnr_db": float(psnr), "ssim": ssim}


def _run_universal(
    clean_img,
    corrupted_img,
    applied: Dict[str, Any],
    sample_filename: Optional[str] = None,
) -> UniversalRestorationResponse:
    if not has_model("universal_ae"):
        raise HTTPException(
            status_code=503,
            detail="Universal ONNX model is not loaded. Place task1_universal_ae.onnx in MODEL_DIR and restart the backend.",
        )

    x = image_to_unit_tensor(corrupted_img)
    with timer() as elapsed:
        try:
            outputs = run_model("universal_ae", x)
            if not outputs:
                raise RuntimeError("ONNX model returned no outputs")
            output = np.asarray(outputs[0], dtype=np.float32)
            if output.shape != x.shape or not np.isfinite(output).all():
                raise RuntimeError(f"Unexpected model output shape or values: {output.shape}")
            restored_img = unit_tensor_to_image(output)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Universal ONNX inference failed: {exc}")

    return UniversalRestorationResponse(
        input_image=image_to_base64_png(clean_img),
        corrupted_image=image_to_base64_png(corrupted_img),
        output_image=image_to_base64_png(restored_img),
        error_map_image=error_map_to_base64_png(clean_img, restored_img),
        corruption_applied=CorruptionApplied(**applied),
        inference_time_ms=round(elapsed["ms"], 2),
        quality_metrics=_quality_metrics(clean_img, restored_img),
        sample_filename=sample_filename,
    )


@router.get("/universal-restoration/validation-samples", response_model=list[ValidationSample])
def list_validation_samples():
    manifest = _read_validation_manifest()
    samples = []
    for filename, entry in sorted(manifest.items()):
        kind = str(entry.get("corruption_type", "")).lower()
        if kind not in ("clean", "salt", "salt_pepper", "salt-pepper", "blur", "occlusion"):
            continue
        normalized_kind = "salt_pepper" if kind in ("salt", "salt-pepper") else kind
        samples.append(
            ValidationSample(
                filename=filename,
                corruption_type=normalized_kind,
                severity=str(entry.get("severity", "none")),
                params={key: value for key, value in entry.items() if key != "corruption_type"},
            )
        )
    return samples


@router.post(
    "/universal-restoration/validation-samples/{filename}",
    response_model=UniversalRestorationResponse,
)
def restore_validation_sample(filename: str):
    image_path, manifest_entry = _validation_sample(filename)
    try:
        clean_img = load_image(image_path.read_bytes())
        corrupted_img, applied = apply_manifest_corruption(clean_img, manifest_entry)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not prepare validation sample: {exc}")
    return _run_universal(clean_img, corrupted_img, applied, sample_filename=filename)


@router.post("/universal-restoration", response_model=UniversalRestorationResponse)
async def universal_restoration(
    image: UploadFile = File(..., description="Uploaded image file (JPEG, PNG, etc.)"),
    corruption_type: Optional[str] = Form(None, description="clean, salt_pepper, blur, occlusion"),
    severity: Optional[str] = Form(None, description="low, medium, high"),
):
    file_bytes = await image.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty")

    try:
        clean_img = load_image(file_bytes)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {exc}")

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

    try:
        corrupted_img, applied = apply_corruption_to_image(clean_img, c_type, sev)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Error applying corruption: {exc}")
    return _run_universal(clean_img, corrupted_img, applied)
