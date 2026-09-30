from fastapi import APIRouter
from app.services.onnx_runtime_manager import loaded_model_keys, active_providers

router = APIRouter()

@router.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": loaded_model_keys(),
        "onnx_providers": active_providers(),
    }
