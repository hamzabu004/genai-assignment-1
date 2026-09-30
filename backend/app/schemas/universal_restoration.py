from typing import Any, Dict
from pydantic import BaseModel, Field

class CorruptionApplied(BaseModel):
    type: str = Field(..., description="Corruption type applied, e.g. blur, salt_pepper, occlusion, clean")
    severity: str = Field(..., description="Severity level: low, medium, high, or clean/none")
    params: Dict[str, Any] = Field(default_factory=dict, description="Parameters used for corruption")

class UniversalRestorationResponse(BaseModel):
    input_image: str = Field(..., description="Base64 encoded original input image")
    corrupted_image: str = Field(..., description="Base64 encoded corrupted image")
    output_image: str = Field(..., description="Base64 encoded restored image")
    error_map_image: str = Field(..., description="Base64 encoded residual error map image")
    corruption_applied: CorruptionApplied
    inference_time_ms: float = Field(..., description="Model inference runtime in milliseconds")

