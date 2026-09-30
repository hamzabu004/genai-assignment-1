from typing import Dict
from pydantic import BaseModel, Field

class SoftMixtureResponse(BaseModel):
    input_image: str = Field(..., description="Base64 encoded original input image")
    corrupted_image: str = Field(..., description="Base64 encoded corrupted image")
    output_image: str = Field(..., description="Base64 encoded restored image")
    routing_weights: Dict[str, float] = Field(..., description="Continuous softmax gating weights summing to 1.0")
    dominant_expert: str = Field(..., description="Key of highest-weighted expert network")
    inference_time_ms: float = Field(..., description="Total pipeline inference time in milliseconds")

