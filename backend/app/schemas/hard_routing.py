from typing import Dict
from pydantic import BaseModel, Field

class HardRoutingResponse(BaseModel):
    input_image: str = Field(..., description="Base64 encoded original input image")
    corrupted_image: str = Field(..., description="Base64 encoded corrupted image")
    output_image: str = Field(..., description="Base64 encoded restored image")
    class_probabilities: Dict[str, float] = Field(..., description="Classifier probabilities per degradation class")
    predicted_class: str = Field(..., description="Argmax predicted degradation class")
    selected_expert: str = Field(..., description="Name of selected expert model or pass-through")
    inference_time_ms: float = Field(..., description="Total pipeline inference time in milliseconds")

