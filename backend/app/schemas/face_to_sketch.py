from pydantic import BaseModel, Field

class FaceToSketchResponse(BaseModel):
    input_image: str = Field(..., description="Base64 encoded original facial photo")
    sketch_image: str = Field(..., description="Base64 encoded synthesized sketch output")
    style_used: str = Field(..., description="Artistic sketch style used (style_1, style_2, style_3)")
    inference_time_ms: float = Field(..., description="Model inference runtime in milliseconds")

