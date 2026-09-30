from typing import Optional, List
from pydantic import BaseModel

class ApiErrorResponse(BaseModel):
    error: bool = True
    message: str
    detail: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    models_loaded: List[str]

