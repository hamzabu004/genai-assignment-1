import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        protected_namespaces=(),
    )

    model_dir: str = "./models"
    img_size: int = 128
    allowed_origins: str = "http://localhost:3000"
    log_level: str = "info"

    # Preprocessing contract — single source of truth (see Plan 2, Section 4)
    normalize_mean: list[float] = [0.5, 0.5, 0.5]
    normalize_std: list[float] = [0.5, 0.5, 0.5]
    channel_order: str = "RGB"
    tensor_layout: str = "NCHW"

    @property
    def resolved_model_dir(self) -> str:
        """Resolves model directory regardless of whether invoked from root or backend/."""
        # 1. Direct path with .onnx files
        if os.path.isdir(self.model_dir):
            try:
                if any(f.endswith(".onnx") for f in os.listdir(self.model_dir)):
                    return self.model_dir
            except OSError:
                pass

        # 2. Bundled models in Docker container
        bundled_docker = "/app/bundled_models"
        if os.path.isdir(bundled_docker):
            try:
                if any(f.endswith(".onnx") for f in os.listdir(bundled_docker)):
                    return bundled_docker
            except OSError:
                pass

        # 3. Relative backend/models directory
        candidate = os.path.join("backend", self.model_dir.lstrip("./"))
        if os.path.isdir(candidate):
            try:
                if any(f.endswith(".onnx") for f in os.listdir(candidate)):
                    return candidate
            except OSError:
                pass

        # 4. Fallback relative to this file: backend/app/core/config.py -> backend/models
        here = os.path.dirname(os.path.abspath(__file__))
        backend_dir = os.path.dirname(os.path.dirname(here))
        fallback = os.path.join(backend_dir, "models")
        if os.path.isdir(fallback):
            return fallback

        return self.model_dir


settings = Settings()

