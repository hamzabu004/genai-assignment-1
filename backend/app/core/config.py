import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        protected_namespaces=(),
    )

    model_dir: str = "./models"
    validation_manifest_path: str = "./validation/val_manifest_official.json"
    validation_image_dir: str = "./validation/images"
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

    @property
    def resolved_validation_manifest_path(self) -> str:
        if os.path.isfile(self.validation_manifest_path):
            return self.validation_manifest_path
        bundled_manifest = "/app/bundled_validation/val_manifest_official.json"
        if os.path.isfile(bundled_manifest):
            return bundled_manifest
        backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        return os.path.join(os.path.dirname(backend_dir), "research", "val_manifest_official.json")

    @property
    def resolved_validation_image_dir(self) -> str:
        if os.path.isdir(self.validation_image_dir):
            try:
                if any(os.scandir(self.validation_image_dir)):
                    return self.validation_image_dir
            except OSError:
                pass
        bundled_images = "/app/bundled_validation/images"
        if os.path.isdir(bundled_images):
            try:
                if any(os.scandir(bundled_images)):
                    return bundled_images
            except OSError:
                pass
        backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        return os.path.join(
            os.path.dirname(backend_dir), "research", "datasets", "oxford-iiit-pet", "images"
        )


settings = Settings()
