import base64
import io
import numpy as np
from PIL import Image
from app.core.config import settings


def load_image(file_bytes: bytes) -> Image.Image:
    """
    Decodes image bytes, ensures RGB format, and resizes to IMG_SIZE x IMG_SIZE using bilinear interpolation.
    Raises ValueError on invalid image data.
    """
    try:
        img = Image.open(io.BytesIO(file_bytes))
        img = img.convert("RGB")
        # Ensure target size
        if img.size != (settings.img_size, settings.img_size):
            img = img.resize((settings.img_size, settings.img_size), Image.Resampling.BILINEAR)
        return img
    except Exception as exc:
        raise ValueError(f"Failed to decode image: {str(exc)}") from exc


def image_to_tensor(img: Image.Image) -> np.ndarray:
    """
    Converts PIL RGB Image to normalized NCHW float32 numpy tensor.
    Matches research pipeline exactly.
    """
    if img.size != (settings.img_size, settings.img_size):
        img = img.resize((settings.img_size, settings.img_size), Image.Resampling.BILINEAR)

    arr = np.asarray(img).astype(np.float32) / 255.0
    mean = np.array(settings.normalize_mean, dtype=np.float32)
    std = np.array(settings.normalize_std, dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
    arr = np.expand_dims(arr, axis=0)  # add batch dim -> NCHW
    return arr.astype(np.float32)


def load_image_to_tensor(file_bytes: bytes) -> np.ndarray:
    """RGB decode -> resize -> normalize -> NCHW float32, matches research pipeline exactly."""
    img = load_image(file_bytes)
    return image_to_tensor(img)


def tensor_to_image(tensor: np.ndarray) -> Image.Image:
    """
    Denormalizes NCHW or CHW float tensor back to PIL RGB Image.
    """
    arr = tensor[0] if tensor.ndim == 4 else tensor
    if arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))  # CHW -> HWC
    elif arr.shape[0] == 1:
        # Grayscale channel -> duplicate to 3 channels
        arr = np.transpose(arr, (1, 2, 0))
        arr = np.repeat(arr, 3, axis=2)

    mean = np.array(settings.normalize_mean, dtype=np.float32)
    std = np.array(settings.normalize_std, dtype=np.float32)
    arr = arr * std + mean
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def image_to_base64_png(img: Image.Image) -> str:
    """Encodes a PIL Image to a base64 data URI string."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


def tensor_to_base64_png(tensor: np.ndarray) -> str:
    """Denormalize NCHW float tensor -> base64 PNG data URI string."""
    img = tensor_to_image(tensor)
    return image_to_base64_png(img)

