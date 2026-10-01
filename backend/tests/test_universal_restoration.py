import base64
import io
import json
import math
from pathlib import Path

import pytest
from PIL import Image

from app.core.config import settings
from app.routers import universal_restoration as universal_router
from app.services.preprocessing import image_to_unit_tensor, unit_tensor_to_image


def test_universal_restoration_clean(client, sample_image_bytes):
    res = client.post(
        "/universal-restoration",
        files={"image": ("test.png", sample_image_bytes, "image/png")},
        data={"corruption_type": "clean"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["input_image"].startswith("data:image/png;base64,")
    assert data["corrupted_image"].startswith("data:image/png;base64,")
    assert data["output_image"].startswith("data:image/png;base64,")
    assert data["error_map_image"].startswith("data:image/png;base64,")
    assert data["corruption_applied"]["type"] == "clean"
    assert data["inference_time_ms"] >= 0


@pytest.mark.parametrize(
    "c_type,sev",
    [
        ("blur", "medium"),
        ("blur", "low"),
        ("blur", "high"),
        ("salt_pepper", "low"),
        ("salt_pepper", "medium"),
        ("salt_pepper", "high"),
        ("occlusion", "low"),
        ("occlusion", "medium"),
        ("occlusion", "high"),
    ],
)
def test_universal_restoration_corruptions(client, sample_image_bytes, c_type, sev):
    res = client.post(
        "/universal-restoration",
        files={"image": ("test.png", sample_image_bytes, "image/png")},
        data={"corruption_type": c_type, "severity": sev},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["corruption_applied"]["type"] == c_type
    assert data["corruption_applied"]["severity"] == sev
    assert isinstance(data["corruption_applied"]["params"], dict)
    assert len(data["corruption_applied"]["params"]) > 0


def test_universal_restoration_auto_resizing(client, sample_large_image_bytes):
    res = client.post(
        "/universal-restoration",
        files={"image": ("large.png", sample_large_image_bytes, "image/png")},
        data={"corruption_type": "blur", "severity": "low"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["output_image"].startswith("data:image/png;base64,")


def test_universal_restoration_missing_severity(client, sample_image_bytes):
    res = client.post(
        "/universal-restoration",
        files={"image": ("test.png", sample_image_bytes, "image/png")},
        data={"corruption_type": "blur"},
    )
    assert res.status_code == 400
    data = res.json()
    assert data["error"] is True
    assert "severity" in data["message"].lower()


def test_universal_restoration_invalid_corruption(client, sample_image_bytes):
    res = client.post(
        "/universal-restoration",
        files={"image": ("test.png", sample_image_bytes, "image/png")},
        data={"corruption_type": "unknown_noise", "severity": "medium"},
    )
    assert res.status_code == 400
    data = res.json()
    assert data["error"] is True


def test_universal_restoration_requires_loaded_onnx_model(client, sample_image_bytes, monkeypatch):
    monkeypatch.setattr(universal_router, "has_model", lambda key: False)
    res = client.post(
        "/universal-restoration",
        files={"image": ("test.png", sample_image_bytes, "image/png")},
        data={"corruption_type": "clean"},
    )
    assert res.status_code == 503
    assert "ONNX model is not loaded" in res.json()["message"]


def _install_identity_model(monkeypatch):
    monkeypatch.setattr(universal_router, "has_model", lambda key: True)
    monkeypatch.setattr(universal_router, "run_model", lambda key, x: [x.copy()])


def test_validation_sample_catalog_matches_official_manifest(client):
    manifest_path = Path(settings.resolved_validation_manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    response = client.get("/universal-restoration/validation-samples")
    assert response.status_code == 200
    samples = response.json()
    assert len(samples) == len(manifest) == 736
    assert {sample["filename"] for sample in samples} == set(manifest)


@pytest.mark.parametrize(
    ("manifest_kind", "api_kind"),
    [("clean", "clean"), ("salt", "salt_pepper"), ("blur", "blur"), ("occlusion", "occlusion")],
)
def test_manifest_sample_uses_recorded_corruption(client, monkeypatch, manifest_kind, api_kind):
    _install_identity_model(monkeypatch)
    manifest = json.loads(Path(settings.resolved_validation_manifest_path).read_text(encoding="utf-8"))
    filename, entry = next(
        (name, item) for name, item in manifest.items() if item["corruption_type"] == manifest_kind
    )
    response = client.post(f"/universal-restoration/validation-samples/{filename}")
    assert response.status_code == 200, response.text
    result = response.json()
    for key in ("input_image", "corrupted_image", "output_image", "error_map_image"):
        assert result[key].startswith("data:image/png;base64,")
    assert result["sample_filename"] == filename
    assert result["corruption_applied"]["type"] == api_kind
    assert result["corruption_applied"]["severity"] == entry.get("severity", "none")
    assert result["corruption_applied"]["params"] == {
        key: value for key, value in entry.items() if key != "corruption_type"
    }
    assert set(result["quality_metrics"]) == {"psnr_db", "ssim"}
    assert all(math.isfinite(value) for value in result["quality_metrics"].values())

    corrupted_bytes = base64.b64decode(result["corrupted_image"].split(",", 1)[1])
    with Image.open(io.BytesIO(corrupted_bytes)) as corrupted:
        assert corrupted.size == (128, 128)
        input_bytes = base64.b64decode(result["input_image"].split(",", 1)[1])
        with Image.open(io.BytesIO(input_bytes)) as clean:
            if manifest_kind == "clean":
                assert list(corrupted.getdata()) == list(clean.getdata())
            else:
                assert list(corrupted.getdata()) != list(clean.getdata())

    repeated = client.post(f"/universal-restoration/validation-samples/{filename}")
    assert repeated.status_code == 200
    assert repeated.json()["corrupted_image"] == result["corrupted_image"]


@pytest.mark.parametrize("filename", ["../val_manifest_official.json", "../../etc/passwd"])
def test_validation_sample_rejects_path_traversal(client, filename):
    response = client.post(f"/universal-restoration/validation-samples/{filename}")
    assert response.status_code in (400, 404)


def test_validation_sample_rejects_filename_not_in_manifest(client):
    response = client.post("/universal-restoration/validation-samples/not-a-validation-image.jpg")
    assert response.status_code == 404


def test_task1_preprocessing_keeps_unit_range(sample_image_bytes):
    from app.services.preprocessing import load_image

    image = load_image(sample_image_bytes)
    tensor = image_to_unit_tensor(image)
    assert tensor.shape == (1, 3, 128, 128)
    assert tensor.dtype.name == "float32"
    assert float(tensor.min()) >= 0.0
    assert float(tensor.max()) <= 1.0
    restored = unit_tensor_to_image(tensor)
    assert restored.size == (128, 128)
