import pytest


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
