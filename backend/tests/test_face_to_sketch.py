import pytest


@pytest.mark.parametrize("style", ["style_1", "style_2", "style_3"])
def test_face_to_sketch_styles(client, sample_image_bytes, style):
    res = client.post(
        "/face-to-sketch",
        files={"image": ("face.png", sample_image_bytes, "image/png")},
        data={"style": style},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["style_used"] == style
    assert data["input_image"].startswith("data:image/png;base64,")
    assert data["sketch_image"].startswith("data:image/png;base64,")
    assert data["inference_time_ms"] >= 0


def test_face_to_sketch_invalid_style(client, sample_image_bytes):
    res = client.post(
        "/face-to-sketch",
        files={"image": ("face.png", sample_image_bytes, "image/png")},
        data={"style": "cubism"},
    )
    assert res.status_code == 400
    data = res.json()
    assert data["error"] is True
    assert "cubism" in data["message"]


def test_face_to_sketch_missing_style(client, sample_image_bytes):
    res = client.post(
        "/face-to-sketch",
        files={"image": ("face.png", sample_image_bytes, "image/png")},
    )
    assert res.status_code == 422
    data = res.json()
    assert data["error"] is True

