def test_empty_image_error(client):
    res = client.post(
        "/universal-restoration",
        files={"image": ("empty.png", b"", "image/png")},
        data={"corruption_type": "clean"},
    )
    assert res.status_code == 400
    data = res.json()
    assert data["error"] is True
    assert "empty" in data["message"].lower()


def test_corrupt_file_error(client):
    res = client.post(
        "/universal-restoration",
        files={"image": ("corrupted.png", b"not_an_image_binary_data", "image/png")},
        data={"corruption_type": "clean"},
    )
    assert res.status_code == 400
    data = res.json()
    assert data["error"] is True
    assert "invalid image format" in data["message"].lower()


def test_validation_error_shape(client):
    res = client.post("/universal-restoration")
    assert res.status_code == 422
    data = res.json()
    assert data["error"] is True
    assert "message" in data

