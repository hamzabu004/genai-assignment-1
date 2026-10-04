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


def test_universal_restoration_missing_model_error(client, sample_image_bytes, monkeypatch):
    from app.routers import universal_restoration
    monkeypatch.setattr(universal_restoration, "has_model", lambda key: False)

    res = client.post(
        "/universal-restoration",
        files={"image": ("test.png", sample_image_bytes, "image/png")},
        data={"corruption_type": "clean"},
    )
    assert res.status_code == 503
    data = res.json()
    assert data["error"] is True
    assert "model file not available" in data["message"].lower()
    assert "task1_universal_ae.onnx" in data["message"]


def test_hard_routing_missing_classifier_error(client, sample_image_bytes, monkeypatch):
    from app.routers import hard_routing
    monkeypatch.setattr(hard_routing, "has_model", lambda key: False)

    res = client.post(
        "/hard-routing",
        files={"image": ("test.png", sample_image_bytes, "image/png")},
        data={"corruption_type": "clean"},
    )
    assert res.status_code == 503
    data = res.json()
    assert data["error"] is True
    assert "model file not available" in data["message"].lower()
    assert "task2_classifier.onnx" in data["message"]


def test_hard_routing_missing_specialist_error(client, sample_image_bytes, monkeypatch):
    from app.routers import hard_routing
    # Classifier is present, but blur specialist is missing
    monkeypatch.setattr(hard_routing, "has_model", lambda key: key != "specialist_blur")

    res = client.post(
        "/hard-routing",
        files={"image": ("test.png", sample_image_bytes, "image/png")},
        data={"corruption_type": "blur", "severity": "high"},
    )
    assert res.status_code == 503
    data = res.json()
    assert data["error"] is True
    assert "model file not available" in data["message"].lower()
    assert "task2_specialist_blur.onnx" in data["message"]


def test_soft_mixture_missing_models_error(client, sample_image_bytes, monkeypatch):
    from app.routers import soft_mixture
    monkeypatch.setattr(soft_mixture, "has_model", lambda key: False)

    res = client.post(
        "/soft-mixture",
        files={"image": ("test.png", sample_image_bytes, "image/png")},
        data={"corruption_type": "blur", "severity": "medium"},
    )
    assert res.status_code == 503
    data = res.json()
    assert data["error"] is True
    assert "model file not available" in data["message"].lower()
    assert "task3_soft_moe.onnx" in data["message"]


def test_face_to_sketch_missing_generator_error(client, sample_image_bytes, monkeypatch):
    from app.routers import face_to_sketch
    monkeypatch.setattr(face_to_sketch, "has_model", lambda key: False)

    res = client.post(
        "/face-to-sketch",
        files={"image": ("face.png", sample_image_bytes, "image/png")},
        data={"style": "style_1"},
    )
    assert res.status_code == 503
    data = res.json()
    assert data["error"] is True
    assert "model file not available" in data["message"].lower()
    assert "task4_generator.onnx" in data["message"]


