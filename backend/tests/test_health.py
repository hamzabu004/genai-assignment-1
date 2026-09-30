def test_health_check(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert isinstance(data["models_loaded"], list)
    expected_models = [
        "universal_ae",
        "classifier",
        "specialist_salt",
        "specialist_blur",
        "specialist_occlusion",
        "soft_moe",
        "generator",
    ]
    for model_name in expected_models:
        assert model_name in data["models_loaded"]

