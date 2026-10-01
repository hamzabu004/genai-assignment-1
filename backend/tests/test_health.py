def test_health_check(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert isinstance(data["models_loaded"], list)
    assert "universal_ae" in data["models_loaded"]
    assert set(data["models_loaded"]).issubset(
        {
            "universal_ae",
            "classifier",
            "specialist_salt",
            "specialist_blur",
            "specialist_occlusion",
            "soft_moe",
            "generator",
        }
    )
    assert data["onnx_providers"]
