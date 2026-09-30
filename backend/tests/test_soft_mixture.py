def test_soft_mixture_basic(client, sample_image_bytes):
    res = client.post(
        "/soft-mixture",
        files={"image": ("test.png", sample_image_bytes, "image/png")},
        data={"corruption_type": "blur", "severity": "medium"},
    )
    assert res.status_code == 200
    data = res.json()
    assert "routing_weights" in data
    weights = data["routing_weights"]
    for k in ["identity", "salt_pepper", "blur", "occlusion"]:
        assert k in weights
        assert weights[k] >= 0.0

    weight_sum = sum(weights.values())
    assert 0.99 <= weight_sum <= 1.01

    assert data["dominant_expert"] in ["identity", "salt_pepper", "blur", "occlusion"]
    assert data["dominant_expert"] == max(weights, key=weights.get)
    assert data["output_image"].startswith("data:image/png;base64,")

