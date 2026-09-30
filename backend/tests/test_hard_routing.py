def test_hard_routing_clean(client, sample_image_bytes):
    res = client.post(
        "/hard-routing",
        files={"image": ("test.png", sample_image_bytes, "image/png")},
        data={"corruption_type": "clean"},
    )
    assert res.status_code == 200
    data = res.json()
    assert "class_probabilities" in data
    assert "clean" in data["class_probabilities"]
    assert "salt_pepper" in data["class_probabilities"]
    assert "blur" in data["class_probabilities"]
    assert "occlusion" in data["class_probabilities"]
    assert data["predicted_class"] in ["clean", "salt_pepper", "blur", "occlusion"]
    assert data["selected_expert"] in [
        "identity_pass",
        "salt_specialist",
        "blur_specialist",
        "occlusion_specialist",
    ]
    assert data["output_image"].startswith("data:image/png;base64,")


def test_hard_routing_blur(client, sample_image_bytes):
    res = client.post(
        "/hard-routing",
        files={"image": ("test.png", sample_image_bytes, "image/png")},
        data={"corruption_type": "blur", "severity": "high"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["corrupted_image"].startswith("data:image/png;base64,")
    assert data["output_image"].startswith("data:image/png;base64,")
    prob_sum = sum(data["class_probabilities"].values())
    assert 0.99 <= prob_sum <= 1.01

