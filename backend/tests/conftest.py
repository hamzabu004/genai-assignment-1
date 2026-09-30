import io
import sys
import os
import pytest
from PIL import Image
from fastapi.testclient import TestClient

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.main import app


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_image_bytes():
    img = Image.new("RGB", (128, 128), color=(120, 160, 200))
    for x in range(30, 98):
        for y in range(30, 98):
            img.putpixel((x, y), (200, 100, 80))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def sample_large_image_bytes():
    img = Image.new("RGB", (256, 256), color=(70, 90, 110))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

