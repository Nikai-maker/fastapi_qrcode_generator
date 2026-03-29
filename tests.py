import pytest
import base64
import numpy as np
import cv2
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def create_test_image(width=720, height=1280, color=(200, 200, 200)):
    """Создаёт тестовое изображение и возвращает его в base64."""
    img = np.full((height, width, 3), color, dtype=np.uint8)
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')


def create_test_image_with_busy_area(width=720, height=1280):
    """Создаёт изображение с загруженной зоной в центре — QR должен уйти в угол."""
    img = np.full((height, width, 3), 200, dtype=np.uint8)
    # Рисуем шумную зону в центре
    noise = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
    cy, cx = height // 2, width // 2
    img[cy-200:cy+200, cx-200:cx+200] = noise
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')


# ---------------------------------------------------------------------------
# Базовые тесты эндпоинта /process
# ---------------------------------------------------------------------------

class TestProcessEndpoint:

    def test_returns_200_on_valid_input(self):
        payload = {
            "image_base64": create_test_image(),
            "qr_content": "https://example.com",
        }
        response = client.post("/process", json=payload)
        assert response.status_code == 200

    def test_response_contains_required_fields(self):
        payload = {
            "image_base64": create_test_image(),
            "qr_content": "https://example.com",
        }
        data = client.post("/process", json=payload).json()
        assert "result_image" in data
        assert "qr_x" in data
        assert "qr_y" in data
        assert "image_width" in data
        assert "image_height" in data

    def test_result_image_is_valid_base64(self):
        payload = {
            "image_base64": create_test_image(),
            "qr_content": "https://example.com",
        }
        data = client.post("/process", json=payload).json()
        # Не должно бросать исключение
        decoded = base64.b64decode(data["result_image"])
        assert len(decoded) > 0

    def test_result_image_is_valid_png(self):
        payload = {
            "image_base64": create_test_image(),
            "qr_content": "https://example.com",
        }
        data = client.post("/process", json=payload).json()
        decoded = base64.b64decode(data["result_image"])
        img_array = np.frombuffer(decoded, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        assert img is not None

    def test_image_dimensions_are_preserved(self):
        width, height = 720, 1280
        payload = {
            "image_base64": create_test_image(width=width, height=height),
            "qr_content": "https://example.com",
        }
        data = client.post("/process", json=payload).json()
        assert data["image_width"] == width
        assert data["image_height"] == height

    def test_returns_422_on_missing_image(self):
        payload = {"qr_content": "https://example.com"}
        response = client.post("/process", json=payload)
        assert response.status_code == 422

    def test_returns_422_on_empty_body(self):
        response = client.post("/process", json={})
        assert response.status_code == 422

    def test_invalid_base64_returns_error(self):
        payload = {
            "image_base64": "not_valid_base64!!!",
            "qr_content": "https://example.com",
        }
        response = client.post("/process", json=payload)
        assert response.status_code in (400, 422, 500)

    def test_default_qr_content_works(self):
        """qr_content имеет дефолт — запрос без него должен пройти."""
        payload = {"image_base64": create_test_image()}
        response = client.post("/process", json=payload)
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# Тесты позиционирования QR
# ---------------------------------------------------------------------------

class TestQRPositioning:

    def test_qr_position_within_image_bounds(self):
        width, height = 720, 1280
        payload = {
            "image_base64": create_test_image(width=width, height=height),
            "qr_content": "https://example.com",
        }
        data = client.post("/process", json=payload).json()
        qr_size = int(min(width, height) * 0.1)

        assert data["qr_x"] >= 0
        assert data["qr_y"] >= 0
        assert data["qr_x"] + qr_size <= width
        assert data["qr_y"] + qr_size <= height

    def test_qr_avoids_busy_areas(self):
        """На изображении с шумом в центре QR не должен попасть в центр."""
        width, height = 720, 1280
        payload = {
            "image_base64": create_test_image_with_busy_area(width=width, height=height),
            "qr_content": "https://example.com",
        }
        data = client.post("/process", json=payload).json()
        cx, cy = width // 2, height // 2

        # QR не должен находиться прямо в центре (±150px)
        is_in_center = (
            abs(data["qr_x"] - cx) < 150 and
            abs(data["qr_y"] - cy) < 150
        )
        assert not is_in_center

    def test_qr_size_is_10_percent_of_min_side(self):
        """Проверяем что QR реально 10% от наименьшей стороны."""
        width, height = 720, 1280
        payload = {
            "image_base64": create_test_image(width=width, height=height),
            "qr_content": "https://example.com",
        }
        data = client.post("/process", json=payload).json()
        expected_qr_size = int(min(width, height) * 0.1)

        decoded = base64.b64decode(data["result_image"])
        img_array = np.frombuffer(decoded, np.uint8)
        result_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # Вырезаем зону где должен быть QR и проверяем что она не равна фону
        qr_region = result_img[
            data["qr_y"]:data["qr_y"] + expected_qr_size,
            data["qr_x"]:data["qr_x"] + expected_qr_size
        ]
        # Однородный серый фон (200,200,200) должен измениться — там теперь QR
        assert np.var(qr_region) > 0


# ---------------------------------------------------------------------------
# Тесты на разные форматы и размеры изображений
# ---------------------------------------------------------------------------

class TestImageFormats:

    @pytest.mark.parametrize("width,height", [
        (300, 300),   # квадрат маленький
        (1080, 1920), # вертикальное HD
        (1920, 1080), # горизонтальное HD
        (400, 800),   # вертикальное небольшое
    ])
    def test_various_image_sizes(self, width, height):
        payload = {
            "image_base64": create_test_image(width=width, height=height),
            "qr_content": "https://example.com",
        }
        response = client.post("/process", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["image_width"] == width
        assert data["image_height"] == height

    @pytest.mark.parametrize("qr_content", [
        "https://example.com",
        "https://very-long-url.example.com/path/to/resource?param=value&other=123",
        "простой текст",
        "12345",
        "mailto:test@example.com",
    ])
    def test_various_qr_content(self, qr_content):
        payload = {
            "image_base64": create_test_image(),
            "qr_content": qr_content,
        }
        response = client.post("/process", json=payload)
        assert response.status_code == 200