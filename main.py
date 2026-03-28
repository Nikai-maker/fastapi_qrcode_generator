from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import cv2
import numpy as np
import base64
import qrcode
from PIL import Image
import io

app = FastAPI()

class ImageRequest(BaseModel):
    image_base64: str
    qr_content: str = "https://example.com"
    qr_size: int = 200

@app.post("/process")
def process(req: ImageRequest):
    # Декодируем входное изображение
    img_bytes = base64.b64decode(req.image_base64)
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    h, w = img.shape[:2]
    qr = int(min(h, w) * 0.15) 
    padding = int(qr * 0.5)  # отступ вокруг QR — 50% от его размера
    best_x, best_y = 0, 0
    min_variance = float('inf')
    step = 10  # точнее чем 20

    for y in range(padding, h - qr - padding, step):
        for x in range(padding, w - qr - padding, step):
            # Оцениваем зону QR плюс отступы вокруг
            region = img[y-padding:y+qr+padding, x-padding:x+qr+padding]
            variance = np.var(region)
            if variance < min_variance:
                min_variance = variance
                best_x, best_y = x, y

    # Генерируем QR код
    qr_img = qrcode.make(req.qr_content)
    qr_img = qr_img.resize((qr, qr), Image.LANCZOS)
    qr_array = np.array(qr_img.convert("RGB"))
    qr_bgr = cv2.cvtColor(qr_array, cv2.COLOR_RGB2BGR)

    # Накладываем QR на изображение
    img[best_y:best_y+qr, best_x:best_x+qr] = qr_bgr

    # Кодируем результат обратно в base64
    _, buffer = cv2.imencode('.png', img)
    result_base64 = base64.b64encode(buffer).decode('utf-8')

    return {
        "result_image": result_base64,
        "qr_x": best_x,
        "qr_y": best_y,
        "image_width": w,
        "image_height": h
    }
