from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import cv2
import numpy as np
import os
from pathlib import Path
import uuid

app = FastAPI()

MODEL_PATH = r"models\detectionModel\best.pt"
CROP_DIR = Path("assets/croppedImages")
UPLOAD_DIR = Path("assets/uploads")
CROP_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

PADDING_RATIO = 0.15

model = YOLO(MODEL_PATH)

@app.post("/detect-crop")
async def detect_crop(file: UploadFile = File(...)):
    # Save upload
    suffix = Path(file.filename).suffix or ".jpg"
    upload_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    data = await file.read()
    upload_path.write_bytes(data)

    # Read image
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return {"ok": False, "error": "Invalid image"}

    # Run inference
    results = model.predict(source=image, conf=0.25)

    crop_count = 0
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = result.names[class_id]

            img_h, img_w = image.shape[:2]
            box_w = max(0, x2 - x1)
            box_h = max(0, y2 - y1)
            pad_x = int(box_w * PADDING_RATIO)
            pad_y = int(box_h * PADDING_RATIO)

            x1p = max(0, x1 - pad_x)
            y1p = max(0, y1 - pad_y)
            x2p = min(img_w, x2 + pad_x)
            y2p = min(img_h, y2 + pad_y)

            if x2p <= x1p or y2p <= y1p:
                continue

            cropped = image[y1p:y2p, x1p:x2p]
            crop_filename = f"{class_name}_{crop_count}_{confidence:.2f}.jpg"
            crop_path = CROP_DIR / crop_filename
            cv2.imwrite(str(crop_path), cropped)
            crop_count += 1

    return {"ok": True, "crops": crop_count, "saved_to": str(CROP_DIR)}