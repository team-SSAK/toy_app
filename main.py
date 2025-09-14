import io
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse 
from fastapi.staticfiles import StaticFiles
from PIL import Image
from ultralytics import YOLO

MODEL_PATH = "models/yolo8m.pt"
model = YOLO(MODEL_PATH)

app = FastAPI(title="YOLOv8 잔반 비율 계산 API (통합 서버)")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

def calculate_leftover_ratio(image: Image.Image):
    """
    - 'dishes'는 항상 포함
    - 다른 클래스는 신뢰도 0.4 이상일 때만 포함
    """
    results = model(image, verbose=False)
    result = results[0]

    if result.masks is None or result.boxes is None:
        return 0.0

    h, w = result.orig_shape
    leftover_mask = np.zeros((h, w), dtype=bool)
    dish_mask = np.zeros((h, w), dtype=bool)

    for i in range(len(result.boxes)):
        class_id = int(result.boxes.cls[i])
        class_name = result.names[class_id]
        confidence = float(result.boxes.conf[i])

        # 필터링 로직: dishes는 무조건, 나머지는 신뢰도 0.4 이상
        is_kept = False
        if class_name == "dishes":
            is_kept = True
        elif class_name == "leftovers" and confidence >= 0.4:
            is_kept = True
        
        if not is_kept:
            continue

        # 필터링 통과한 객체의 마스크 처리
        instance_mask_tensor = result.masks.data[i]
        instance_mask_np = instance_mask_tensor.cpu().numpy().astype(np.uint8)
        instance_mask_resized = cv2.resize(instance_mask_np, (w, h)).astype(bool)

        if class_name == "leftovers":
            leftover_mask |= instance_mask_resized
        elif class_name == "dishes":
            dish_mask |= instance_mask_resized

    # 겹치는 영역 처리: 그릇 영역에서 잔반 영역은 제외
    final_dish_mask = dish_mask & (~leftover_mask)

    leftover_pixels = np.sum(leftover_mask)
    dishes_pixels = np.sum(final_dish_mask)
    total_dish_area = leftover_pixels + dishes_pixels

    if total_dish_area == 0:
        return 0.0

    ratio = (leftover_pixels / total_dish_area)
    return ratio

# --- API 엔드포인트 생성 ---
@app.post("/predict/leftover_ratio")
async def get_leftover_ratio(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일이 아닙니다.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        ratio = calculate_leftover_ratio(image)
        return {"leftover_ratio": float(ratio)}
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"이미지 처리 중 오류 발생: {e}")