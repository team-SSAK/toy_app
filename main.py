import io
import os
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from PIL import Image

from services.mmseg_service import MMSegService
from services.s3_service import S3Service

# mmseg 설정
MMSEG_REPO_ROOT = "models/segmentation_plate_leftover-main"
CKPT_PATH = "models/best_mIoU_iter_12000.pth"

# 서비스 초기화
print("🚀 모델 서버 시작 중...")
seg_service = MMSegService(MMSEG_REPO_ROOT, CKPT_PATH)
s3_service = S3Service()

app = FastAPI(title="AI Recognition Model Server")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/api/predict")
async def predict_and_upload(file: UploadFile = File(...)):
    """
    이미지를 받아 분석 후, S3 URL과 인식 결과(ratio)를 반환함.
    DB 저장은 여기서 하지 않고, 반환값을 받은 자바 서버가 처리함.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일이 아닙니다.")

    try:
        # 1. 이미지 읽기
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # 2. 잔반 비율 계산 (AI 추론)
        ratio = seg_service.calculate_leftover_ratio(image)
        
        # 3. S3에 이미지 업로드 (추후 플랫폼 서버에서 조회용)
        # 여기서 업로드하면 플랫폼 서버는 이미지 파일을 직접 다룰 필요가 없어짐
        image_url = s3_service.upload_image(contents)
        
        # 4. 결과 반환 (자바 서버가 받을 내용)
        return {
            "leftover_ratio": float(ratio),
            "image_url": image_url,
            "status": "success"
        }
        
    except Exception as e:
        import traceback
        print("ERROR:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"AI 분석 중 오류 발생: {e}")
