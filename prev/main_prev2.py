import io
import os
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from database.schemas import UserRegister, UserLogin, UserLoginWithName, Token
from services.auth_service import get_current_user
from services.yolo_service import YOLOService
from services.s3_service import S3Service
from database.db_manager import DatabaseManager
from database.init_db import init_database

# 서비스 초기화
MODEL_PATH = "models/yolo8m.pt"
yolo_service = YOLOService(MODEL_PATH)
s3_service = S3Service()
db_manager = DatabaseManager()

app = FastAPI(title="YOLOv8 잔반 비율 계산 API (모듈화 버전)")

# 정적 파일 제공
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.on_event("startup")
async def startup_event():
    """앱 시작 시 데이터베이스 초기화"""
    init_database()

# ========== 페이지 라우트 ==========

@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

@app.get("/login")
async def login_page():
    """로그인 페이지"""
    return FileResponse('static/login.html')

# ========== API 엔드포인트 ==========

@app.post("/api/register")
async def register(user: UserRegister):
    """회원가입"""
    try:
        db_manager.create_user(user.name, user.phoneNum, user.accountId)
        return {"message": "회원가입이 완료되었습니다."}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/login", response_model=Token)
async def login(user: UserLogin):
    """로그인 (전화번호만)"""
    try:
        token = db_manager.login_user(user.phoneNum)
        return {"access_token": token, "token_type": "bearer"}
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/login/with-name", response_model=Token)
async def login_with_name(user: UserLoginWithName):
    """로그인 (이름 + 전화번호)"""
    try:
        token = db_manager.login_user_with_name(user.name, user.phoneNum)
        return {"access_token": token, "token_type": "bearer"}
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict")
async def predict_with_save(
    file: UploadFile = File(...),
    user_id: int = Depends(get_current_user)
):
    """잔반 비율 측정 및 결과 저장 (인증 필요)"""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일이 아닙니다.")

    try:
        # 이미지 읽기
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # 잔반 비율 계산
        ratio = yolo_service.calculate_leftover_ratio(image)
        
        # S3에 이미지 업로드
        image_url = s3_service.upload_image(contents)
        
        # DB에 결과 저장
        measurement_id = db_manager.save_measurement(user_id, image_url, ratio)
        
        return {
            "measurement_id": measurement_id,
            "leftover_ratio": float(ratio),
            "image_url": image_url
        }
    except Exception as e:
        import traceback
        print("ERROR:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"처리 중 오류 발생: {e}")

@app.get("/api/history")
async def get_history(user_id: int = Depends(get_current_user)):
    """사용자의 측정 이력 조회"""
    try:
        history = db_manager.get_user_history(user_id)
        return {"history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/user/info")
async def get_user_info(user_id: int = Depends(get_current_user)):
    """사용자 정보 조회"""
    try:
        user_info = db_manager.get_user_info(user_id)
        if not user_info:
            raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")
        return user_info
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/leftover_ratio")
async def get_leftover_ratio(file: UploadFile = File(...)):
    """레거시 API - 인증 없이 잔반 비율만 반환"""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일이 아닙니다.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        ratio = yolo_service.calculate_leftover_ratio(image)
        return {"leftover_ratio": float(ratio)}
    except Exception as e:
        import traceback
        print("ERROR:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"이미지 처리 중 오류 발생: {e}")