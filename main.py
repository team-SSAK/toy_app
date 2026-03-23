import io
import os
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import FileResponse, RedirectResponse
from PIL import Image

from database.schemas import UserRegister, UserLogin, UserLoginWithName, Token
from services.auth_service import get_current_user
from services.mmseg_service import MMSegService
from services.s3_service import S3Service
from database.db_manager import DatabaseManager
from database.init_db import init_database

# 캠페인 종료 여부 (환경 변수로 제어)
CAMPAIGN_CLOSED = os.getenv("CAMPAIGN_CLOSED", "false").lower() == "true"

# mmseg 레포 루트(= configs/, mmseg/ 있는 폴더)
MMSEG_REPO_ROOT = "models/segmentation_plate_leftover-main"
CKPT_PATH = "models/best_mIoU_iter_12000.pth"

# 서비스 초기화
CLOSED_PATH = "static/closed.html"
print("🚀 서버 시작 중...")
seg_service = MMSegService(MMSEG_REPO_ROOT, CKPT_PATH)
s3_service = S3Service()
db_manager = DatabaseManager()

app = FastAPI(title="YOLOv8 잔반 비율 계산 API")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.on_event("startup")
async def startup_event():
    """앱 시작 시 DB 초기화"""
    init_database()
    if CAMPAIGN_CLOSED:
        print("⚠️  캠페인 종료 모드로 실행 중입니다.")

# ========== 페이지 라우트 ==========

@app.get("/")
async def root():
    """메인 페이지 (잔반 측정) - 캠페인 종료 시 리다이렉트"""
    if CAMPAIGN_CLOSED:
        return RedirectResponse(url="/closed")
    return FileResponse('static/index.html')

@app.get("/login")
async def login_page():
    """로그인 페이지 - 캠페인 종료 시 리다이렉트"""
    if CAMPAIGN_CLOSED:
        return RedirectResponse(url="/closed")
    return FileResponse('static/login.html')

@app.get("/closed")
async def closed_page():
    """캠페인 종료 페이지"""
    return FileResponse(CLOSED_PATH)

@app.get("/exchange")
async def exchange_page():
    """포인트 교환 페이지 - 캠페인 종료 시 리다이렉트"""
    if CAMPAIGN_CLOSED:
        return RedirectResponse(url="/closed")
    return FileResponse('static/exchange.html')

@app.get("/coupons")
async def coupons_page():
    """마이 쿠폰함 페이지 - 캠페인 종료 시 리다이렉트"""
    if CAMPAIGN_CLOSED:
        return RedirectResponse(url="/closed")
    return FileResponse('static/coupons.html')

# ========== API 엔드포인트 ==========

@app.post("/api/register")
async def register(user: UserRegister):
    """회원가입"""
    try:
        referrerPhoneNum = user.referrerPhoneNum if user.referrerPhoneNum and user.referrerPhoneNum.strip() else None
        db_manager.create_user(user.name, user.phoneNum, user.mealSize, referrerPhoneNum)
        return {"message": "회원가입이 완료되었습니다."}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"❌ 회원가입 에러: {str(e)}")  # ✅ 디버깅용 로그
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
        ratio = seg_service.calculate_leftover_ratio(image)
        
        # S3에 이미지 업로드
        image_url = s3_service.upload_image(contents)
        
        # DB에 결과 저장
        measurement_id = db_manager.save_measurement(user_id, image_url, ratio)
        
        return {
            "measurement_id": measurement_id,
            "leftover_ratio": float(ratio),
            "image_url": image_url
        }
    except ValueError as e:
        # 이미 측정한 경우
        raise HTTPException(status_code=400, detail=str(e))
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
        result = seg_service.predict(image)
        return result
    
    except Exception as e:
        import traceback
        print("ERROR:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"이미지 처리 중 오류 발생: {e}")

@app.post("/api/exchange")
async def request_change(user_id: int = Depends(get_current_user)):
    """커피쿠폰 교환 신청"""
    try:
        # 사용자 정보 조회
        user_info = db_manager.get_user_info(user_id)
        if not user_info:
            raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")

        # 포인트 확인
        required_points = 300
        if user_info.get("point", 0) < required_points:
            raise HTTPException(status_code=400, detail=f"포인트가 부족합니다. (필요 {required_points}점, 보유: {user_info.get('point', 0)}점)")
        
        # 교환 신청 처리
        db_manager.request_exchange(user_id)
        return {"message": "커피쿠폰 교환 신청이 접수되었습니다."}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/exchange/history")
async def get_exchange_history(user_id:int = Depends(get_current_user)):
    """사용자의 교환 이력 조회"""
    try:
        history = db_manager.get_user_exchange_history(user_id)
        return {
            "total": len(history),
            "history": history
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"이력 조회 실패: {str(e)}")
    
@app.post("/api/exchange/{exchange_id}/use")
async def use_coupon(exchange_id: int, user_id: int = Depends(get_current_user)):
    """쿠폰 사용 완료 처리"""
    try:
        success = db_manager.use_coupon(exchange_id, user_id)
        
        if success:
            return {
                "message": "쿠폰이 사용 완료되었습니다.",
                "exchange_id": exchange_id
            }
        else:
            raise HTTPException(status_code=400, detail="쿠폰 사용 처리에 실패했습니다.")
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"쿠폰 사용 처리 실패: {str(e)}")