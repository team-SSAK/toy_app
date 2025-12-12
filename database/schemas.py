from pydantic import BaseModel
from typing import Optional # accountId가 Nullable일 경우 대비

# 회원가입 및 인증 관련 스키마
class UserRegister(BaseModel):
    """
    회원가입 요청 스키마 (users 테이블의 name, phoneNum, accountId와 일치)
    """
    name: str
    phoneNum: str
    accountId: str # user 테이블에 NOT NULL 제약조건이 없으므로 Optional도 가능

class UserLogin(BaseModel):
    """
    로그인 요청 스키마 (전화번호만)
    """
    phoneNum: str

class UserLoginWithName(BaseModel):
    """
    로그인 요청 스키마 (이름 + 전화번호)
    """
    name: str
    phoneNum: str

class Token(BaseModel):
    """
    JWT 토큰 응답 스키마
    """
    access_token: str
    token_type: str

# 데이터 조회/응답 관련 스키마 (필요에 따라 추가)
class MeasurementBase(BaseModel):
    """
    측정 결과의 기본 스키마 (measurements 테이블 컬럼 반영)
    """
    image_url: str
    leftover_ratio: float
    # measured_at, user_id, id는 DB가 생성하므로 응답 스키마에만 포함

class MeasurementResponse(MeasurementBase):
    """
    측정 이력 조회 시 사용
    """
    id: int
    measured_at: str # TIMESTAMP를 문자열로 가정
    
    class Config:
        # Pydantic이 ORM 객체(예: DB 결과)에서 필드를 읽을 수 있도록 허용
        from_attributes = True 

class UserInfoResponse(BaseModel):
    """
    사용자 정보 조회 시 사용 (users 테이블 컬럼 반영)
    """
    id: int
    name: str
    phoneNum: str
    accountId: Optional[str] = None
    measure_cnt: int
    
    class Config:
        from_attributes = True