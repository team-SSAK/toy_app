# 1. 베이스 이미지: Python 3.11 slim (CPU 전용)
FROM python:3.11-slim

# 캐시 줄이기
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

# 2. 필수 OS 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 3. 작업 디렉토리 설정
WORKDIR /app

# 4. requirements.txt 복사 및 설치
COPY requirements.txt .

# numpy 먼저 설치
RUN pip install --no-cache-dir "numpy<2"

# CPU 전용 torch 설치
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1
# 나머지 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# 5. 소스 코드 복사
COPY . .

# 6. 모델 경로 확인 (선택)
# RUN ls models

# 7. FastAPI 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
