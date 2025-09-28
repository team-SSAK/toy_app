# 1. Python 3.10 slim 버전 사용 (가볍고 안정적)
FROM python:3.10-slim

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. 시스템 의존성 설치 (opencv, numpy 등 빌드에 필요)
RUN apt-get update && apt-get install -y \
    git \
    curl \
    gcc \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# 4. requirements 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 소스 복사
COPY . .

# 6. 컨테이너 시작 시 실행할 명령어
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]