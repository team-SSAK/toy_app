# 1) Python 3.10 (OpenMMLab 안정 조합)
FROM python:3.10-slim

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

# 2) OS packages (mmcv/mmdet + opencv runtime 대비)
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 3) requirements 복사
COPY requirements.txt .

# 4) pip 기본 세팅 + numpy 먼저
RUN pip install --no-cache-dir -U pip setuptools wheel
RUN pip install --no-cache-dir "numpy<2"

# 5) torch CPU (requirements에 버전 고정되어 있어도, torch는 여기서 먼저 설치하는 게 안전)
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    torch==2.1.2 torchvision==0.16.2

# 6) OpenMMLab: mmcv는 mim으로 설치해서 ops(_ext) 포함되게
#    (requirements.txt에 mmcv==2.1.0 있어도, 여기서 먼저 깔면 pip가 "이미 설치됨"으로 넘어감)
RUN pip install --no-cache-dir openmim
RUN pip install --no-cache-dir mmengine==0.10.3
RUN mim install "mmcv==2.1.0"

# 7) 나머지 requirements 설치
#    (mmcv/mmengine/torch는 이미 설치되어 있으니 여기서는 나머지만 맞춰짐)
RUN pip install --no-cache-dir -r requirements.txt

# 8) 누락 의존성 (mmseg tokenizer 쪽에서 자주 필요)
RUN pip install --no-cache-dir ftfy regex

# 9) 소스 복사
COPY . .

# (선택) 빌드 단계에서 mmcv ops 확인하고 싶으면 주석 해제
RUN python -c "import mmcv; import mmcv.ops; import mmseg; print('OK')"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
