# Python 3.10-slim 이미지를 기반으로 시작
FROM python:3.10-slim

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 깃 레포지토리에서 프로젝트를 복사
COPY . .

# Python 패키지를 설치 및 Uvicorn 설치
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install uvicorn

# 애플리케이션 포트를 외부에 노출
EXPOSE 8000

# Uvicorn 서버를 시작
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
