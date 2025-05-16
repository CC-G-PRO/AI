# Python 3.10 기반 슬림 이미지 사용
FROM python:3.10-slim

# 시스템 패키지 설치: JVM, C 컴파일러& FastText 모델 다운로드
RUN apt-get update && apt-get install -y \
    default-jdk \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/* 

# JAVA 환경 변수 등록
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH

# 작업 디렉토리 지정
WORKDIR /app

# 전체 코드 복사
COPY . .

# FastText 모델 복사
COPY model_fasttext /app/model_fasttext

# pip 최신화 및 Python 의존성 설치
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# FastAPI 서버 포트
EXPOSE 8000

# 컨테이너 실행 시 자동 시작될 명령어
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

