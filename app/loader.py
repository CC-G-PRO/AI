import boto3
import os
import logging
import boto3.session
import fasttext
import json
from dotenv import load_dotenv
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError


FASTTEXT_MODEL_PATH = "model_fasttext/cc.ko.300.bin"
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_OBJECT_KEY = os.getenv("S3_OBJECT_KEY")

COURSES_JSON_PATH = "courses.json" 


logger = logging.getLogger(__name__)

def download_model_from_s3():
    if os.path.exists(FASTTEXT_MODEL_PATH):
        logger.info(f"FastText 모델이 이미 존재: {FASTTEXT_MODEL_PATH} (S3 다운로드 생략)")
        return 
    
    logger.info("[FastText] 모델이 존재하지 않아 S3에서 다운로드를 시작합니다.")
    logger.info(f"[FastText] 모델 다운로드 경로: s3: //{S3_BUCKET_NAME}/{S3_OBJECT_KEY} -> {FASTTEXT_MODEL_PATH}")
    os.makedirs(os.path.dirname(FASTTEXT_MODEL_PATH), exist_ok=True)

    try:
        session = boto3.session.Session()
        s3 = session.client("s3", region_name=os.getenv('AWS_REGION', "eu-north-1"))

        credentials = session.get_credentials()
        current_region = session.region_name
        logger.info(f"🔐 AWS Region: {current_region}")
        logger.info(f"🔐 AWS Credentials: {credentials.get_frozen_credentials() if credentials else 'None'}")

        s3.download_file(S3_BUCKET_NAME, S3_OBJECT_KEY, FASTTEXT_MODEL_PATH)
        logger.info("[FastText] S3에서 모델 다운로드 완료")
    except NoCredentialsError:
        logger.error("❌ AWS 인증 정보를 찾을 수 없습니다. IAM 역할 또는 ~/.aws/credentials 확인 필요")
        raise
    except PartialCredentialsError as e:
        logger.error(f"❌ AWS 인증 정보가 불완전합니다: {e}")
        raise
    except ClientError as e:
        logger.error(f"❌ S3 클라이언트 오류: {e.response['Error']['Message']}")
        raise
    except Exception as e:
        logger.error(f"[FastText] S3 다운로드 실패: {e}")
        raise

def load_model():
    if not os.path.exists(FASTTEXT_MODEL_PATH):
        logger.warning(f"FastText 모델이 없어 자동 다운로드를 시도합니다: {FASTTEXT_MODEL_PATH}")
        download_model_from_s3()

         
        if not os.path.exists(FASTTEXT_MODEL_PATH):
            logger.error("FastText 모델 다운로드 실패 또는 경로에 없음")
            raise FileNotFoundError(f"{FASTTEXT_MODEL_PATH} not found after download")

    try:
        logger.info("FastText 모델 로딩 중...")
        model = fasttext.load_model(FASTTEXT_MODEL_PATH)
        logger.info(f"FastText 모델 로드 완료. 벡터 차원: {model.get_dimension()}")
        return model
    except Exception as e:
        logger.error(f"FastText 모델 로딩 실패: {e}")
        raise

def load_courses(from_api=None):
    if not os.path.exists(COURSES_JSON_PATH):
        logger.error(f"[테스트용] courses.json 파일이 존재하지 않습니다: {COURSES_JSON_PATH}")
        raise FileNotFoundError(f"courses.json 파일이 존재하지 않습니다: {COURSES_JSON_PATH}")
    try:
        with open(COURSES_JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("courses.json의 최상위 데이터는 리스트여야 합니다.")
        logger.info(f"[테스트용] 과목 데이터 {len(data)}건 로드 완료.")
        return data
    except Exception as e:
        logger.error(f"[테스트용] courses.json 로드 실패: {e}")
        raise