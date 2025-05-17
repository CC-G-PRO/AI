# fastttext 모델, courses.json 로더

import gzip
import shutil
import os
import urllib.request
from tqdm import tqdm
import logging
import fasttext
import json

FASTTEXT_MODEL_PATH = "model_fasttext/cc.ko.300.bin"
MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ko.300.bin.gz"

# 테스트 과목 수 확장
COURSES_JSON_PATH = "coursesFull.json" 

logger = logging.getLogger(__name__)

# FastText 모델 다운
def download_model():
    if os.path.exists(FASTTEXT_MODEL_PATH):
        logger.info("모델이 이미 존재, 다운로드 생략")
        return

    logger.info("[FastText] 모델이 존재하지 않아 자동 다운로드를 시작합니다...")
    os.makedirs("model_fasttext", exist_ok=True)
    zip_path = FASTTEXT_MODEL_PATH + ".gz"

    try:
        urllib.request.urlretrieve(MODEL_URL, zip_path)
        logger.info("[FastText] 다운로드 완료")

        with gzip.open(zip_path, 'rb') as f_in, open(FASTTEXT_MODEL_PATH, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

        os.remove(zip_path)
        logger.info("[FastText] 압축 해제 완료")

    except Exception as e:
        logger.error(f"[FastText] 다운로드 중 에러: {e}")
        raise

def load_model():
    if not os.path.exists(FASTTEXT_MODEL_PATH):
        logger.warning(f"FastText 모델이 없어 자동 다운로드를 시도합니다: {FASTTEXT_MODEL_PATH}")
        download_model()

         # 다운로드 실패 시도 대비, 존재 재확인
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

def load_courses():
    if not os.path.exists(COURSES_JSON_PATH):
        logger.error(f"courses.json 파일이 존재하지 않습니다: {COURSES_JSON_PATH}")
        raise FileNotFoundError(f"courses.json 파일이 존재하지 않습니다: {COURSES_JSON_PATH}")
    try:
        with open(COURSES_JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("courses.json의 최상위 데이터는 리스트여야 합니다.")
        logger.info(f"과목 데이터 {len(data)}건 로드 완료.")
        return data
    except Exception as e:
        logger.error(f"courses.json 로드 실패: {e}")
        raise