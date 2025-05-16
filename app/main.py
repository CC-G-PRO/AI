# Fast API 서버 진입점

import logging
from fastapi import FastAPI, Query
from app.recommender import recommend_courses

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.get("/recommended-courses")
def recommend(keyword: str = Query(..., description="키워드")):
    logger.info(f"[요청] 추천 키워드: {keyword}")
    results = recommend_courses(keyword)
    logger.info(f"[응답] 추천 과목 수: {len(results)}")
    return {
        "msg": "추천 과목 리스트 생성 성공",
        "recommended": results
    }
