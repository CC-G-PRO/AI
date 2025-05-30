import logging
import os
from dotenv import load_dotenv
from typing import List, Optional
from fastapi import FastAPI, Query
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi import HTTPException
from app.cache import PROCESSED_COURSES

from app.recommender import recommend_courses, recommend_by_keywords

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

USE_INTERNAL_TEST = os.environ.get("INTERNAL_TEST", "true").lower() == "true"

app = FastAPI()

# 내부 테스트일 때만 사용
if USE_INTERNAL_TEST:
    @app.get("/recommended-courses")
    def recommend(keyword: str = Query(..., description="키워드")):
        logger.info(f"[요청] 추천 키워드: {keyword}")
        results = recommend_courses(keyword, PROCESSED_COURSES)
        logger.info(f"[응답] 추천 과목 수: {len(results)}")
        return {
            "msg": "추천 과목 리스트 생성 성공",
            "recommended": results
        }

class FilteredLecture(BaseModel):
    lectureId: str
    courseName: str
    aiDescription: str
    #description: Optional[str] = None  

class AiFilterRequest(BaseModel):
    userWantedKeywords: List[str]
    filteredLectures: List[FilteredLecture]

class ScoredLecture(FilteredLecture):
    score: float

class AiFilterResponse(BaseModel):
    userWantedKeywords: List[str]
    filteredLectures: List[ScoredLecture]

@app.post("/recommend-api", response_model=AiFilterResponse)
def recommend_api(request: AiFilterRequest):

    try:
        logger.info(f"[POST 요청] 키워드: {request.userWantedKeywords}, 강의 수:{len(request.filteredLectures)}")

        filtered = recommend_by_keywords(
            user_keywords=request.userWantedKeywords,
            filtered_lectures=[lecture.dict() for lecture in request.filteredLectures]
        )

        return {
            "userWantedKeywords": request.userWantedKeywords,
            "filteredLectures": filtered
        }
    except Exception as e:
        logger.exception("[API 오류] 추천 처리 중 에러 발생")
        raise HTTPException(status_code=500, detail="추천 처리 중 오류가 발생했습니다.")

