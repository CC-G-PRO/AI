# Fast API 서버 진입점

import logging
from typing import List, Optional
from fastapi import FastAPI, Query
from pydantic import BaseModel

from app.recommender import recommend_courses, recommend_by_keywords

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# 내부 테스트용 추후에 삭제 예정
@app.get("/recommended-courses")
def recommend(keyword: str = Query(..., description="키워드")):
    logger.info(f"[요청] 추천 키워드: {keyword}")
    results = recommend_courses(keyword)
    logger.info(f"[응답] 추천 과목 수: {len(results)}")
    return {
        "msg": "추천 과목 리스트 생성 성공",
        "recommended": results
    }

# AiClient에 맞춰서 수정
class FilteredLecture(BaseModel):
    lectureId: str
    courseName: str
    aiDescription: str
    description: Optional[str] = None  # Spring 구조에 맞춰 optional 필드 추가

class AiFilterRequest(BaseModel):
    userWantedKeywords: List[str]
    filteredLectures: List[FilteredLecture]

class AiFilterResponse(BaseModel):
    userWantedKeywords: List[str]
    filteredLectures: List[FilteredLecture]

@app.post("/recommend-api", response_model=AiFilterResponse)
def recommend_api(request: AiFilterRequest):
    logger.info(f"[POST 요청] 키워드: {request.userWantedKeywords}, 강의 수:{len(request.filteredLectures)}")

    filtered = recommend_by_keywords(
        user_keywords=request.userWantedKeywords,
        filtered_lectures=[lecture.dict() for lecture in request.filteredLectures]
    )

    return {
        "userWantedKeywords": request.userWantedKeywords,
        "filteredLectures": filtered
        }
