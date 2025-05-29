import os
import logging
from app.loader import load_courses
from app.keyword_extractor import extract_keywords

logger = logging.getLogger(__name__)


# 내부 테스트 여부 판별
USE_INTERNAL_TEST = os.environ.get("INTERNAL_TEST", "true").lower() == "true"

def build_processed_courses(courses=None):
    if courses is None:
        if USE_INTERNAL_TEST:
            logger.info("[테스트용] courses.json에서 과목 데이터 로딩")
            courses = load_courses()
        else:
            raise ValueError("API 연동 시에는 courses 인자를 반드시 전달해야합니다.")
        
    processed = []
    for idx, course in enumerate(courses):
        name = course.get("courseName", f"N/A_{idx}")
        desc = course.get("aiDescription", "")

        if not desc:
            logger.warning(f"[{name}] 설명 없음 - 건너뜀")
            continue

        all_keywords, filtered_keyword_objects, filtered_keyword_names = extract_keywords(desc)
        if not filtered_keyword_names:
            logger.warning(f"[{name}] 키워드 추출 실패")
            continue

        # 등장 횟수 기반 유사도 강화
        word_frequencies = {}
        desc_lower = desc.lower()
        for word in filtered_keyword_names:
            word_frequencies[word] = desc_lower.count(word.lower())     

        processed.append({
            "lectureId": course.get("lectureId", f"UNK_{idx}"),
            "courseName": name,
            "all_keywords": all_keywords,
            "filtered_keywords_objects": filtered_keyword_objects,
            "filtered_keyword_names": filtered_keyword_names,
            "word_frequencies": word_frequencies
        })
    logger.info(f"[키워드 전처리 완료] 처리된 과목 수: {len(processed)}")
    return processed

# 내부 테스트인 경우만 캐시 미리 생성
PROCESSED_COURSES = build_processed_courses() if USE_INTERNAL_TEST else None