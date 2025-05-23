# 사용자 입력과 비교하기 전, 각 course에서 미리 키워드를 추출
# test.py에서 107줄 4. 에 해당

import logging
from app.loader import load_courses
from app.keyword_extractor import extract_keywords

logger = logging.getLogger(__name__)

def build_processed_courses():
    courses = load_courses()
    processed = []
    logger.info("[키워드 전처리 시작]")
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

PROCESSED_COURSES = build_processed_courses()