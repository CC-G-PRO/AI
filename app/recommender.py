# 미리 추출된 과목 정보 키워드와 사용자가 입력한 키워드와의 유사도 계산
# test.py에서 153줄 5. 에 해당

import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from app.loader import load_model
from app.cache import PROCESSED_COURSES

logger = logging.getLogger(__name__)

model = load_model()

def recommend_courses(prefer_keyword: str):
    # 사용자 입력 키워드 임베딩 부분
    try :
        prefer_vec = model.get_word_vector(prefer_keyword).reshape(1, -1)
    except Exception as e:
        logger.error(f"[FastText 오류] '{prefer_keyword}' 임베딩 실패: {e}")
        results = []
    
    results = []

    for course in PROCESSED_COURSES:
        keyword_vecs = []
        valid_keywords = []

        # 필터링된 키워드에 대해 벡터 추출
        for word in course["filtered_keyword_names"]:
            try:
                keyword_vecs.append(model.get_word_vector(word))
                valid_keywords.append(word)
            except Exception as e:
                logger.warning(f"[임베딩 실패] '{word}': {e}")
                continue

        if not keyword_vecs:
            continue
        
        # cosine similarity 계산
        sim_scores = cosine_similarity(prefer_vec, np.array(keyword_vecs))[0]

        weighted_sum = 0
        total_weight = 0

        # kRWordRank 점수 * 유사도 합산
        for i, word in enumerate(valid_keywords):
            weight = course["all_keywords"].get(word, 0)
            freq = course.get("word_frequencies", {}).get(word, 1) # 빈도수 포함 계산
            adjusted_weight = weight * freq

            weighted_sum += adjusted_weight*weight * sim_scores[i]
            total_weight += adjusted_weight

        # 평균 가중 유사도 계산
        avg_weighted_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        results.append({
            "course_name": course["course_name"],
            "subject_code": course["subject_code"],
            "score": round(float(avg_weighted_score), 4),
            "num_keywords": len(valid_keywords)
        })

    # 유사도 기준 정렬
    results.sort(key=lambda x: x["score"], reverse=True)
    return results
