import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from app.loader import load_model
from app.cache import PROCESSED_COURSES, build_processed_courses
import re

logger = logging.getLogger(__name__)

model = load_model()


def recommend_courses(prefer_keyword: str, processed_courses: list):
    try :
        prefer_vec = model.get_word_vector(prefer_keyword).reshape(1, -1)
    except Exception as e:
        logger.error(f"[FastText 오류] '{prefer_keyword}' 임베딩 실패: {e}")
        return []
    
    results = []

    for course in processed_courses:
        keyword_vecs = []
        valid_keywords = []

        # 키워드 벡터 추출
        for word in course["filtered_keyword_names"]:
            try:
                keyword_vecs.append(model.get_word_vector(word))
                valid_keywords.append(word)
            except Exception as e:
                logger.warning(f"[임베딩 실패] '{word}': {e}")
                continue

        # 과목 이름도 키워드로 사용
        course_name = re.sub(r'\(.*?\)', '', course["courseName"]).rstrip()
        try:
            keyword_vecs.append(model.get_word_vector(course_name))
            valid_keywords.append(course_name)
        except Exception as e:
            logger.warning(f"[임베딩 실패] '{course_name}': {e}")
            continue

        if not keyword_vecs:
            logger.info(f"[{course_name}] 유효한 키워드가 없어 추천 제외")
            continue
        
        # cosine similarity 계산
        sim_scores = cosine_similarity(prefer_vec, np.array(keyword_vecs))[0]

        weighted_sum = 0
        total_weight = 0

        # kRWordRank 점수 * 유사도 합산
        for i, word in enumerate(valid_keywords):
            weight = course["all_keywords"].get(word, 0) 

            #과목 이름의 경우 가중치 2.00를 부여
            if word == course_name:
                weight = 2.00

            #freq가 큰 단어는 가중치도 일반적으로 크게 부여되는 경향이 있음. '알고리즘' 과목의 경우 '알고리즘' 단어 반복이 많아 지나치게 큰 비중을 차지하는 경우가 발생함. 따라서 빈도수는 고려하지 않고 freq는 1로 고정
            freq = 1
            adjusted_weight = weight * freq
            weighted_sum += adjusted_weight * sim_scores[i]
            total_weight += adjusted_weight

        # 평균 가중 유사도 계산
        avg_weighted_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        results.append({
            "courseName": course["courseName"],
            "lectureId": course["lectureId"],
            "score": round(float(avg_weighted_score), 4),
            "num_keywords": len(valid_keywords)
        })

    # 유사도 기준 정렬
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def recommend_by_keywords(user_keywords: list, filtered_lectures: list, top_k: int = 100): 

    processed_courses = PROCESSED_COURSES if PROCESSED_COURSES else build_processed_courses(filtered_lectures)

    keyword_results = {}

    for keyword in user_keywords:

        # 사용자 키워드가 여러개 -> 각 키워드마다 recommend_courses 함수 실행
        single_results = recommend_courses(keyword, processed_courses) 
        for course in single_results:
            key = course["lectureId"]

            # 처음 과목이 추천된 경우 
            if key not in keyword_results: 
                keyword_results[key] = {
                    "courseName": course["courseName"],
                    "lectureId": course["lectureId"],
                    "score_sum": 0.0,
                    "num_keywords": course["num_keywords"],
                    "count": 0
                }
            # 다른 키워드에서 동일한 과목 추천된 경우, 두 점수 합산    
            keyword_results[key]["score_sum"] += course["score"] 
            keyword_results[key]["count"] += 1

    # api서버에서 받은 과목 리스트 필터링 -> recommend_courses로 유사도 계산 안 된 과목 필터링
    final_results = []
    for course in filtered_lectures:
        lecture_id = course.get("lectureId")
        entry = keyword_results.get(lecture_id)
        # 평균 유사도 계산해서 반환
        if entry:
            final_results.append({
                "courseName": course.get("courseName"),
                "lectureId": lecture_id,
                "aiDescription": course.get("aiDescription", ""),
                "score": round(entry["score_sum"] / entry["count"], 4)
            })

    final_results.sort(key=lambda x: x["score"], reverse=True)
    return final_results[:top_k]

