# 미리 추출된 과목 정보 키워드와 사용자가 입력한 키워드와의 유사도 계산
# test.py에서 153줄 5. 에 해당

import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from app.loader import load_model
from app.cache import PROCESSED_COURSES
import re

logger = logging.getLogger(__name__)

model = load_model()

""" 
미리 추출된 과목 키워드와 단일 사용자 키워드와의 유사도 계산 함수 
"""
def recommend_courses(prefer_keyword: str):
    # 단일 키워드 임베딩 부분
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

        # 과목 이름에 대해 벡터 추출 및 전처리 진행 (과목 이름도 키워드로 사용)
        course_name = re.sub(r'\(.*?\)', '', course["courseName"]).rstrip()
        try:
            keyword_vecs.append(model.get_word_vector(course_name))
            valid_keywords.append(course_name)
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
            #과목 이름의 경우 가중치 2.00를 부여
            if word == course_name:
                weight = 2.00
            #freq = course.get("word_frequencies", {}).get(word, 1) # 빈도수 포함 계산
            #freq가 큰 단어는 가중치도 일반적으로 크게 부여되는 경향이 있음. '알고리즘' 과목의 경우 '알고리즘' 단어 반복이 많아 지나치게 큰 비중을 차지하는 경우가 발생함. 따라서 빈도수는 고려하지 않고 freq는 1로 고정
            freq = 1
            #logger.info(f"[과목명] {course['courseName']}, {word} : {weight}, frequency: {freq}")
            adjusted_weight = weight * freq

            #기존엔 weighted_sum += adjusted_weight * weight * sim_scores[i] 식으로 되어있어서 weighted_sum에 weight가 제곱해서 반영되는 문제가 있었음. 이미 adjusted_weight엔 weight가 반영된 값임.
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

""" 
사용자 입력 키워드를 기반 recommend_coursed 결과 통합, 유사도 높은 과목만 필터링해서 반환 
"""
def recommend_by_keywords(user_keywords: list, filtered_lectures: list, top_k=10): 
    keyword_results = {}

    for keyword in user_keywords:
        # 사용자 키워드가 여러개 -> 각 키워드마다 recommend_courses 함수 실행
        single_results = recommend_courses(keyword) 
        for course in single_results:
            key = course["courseName"]
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
        course_name = course.get("courseName", "")
        entry = keyword_results.get(course_name)

        # 평균 유사도 계산해서 반환
        if entry:
            final_results.append({
                "courseName": course_name,
                "lectureId": course.get("lectureId"),
                "aiDescription": course.get("aiDescription", ""),
                "score": round(entry["score_sum"] / entry["count"], 4)
            })

    final_results.sort(key=lambda x: x["score"], reverse=True)
    return final_results[:top_k]

