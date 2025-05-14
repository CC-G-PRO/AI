import json
import os
import re
from krwordrank.word import KRWordRank
from konlpy.tag import Okt
import fasttext
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- 설정값 ---
FASTTEXT_MODEL_PATH = 'model_fasttext/cc.ko.300.bin'
COURSES_JSON_PATH = 'courses.json'

# --- 유틸리티 함수 ---
def extract_and_filter_keywords(text_content, okt_tokenizer, percentage=0.30, min_k=10, max_k=50):
    """주어진 텍스트에서 KRWordRank 키워드를 추출하고 필터링합니다."""
    if not text_content:
        return {}, [], []

    wordrank_extractor = KRWordRank(min_count=1, max_length=100) # max_length는 일단 크게 유지

    sentence_pos = okt_tokenizer.pos(text_content, stem=True)
    # !!! 한국어 필터링 부분 제거 !!!
    # 이전: nouns = [word for word, pos in sentence_pos if pos == 'Noun' and re.fullmatch(r'[가-힣]+', word)]
    nouns = [word for word, pos in sentence_pos if pos == 'Noun'] # Okt가 명사로 판단한 모든 단어 사용
    print(nouns)
    result_text = ' '.join(nouns)
    
    if not result_text.strip(): # 명사 추출 후 결과가 비어있으면
        return {}, [], []

    noun_single_text_list = [result_text]
    
    keywords_dict = {}
    try:
        extracted_keywords, _, _ = wordrank_extractor.extract(noun_single_text_list, beta=0.85, max_iter=20)
        if extracted_keywords:
            keywords_dict = extracted_keywords
    except Exception as e:
        # 이 부분은 오류 발생 시 디버깅에 유용하므로 유지하는 것이 좋습니다.
        print(f"DEBUG: KRWordRank 실행 중 오류 발생 (입력: {noun_single_text_list})")
        print(f"DEBUG: 발생한 예외: {e}")
        return {}, [], []

    final_keyword_objects = []
    final_keyword_names = []

    if keywords_dict:
        num_ranked_keywords = len(keywords_dict)
        if num_ranked_keywords == 0:
             return keywords_dict, [], []

        n_by_percentage = round(num_ranked_keywords * percentage)
        if n_by_percentage == 0 and num_ranked_keywords > 0:
            n_by_percentage = 1

        desired_n = max(n_by_percentage, min_k)
        desired_n = min(desired_n, num_ranked_keywords)
        final_n = min(desired_n, max_k)

        sorted_keywords_list = sorted(keywords_dict.items(), key=lambda x: x[1], reverse=True)
        final_keyword_objects = sorted_keywords_list[:final_n]
        final_keyword_names = [word for word, score in final_keyword_objects]
    
    return keywords_dict, final_keyword_objects, final_keyword_names

# --- 메인 로직 ---
def main():
    # 1. fastText 모델 로드 (최초 1회)
    embedding_model = None
    if not os.path.exists(FASTTEXT_MODEL_PATH):
        print(f"오류: fastText 모델 파일 경로를 찾을 수 없습니다: {FASTTEXT_MODEL_PATH}")
        print("FASTTEXT_MODEL_PATH 변수에 올바른 경로를 설정해주세요.")
        return
    try:
        print(f"fastText 모델 로드 중... ({FASTTEXT_MODEL_PATH})")
        embedding_model = fasttext.load_model(FASTTEXT_MODEL_PATH)
        print(f"fastText 모델 로드 완료. (단어 벡터 차원: {embedding_model.get_dimension()})\n")
    except Exception as e:
        print(f"fastText 모델 로드 중 오류 발생: {e}")
        return

    # 2. Okt 초기화
    okt_tokenizer = Okt()

    # 3. courses.json 파일 로드 및 파싱
    if not os.path.exists(COURSES_JSON_PATH):
        print(f"오류: '{COURSES_JSON_PATH}' 파일을 찾을 수 없습니다.")
        return
    
    try:
        with open(COURSES_JSON_PATH, 'r', encoding='utf-8') as f:
            courses_data = json.load(f)
    except json.JSONDecodeError:
        print(f"오류: '{COURSES_JSON_PATH}' 파일이 올바른 JSON 형식이 아닙니다.")
        return
    except Exception as e:
        print(f"'{COURSES_JSON_PATH}' 파일 로드 중 오류 발생: {e}")
        return

    if not isinstance(courses_data, list):
        print(f"오류: '{COURSES_JSON_PATH}' 파일의 최상위 데이터는 리스트여야 합니다. (현재: {type(courses_data)})")
        return

    # 4. 각 course에 대해 키워드 추출 및 정보 저장
    processed_courses = []
    print("--- 각 강의 설명에 대한 키워드 추출 및 필터링 진행 ---")
    for course_idx, course in enumerate(courses_data): # 인덱스 추가 (디버깅이나 특정 강의 식별에 용이)
        course_name = course.get("course_name", f"N/A_course_{course_idx}")
        short_description = course.get("short_description", "")

        print(f"\n[ {course_name} ]")
        if not short_description:
            print("  - 설명 데이터가 없습니다.")
            processed_courses.append({
                "course_name": course_name,
                "all_keywords": {},
                "filtered_keywords_objects": [],
                "filtered_keyword_names": []
            })
            continue

        all_kws, filtered_kws_obj, filtered_kws_names = extract_and_filter_keywords(
            short_description, okt_tokenizer
        )
        
        processed_courses.append({
            "course_name": course_name,
            "all_keywords": all_kws,
            "filtered_keywords_objects": filtered_kws_obj,
            "filtered_keyword_names": filtered_kws_names
        })

        print("  --- KRWordRank 전체 키워드 ---")
        if not all_kws:
            print("    추출된 키워드가 없습니다.")
        else:
            # 모든 키워드 출력 (요청사항 반영)
            sorted_all_kws = sorted(all_kws.items(), key=lambda x:x[1], reverse=True)
            for word, r_score in sorted_all_kws[:]: # 슬라이싱 제거
                print(f'    {word:8s}:\t{r_score:.4f}')
        
        print("  --- 필터링된 키워드 ---")
        if not filtered_kws_names:
            print("    필터링 후 선택된 키워드가 없습니다.")
        else:
            for word, r_score in filtered_kws_obj:
                print(f'    {word:8s}:\t{r_score:.4f}')
    print("-" * 40)

    # 5. 사용자로부터 선호 키워드 입력받고 유사도 계산 반복
    while True:
        print("\n선호하는 키워드를 입력하세요 (종료하려면 'exit' 또는 '종료' 입력):")
        prefer_keyword_input = input("> ").strip()

        if prefer_keyword_input.lower() in ['exit', '종료']:
            print("프로그램을 종료합니다.")
            break
        
        if not prefer_keyword_input:
            print("키워드를 입력해주세요.")
            continue

        print(f"\n--- '{prefer_keyword_input}'와(과) 각 강의의 필터링된 키워드 간 코사인 유사도 및 평균 가중 유사도 ---")
        
        try:
            prefer_keyword_embedding = embedding_model.get_word_vector(prefer_keyword_input).reshape(1, -1)
        except KeyError:
            print(f"주의: '{prefer_keyword_input}' 단어는 fastText 모델의 어휘 사전에 명시적으로 존재하지 않을 수 있습니다.")
            prefer_keyword_embedding = embedding_model.get_word_vector(prefer_keyword_input).reshape(1, -1)
        except Exception as e:
            print(f"'{prefer_keyword_input}' 임베딩 생성 중 오류: {e}")
            continue

        # 과목별 평균 가중 유사도를 저장할 리스트 (정렬용)
        course_similarity_scores = []

        for course_info in processed_courses:
            course_name = course_info["course_name"]
            filtered_keyword_names = course_info["filtered_keyword_names"]
            all_keywords_for_course = course_info["all_keywords"]

            # print(f"\n[ {course_name} ]") # 개별 키워드 유사도 출력 대신 평균 점수만 출력

            if not filtered_keyword_names:
                # print("  - 비교할 필터링된 키워드가 없습니다.") # 개별 출력 안 함
                course_similarity_scores.append({"course_name": course_name, "avg_weighted_similarity": 0.0, "num_keywords": 0})
                continue

            target_keyword_embeddings = []
            valid_target_keyword_names = []

            for keyword_name in filtered_keyword_names:
                try:
                    target_keyword_embeddings.append(embedding_model.get_word_vector(keyword_name))
                    valid_target_keyword_names.append(keyword_name)
                except Exception: # 간단히 처리
                    pass # 임베딩 실패한 단어는 제외

            if not target_keyword_embeddings:
                # print("  - 유효한 대상 키워드 임베딩이 없습니다.") # 개별 출력 안 함
                course_similarity_scores.append({"course_name": course_name, "avg_weighted_similarity": 0.0, "num_keywords": 0})
                continue
            
            target_keyword_embeddings_np = np.array(target_keyword_embeddings)
            
            try:
                similarities = cosine_similarity(prefer_keyword_embedding, target_keyword_embeddings_np)
            except ValueError:
                # print(f"  유사도 계산 오류 발생") # 개별 출력 안 함
                course_similarity_scores.append({"course_name": course_name, "avg_weighted_similarity": 0.0, "num_keywords": len(valid_target_keyword_names)})
                continue

            weighted_similarity_sum = 0
            num_valid_keywords_for_avg = 0

            # 개별 키워드 유사도 출력 부분 (주석 처리 또는 삭제 가능)
            # print(f"  --- '{prefer_keyword_input}' vs 필터링된 키워드 상세 유사도 ---")
            for i, keyword_name in enumerate(valid_target_keyword_names):
                similarity_score = similarities[0][i]
                krw_rank_score = all_keywords_for_course.get(keyword_name, 0)
                
                # 개별 키워드 유사도 출력 (선택 사항)
                # print(f"    '{prefer_keyword_input}' vs '{keyword_name}' (KRWRank: {krw_rank_score:.4f}):\t{similarity_score:.4f}")
                
                weighted_similarity_sum += krw_rank_score * similarity_score
                num_valid_keywords_for_avg += 1
            
            avg_weighted_similarity = 0
            if num_valid_keywords_for_avg > 0:
                avg_weighted_similarity = weighted_similarity_sum / num_valid_keywords_for_avg
            
            course_similarity_scores.append({
                "course_name": course_name,
                "avg_weighted_similarity": avg_weighted_similarity,
                "num_keywords": num_valid_keywords_for_avg
            })

        # 평균 가중 유사도 기준으로 과목 정렬 및 출력
        course_similarity_scores.sort(key=lambda x: x['avg_weighted_similarity'], reverse=True)
        
        print(f"\n--- '{prefer_keyword_input}'에 대한 과목별 평균 (KRWRank * 유사도) 점수 (높은 순) ---")
        for score_info in course_similarity_scores:
            print(f"  - {score_info['course_name']} (키워드 {score_info['num_keywords']}개): {score_info['avg_weighted_similarity']:.4f}")

if __name__ == '__main__':
    main()