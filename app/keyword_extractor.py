# 키워드 추출 함수
# test.py에서 extract_and_filter_keywords() 
# okt 객체는 내부에서 생성하도록 변경했습니다.

import logging
from krwordrank.word import KRWordRank
from konlpy.tag import Okt

logger = logging.getLogger(__name__)

def extract_keywords(text: str, percentage=0.3, min_k=10, max_k=50):
    okt = Okt() 
    pos = okt.pos(text, stem=True)
    nouns = [word for word, tag in pos if tag == "Noun"]
    joined_text = " ".join(nouns)

    if not joined_text.strip():
        logger.warning("[Okt 추출 실패] 명사 추출 결과 없음")
        return {}, [], []

    sentences = [joined_text]
    extractor = KRWordRank(min_count=1, max_length=100)

    try:
        keywords, _, _ = extractor.extract(sentences, beta=0.85, max_iter=20)
    except Exception as e:
        logger.warning(f"[KRWordRank 예외] 입력: {sentences}")
        logger.warning(f"[예외 내용] {e}")
        return {}, [], []

    final_keyword_objects = []
    final_keyword_names = []

    if keywords:
        num_ranked_keywords = len(keywords)
        n_by_percentage = round(num_ranked_keywords * percentage)
        if n_by_percentage == 0 and num_ranked_keywords > 0:
            n_by_percentage = 1
        
        desired_n = max(n_by_percentage, min_k)
        desired_n = min(desired_n, num_ranked_keywords)
        final_n = min(desired_n, max_k)

        sorted_keywords_list = sorted(keywords.items(), key=lambda x: x[1], reverse=True)
        final_keyword_objects = sorted_keywords_list[:final_n]
        final_keyword_names = [word for word, _ in final_keyword_objects]

        return keywords, final_keyword_objects, final_keyword_names

    else:
        return keywords, final_keyword_objects, final_keyword_names
