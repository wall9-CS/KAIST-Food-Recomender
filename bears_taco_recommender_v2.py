
import re
import json
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# 샘플 데이터 로드
with open("KAIST_MENUS/PROCESSED/Bears Taco.json", "r", encoding="utf-8") as f:
    food_data = json.load(f)

# 유사도 계산
def jaccard_similarity(set1, set2):
    return len(set1 & set2) / len(set1 | set2)

# 음식 속성 키워드 집합 생성
def flatten_keywords(info):
    return set(
        info["taste"] + info["ingredients"] + info["tags"] + [info["temperature"]]
    )

# POS 변환 도우미
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# 강화된 키워드 추출 함수
def extract_keywords(query):
    stop_words = set(stopwords.words("english"))
    domain_stopwords = {"want", "something", "like", "food", "dish"}
    stop_words.update(domain_stopwords)

    lemmatizer = WordNetLemmatizer()
    tokens = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())
    filtered = [w for w in tokens if w not in stop_words]
    tagged = pos_tag(filtered)

    keywords = []
    for word, tag in tagged:
        wn_tag = get_wordnet_pos(tag)
        lemma = lemmatizer.lemmatize(word, wn_tag)
        if wn_tag in [wordnet.NOUN, wordnet.ADJ, wordnet.VERB]:
            keywords.append(lemma)

    print(f"Extracted keywords: {keywords}")
    return set(keywords)

# 메뉴 추천 함수
def recommend_menu(query, food_data, max_price=None, min_protein=None, max_calories=None):
    user_keywords = extract_keywords(query)
    results = []

    for name, info in food_data.items():
        if max_price and info["price"] > max_price:
            continue
        if min_protein and info["nutrition"]["protein"] < min_protein:
            continue
        if max_calories and info["nutrition"]["calories"] > max_calories:
            continue

        food_keywords = flatten_keywords(info)
        similarity = jaccard_similarity(user_keywords, food_keywords)
        results.append((name, similarity))

    results.sort(key=lambda x: x[1], reverse=True)
    return results

# CLI 실행
if __name__ == "__main__":
    import nltk
    nltk.download("stopwords")
    nltk.download("wordnet")
    nltk.download("averaged_perceptron_tagger")

    print("🍽 Welcome to Bears Taco Recommender!")
    query = input("Describe the kind of food you want: ")
    price_limit = input("Maximum price (or leave blank): ")
    protein_min = input("Minimum protein (or leave blank): ")
    calories_max = input("Maximum calories (or leave blank): ")

    price_limit = int(price_limit) if price_limit.strip().isdigit() else None
    protein_min = int(protein_min) if protein_min.strip().isdigit() else None
    calories_max = int(calories_max) if calories_max.strip().isdigit() else None

    results = recommend_menu(
        query,
        food_data,
        max_price=price_limit,
        min_protein=protein_min,
        max_calories=calories_max
    )

    print("\n🥗 Top Menu Recommendations:")
    for name, score in results[:3]:
        print(f"- {name} (similarity: {score:.2f})")
