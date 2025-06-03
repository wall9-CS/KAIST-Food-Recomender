
import re
import json
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

with open("KAIST_MENUS/PROCESSED/Bears Taco.json", "r", encoding="utf-8") as f:
    food_data = json.load(f)

def jaccard_similarity(set1, set2):
    return len(set1 & set2) / len(set1 | set2)

def flatten_keywords(info):
    return set(
        info["taste"] + info["ingredients"] + info["tags"] + [info["temperature"]]
    )

def extract_keywords(query):
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    tokens = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())
    words = [w for w in tokens if w not in stop_words]
    lemmatized = [lemmatizer.lemmatize(w) for w in words]
    print(lemmatized)
    return set(lemmatized)

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

if __name__ == "__main__":
    import nltk
    nltk.download("stopwords")
    nltk.download("wordnet")

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
