import os
import json
from tqdm import tqdm
from nltk.corpus import wordnet as wn
from sentence_transformers import SentenceTransformer, util

# BERT 모델 로드
model = SentenceTransformer('all-MiniLM-L6-v2')

# 처리할 JSON 파일 목록
file_names = [
    "bears_taco_menu.json",
    "byeoli_dali_menu.json",
    "campus_toast_menu.json",
    "insang_menu.json",
    "jesoon_menu.json",
    "little_hanoi_menu.json",
    "onigiri_and_lee_gyudong_menu.json",
    "pulbitmaru_menu.json",
    "rolling_pasta_menu.json",
    "subway_15cm_menu.json",
    "the_big_lunch_box_menu_full.json",
    "well_chai_menu.json",
    "yeokjeon_menu.json"
]

# context 문장 정의
context_sentences = {
    'temperature': 'This refers to the serving temperature of the food.',
    'taste': 'This describes the flavor profile of the food.',
    'ingredients': 'These are the ingredients used in the food.',
    'cuisine': 'This is the type of cuisine or cooking style.',
    'allergens': 'These are substances in food that may cause allergic reactions.'
}

# synset 추출 함수
def get_best_synset_by_bert(word, context_sentence):
    synsets = wn.synsets(word)
    if not synsets:
        return word
    context_emb = model.encode(context_sentence, convert_to_tensor=True)
    scored = []
    for s in synsets:
        def_emb = model.encode(s.definition(), convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(context_emb, def_emb).item()
        scored.append((s, similarity))
    best = max(scored, key=lambda x: x[1])
    return best[0].name()

# 경로 설정
input_dir = "./KAIST_MENUS/PROCESSED_JSON/"
output_dir = "./KAIST_MENUS/MATCHED_JSON/"
os.makedirs(output_dir, exist_ok=True)

# 각 파일 처리
for file_name in file_names:
    input_path = os.path.join(input_dir, file_name)
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in tqdm(data, desc=f"Processing {file_name}", leave=False):
        # temperature
        item['temperature'] = get_best_synset_by_bert(item['temperature'], context_sentences['temperature'])

        # taste
        taste_words = [t.strip() for t in item['taste']] if isinstance(item['taste'], list) else [t.strip() for t in item['taste'].split(',')]
        item['taste'] = [get_best_synset_by_bert(t, context_sentences['taste']) for t in taste_words]

        # ingredients
        item['ingredients'] = [get_best_synset_by_bert(ing, context_sentences['ingredients']) for ing in item['ingredients']]

        # allergens
        item['allergens'] = [get_best_synset_by_bert(a, context_sentences['allergens']) for a in item.get('allergens', [])]

    output_path = os.path.join(output_dir, file_name.replace(".json", "_with_synset.json"))
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

print("✅ Complete!")
