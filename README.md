# 🍝 NLTK-based Food Recommendation System

This food recommendation system uses the **NLTK library** and a structured **JSON food database** to recommend meals based on user input. The system avoids deep learning models (e.g., BERT) and instead relies on traditional NLP techniques.

---

## ✅ System Pipeline

```text
[1] User Input (Natural Language)
        ↓
[2] Preprocessing with NLTK:
     - Tokenization
     - Stopword Removal
     - POS Tagging (adjectives, nouns)
     - Lemmatization
        ↓
[3] Keyword Extraction (e.g., "spicy", "noodle", "light")
        ↓
[4] Load Food Database (JSON)
        ↓
[5] Attribute Matching:
     - Flatten food attributes
     - Jaccard similarity between user keywords and food attributes
        ↓
[6] Top-N Food Recommendation
```

---

## 🔍 Example JSON Schema

```json
{
  "bibimbap": {
    "temperature": "hot",
    "taste": ["spicy", "savory"],
    "ingredients": ["rice", "vegetable", "egg"],
    "tags": ["healthy", "filling"],
    "nutrition": {
      "carbs": 80,
      "protein": 25,
      "fat": 18,
      "calories": 700
    }, 
    "location": "North", 
    "Restaurant Name": "Choi's Restaurant"
  }
}
```

---

## ⚖️ Similarity Calculation: Jaccard Index

\[
\text{Jaccard}(A, B) = \frac{|A \cap B|}{|A \cup B|}
\]

Where:
- \( A \): user keyword set
- \( B \): flattened set of food attributes

---

## ⚡ Optional Nutrition Filtering
- Calorie thresholds (e.g., under 500 kcal)
- Minimum protein level
- Filter by fat or carbs

---

## 🔄 Strengths
- Lightweight and interpretable
- No training required
- Flexible food database expansion

## ❌ Limitations
- Keyword-only understanding (no deep semantics)
- Sensitive to input phrasing
- Requires well-structured JSON data

---

## 🔧 Technology Stack
- Python
- NLTK
- JSON
- Jaccard similarity
