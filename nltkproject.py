import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer

import string

# Download required NLTK data (only once) and suppress the outputs, ('punkt') ('stopwords') ('wordnet') ('averaged_perceptron_tagger')

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)



def preprocess_user_input(user_input):

    # Tokenize & lowercase
    tokens = word_tokenize(user_input.lower())


    # Remove stopwords and punctuation
    custom_stopwords = {
    "want", "need", "get", "look", "looking", "like", "would", "something", "maybe", "prefer", "think", "feel", "try", "eat", "drink"}
    stop_words = set(stopwords.words('english')).union(custom_stopwords)



    filtered_tokens = [
        word for word in tokens
        if word not in stop_words and word not in string.punctuation
    ]

    # POS tagging
    tagged_tokens = pos_tag(filtered_tokens)

    # Keep only nouns and adjectives
    relevant_tokens = [
        (word, tag) for word, tag in tagged_tokens
        if tag.startswith('JJ') or tag.startswith('NN')
    ]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized = [
        lemmatizer.lemmatize(word, pos='a' if tag.startswith('JJ') else 'n')
        for word, tag in relevant_tokens
    ]

    # Remove duplicates and return
    keywords = list(set(lemmatized))
    return keywords

# Function to get synonyms for a word
def get_synonyms(word):
    synonyms = set()
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().lower())
    return synonyms


# Example usage:
if __name__ == "__main__":
    sample_input = input("Enter your text: ")
    keywords = preprocess_user_input(sample_input)
    print("Extracted Keywords:", keywords)
