import string
import json
import math
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from Logistic_Regression import CustomLogisticRegression

############ Utility Functions ############
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    # text = ' '.join(word for word in text.split() if word not in stop_words)
    text = ' '.join(word for word in text.split())
    return text

def load_data(file='../intent_detection_dataset/final_intents_dataset.json'):
    with open(file, 'r') as file:
        data = json.load(file)
    return data

def convert_to_dataFrame(data):
    rows = []
    for intent, sentences in data.items():
        for sentence in sentences:
            rows.append({'intent': intent, 'sentence': sentence})
    return pd.DataFrame(rows)

def extract_ngrams(text, n=3):
    words = text.split()
    ngrams = []
    for i in range(1, n + 1):
        ngrams += [' '.join(words[j:j+i]) for j in range(len(words) - i + 1)]
    return ngrams

def compute_term_frequencies(corpus):
    term_frequencies = defaultdict(Counter)
    for idx, document in enumerate(corpus):
        ngrams = extract_ngrams(document)
        term_frequencies[idx] = Counter(ngrams)
    return term_frequencies

def compute_document_frequencies(term_frequencies):
    document_frequencies = Counter()
    for tf in term_frequencies.values():
        for term in tf.keys():
            document_frequencies[term] += 1
    return document_frequencies

def compute_tfidf(corpus):
    term_frequencies = compute_term_frequencies(corpus)
    document_frequencies = compute_document_frequencies(term_frequencies)
    N = len(corpus)
    tfidf = defaultdict(dict)
    for idx, tf in term_frequencies.items():
        for term, count in tf.items():
            tfidf[idx][term] = (count / len(tf)) * math.log(N / (document_frequencies[term] + 1))
    return tfidf, document_frequencies

def create_tfidf_matrix(tfidf, vocabulary):
    tfidf_matrix = np.zeros((len(tfidf), len(vocabulary)))
    for doc_idx, term_scores in tfidf.items():
        for term, score in term_scores.items():
            term_idx = vocabulary.get(term)
            if term_idx is not None:
                tfidf_matrix[doc_idx, term_idx] = score
    return tfidf_matrix

# Function to predict intent of custom input
def predict_intent(custom_sentence):
    # Preprocess the custom input
    processed_sentence = preprocess_text(custom_sentence)
    # Convert to N-gram TF-IDF features
    ngrams = extract_ngrams(processed_sentence)
    custom_tfidf = np.zeros(len(vocabulary))
    term_counts = Counter(ngrams)
    N = len(corpus) + 1
    for term, count in term_counts.items():
        if term in vocabulary:
            term_idx = vocabulary[term]
            tf = count / len(ngrams)
            idf = math.log(N / (1 + document_frequencies.get(term, 0)))
            custom_tfidf[term_idx] = tf * idf
    # Predict the intent
    predicted_intent_idx = lg_model.predict([custom_tfidf])[0]
    predicted_intent = index_to_label[predicted_intent_idx]
    return predicted_intent

'''
Pipeline:
1- Load the data
2- Convert the data to a DataFrame
3- Preprocess the text
4- Compute the TF-IDF scores
5- Create a vocabulary from all N-grams
6- Create a TF-IDF matrix with N-grams
7- Encode the labels
8- Split the data into training and testing sets
9- Train a Logistic Regression model
10- Predict on test data
11- Evaluate the model
12- Test the model with custom input
'''
data = load_data()
df = convert_to_dataFrame(data)
df['processed_sentence'] = df['sentence'].apply(preprocess_text)
corpus = df['processed_sentence'].tolist()

tfidf, document_frequencies  = compute_tfidf(corpus)

bigrams = set()
for tf in tfidf.values():
    bigrams.update(tf.keys())

vocabulary = {term: idx for idx, term in enumerate(bigrams)}

X = create_tfidf_matrix(tfidf, vocabulary)
y = df['intent']

labels = df['intent'].unique()
label_to_index = {label: idx for idx, label in enumerate(labels)}
index_to_label = {idx: label for label, idx in label_to_index.items()}
y = np.array([label_to_index[label] for label in y])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ready-made Logistic Regression Model
lg_model = LogisticRegression()
lg_model.fit(X_train, y_train)

y_pred = lg_model.predict(X_test)

# Custom Logistic Regression Model

y_test_labels = [index_to_label[idx] for idx in y_test]
y_pred_labels = [index_to_label[idx] for idx in y_pred]

print("Accuracy:", accuracy_score(y_test_labels, y_pred_labels))

custom_sentence = "i want a method named ADD "
predicted_intent = predict_intent(custom_sentence)
print(f"Predicted intent for '{custom_sentence}':\n {predicted_intent}")