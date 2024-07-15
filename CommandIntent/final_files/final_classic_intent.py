
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import string
from collections import defaultdict, Counter
import math
from sklearn.preprocessing import LabelBinarizer
import pickle
import numpy as np



# Preprocessing function
    # Convert all to lowercase
    # Remove punctuations
    # Remove stopwords

stop_words = {'a', 'the', 'is', 'it', 'to', 'of', 'in', 'for', 'on', 'with', 'as', 'at', 'by', 'from',
                 'an', 'be', 'are', 'was', 'were', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
                   'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their',
                     'mine', 'yours', 'his', 'hers', 'ours', 'theirs', 'what', 'which', 'who', 'whom', 'whose',
                       'where', 'when', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
                         'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'too', 'very'}
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    # text = ' '.join(word for word in text.split() if word not in stop_words)
    text = ' '.join(word for word in text.split())
    return text

# function save the model
def save_model(model, model_path):
	with open(model_path, 'wb') as file:
		pickle.dump(model, file)

# function load the model
def load_model(model_path):
	with open(model_path, 'rb') as file:
		model = pickle.load(file)
	return model


def load_intent_data(intent_data_path):
	# Load JSON file
	with open(intent_data_path, 'r') as file:
		data = json.load(file)
	# Convert to DataFrame
	rows = []
	for intent, sentences in data.items():
		for sentence in sentences:
			rows.append({'intent': intent, 'sentence': sentence})

	df = pd.DataFrame(rows)

	# Apply preprocessing
	df['processed_sentence'] = df['sentence'].apply(preprocess_text)

	# Display the DataFrame
	return df


df = load_intent_data('../intent_detection_dataset/final_intents_dataset.json')


# Extract N-grams
def extract_ngrams(text, n=3):
    words = text.split()
    ngrams = []
    for i in range(1, n + 1):
        ngrams += [' '.join(words[j:j+i]) for j in range(len(words) - i + 1)]
    return ngrams

# Compute term frequencies
def compute_term_frequencies(corpus):
    term_frequencies = defaultdict(Counter)
    for idx, document in enumerate(corpus):
        ngrams = extract_ngrams(document)
        term_frequencies[idx] = Counter(ngrams)
    return term_frequencies

# Compute document frequencies
def compute_document_frequencies(term_frequencies):
    document_frequencies = Counter()
    for tf in term_frequencies.values():
        for term in tf.keys():
            document_frequencies[term] += 1
    return document_frequencies

# Compute TF-IDF
def compute_tfidf(corpus):
    term_frequencies = compute_term_frequencies(corpus)
    document_frequencies = compute_document_frequencies(term_frequencies)
    N = len(corpus)
    tfidf = defaultdict(dict)
    for idx, tf in term_frequencies.items():
        for term, count in tf.items():
            tfidf[idx][term] = (count / len(tf)) * math.log(N / (document_frequencies[term] + 1))
    return tfidf, document_frequencies

# Create TF-IDF matrix
def create_tfidf_matrix(tfidf, vocabulary):
    num_docs = len(tfidf)
    num_terms = len(vocabulary)
    # Initialize the matrix with zeros
    tfidf_matrix = np.zeros((num_docs, num_terms))
    # Populate the matrix with TF-IDF scores
    for doc_idx, term_scores in tfidf.items():
        for term, score in term_scores.items():
            term_idx = vocabulary.get(term)
            if term_idx is not None:
                tfidf_matrix[doc_idx, term_idx] = score
    return tfidf_matrix

# Prepare corpus
corpus = df['processed_sentence'].tolist()

# Compute TF-IDF scores
tfidf, document_frequencies  = compute_tfidf(corpus)

# Create vocabulary from all N-grams
all_ngrams = set()
for tf in tfidf.values():
    all_ngrams.update(tf.keys())

vocabulary = {term: idx for idx, term in enumerate(all_ngrams)}

# Create TF-IDF matrix
X = create_tfidf_matrix(tfidf, vocabulary)
y = df['intent']

# %%
# Encode labels = 24
labels = df['intent'].unique()
label_to_index = {label: idx for idx, label in enumerate(labels)}
index_to_label = {idx: label for label, idx in label_to_index.items()}
y = np.array([label_to_index[label] for label in y])


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# %% [markdown]
# # Model and training

# %%
# Function to predict intent of custom input
def predict_intent(custom_sentence, model):
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
    predicted_intent_idx = model.predict(np.array([custom_tfidf]))[0]
    predicted_intent = index_to_label[predicted_intent_idx]
    return predicted_intent


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class MulticlassLogisticRegression:
    def __init__(self, lr: float, epochs: int, probability_threshold: float = 0.5, random_state=None):
        self.lr = lr  # The learning rate
        self.epochs = epochs  # The number of training epochs
        self.probability_threshold = probability_threshold  # Threshold for classification
        self.random_state = random_state  # Seed for reproducibility
        self.weights = []  # Store weights for each class

    def _prepare_input(self, X):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        # Add a new input with value 1 to each example (bias term)
        ones = np.ones((X.shape[0], 1), dtype=X.dtype)
        return np.concatenate((ones, X), axis=1)

    def _initialize(self, num_weights: int, stdev: float = 0.01):
        # Initialize the weights using a normally distributed random variable with a small standard deviation
        np.random.seed(self.random_state)
        return np.random.randn(num_weights) * stdev

    def _gradient(self, X, y, weights):
        # Compute and return the gradient of the weights with respect to the loss given X and y
        error = y - sigmoid(np.dot(X, weights))
        weight_gradient = np.dot(-X.T, error)
        return weight_gradient

    def _update(self, X, y, weights):
        # Apply a single iteration on the weights
        gradient = self._gradient(X, y, weights)
        weights -= self.lr * gradient
        return weights

    def fit(self, X, y):
        X = self._prepare_input(X)
        encoder = LabelBinarizer()
        y_oh = encoder.fit_transform(y)

        for i in range(y_oh.shape[1]):
            weights = self._initialize(X.shape[1])
            for _ in range(self.epochs):
                weights = self._update(X, y_oh[:, i], weights)
            self.weights.append(weights)
        return self

    def predict_proba(self, X):
        X = self._prepare_input(X)
        probas = []
        for weights in self.weights:
            proba = sigmoid(np.dot(X, weights))
            probas.append(proba)
        return np.array(probas).T

    def predict(self, X):
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)
    

model = MulticlassLogisticRegression(lr=0.1, epochs=500, probability_threshold=0.5, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))



import pickle
with open('logisticRegressionModel.pkl', 'wb') as file:
	pickle.dump(model, file)

#Load the model
with open('logisticRegressionModel.pkl', 'rb') as file:
	model = pickle.load(file)



sentences = ["select lines 1 through 5",
    "select line 9",
    "highlight from line 3 to line 7",
    "go to line 10",
    "delete line 6",
    "delete lines 2 through 4",
    "focus terminal",
    "kill terminal",
    "open terminal",
    "new terminal",
    "undo",
    "redo",
    "copy",
    "paste"]


for sent in sentences:
	predicted_intent = predict_intent(sentences, model)
	print(f"Predicted intent: {predicted_intent}\n")


