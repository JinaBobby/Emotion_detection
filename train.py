import re
from datasets import load_dataset
import pickle

def clean_text(text):
    text = text.lower()  
    text = re.sub(r"http\S+|@\w+", "", text)  
    text = re.sub(r"[^a-z\s]", " ", text) 
    tokens = text.split()  
    return " ".join(tokens)

# Load dataset
dataset = load_dataset("dair-ai/emotion")

# Clean train and test texts
X_train_cleaned = [clean_text(text) for text in dataset["train"]["text"]]
X_test_cleaned = [clean_text(text) for text in dataset["test"]["text"]]

y_train = dataset["train"]["label"]
y_test = dataset["test"]["label"]

print("Cleaned Training Texts:", X_train_cleaned[:5])
print("Cleaned Test Texts:", X_test_cleaned[:5])
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib


vectorizer = TfidfVectorizer()
X_train_vect = vectorizer.fit_transform(X_train_cleaned)
X_test_vect = vectorizer.transform(X_test_cleaned)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vect, y_train)

# Evaluate
y_pred = model.predict(X_test_vect)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
with open("emotion_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
