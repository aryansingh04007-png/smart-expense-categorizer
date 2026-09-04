from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from dataset import DATA
import re

def clean_text(text):

     return re.sub(r"\d+", "", text).strip()
texts = [clean_text(t) for t, label in DATA]
labels = [label for t, label in DATA]

print(f"Total messages: {len(texts)}")

X_train_text, X_test_text, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)
print(f"Train: {len(X_train_text)} | Test (unseen): {len(X_test_text)}")

vectorizer = TfidfVectorizer(stop_words="english")
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy on unseen test data: {accuracy*100:.2f}%")
print("\nClassification report:")
print(classification_report(y_test, y_pred))

import joblib

# Save the trained model and vectorizer for later use
joblib.dump(model, "expense_model.joblib")
joblib.dump(vectorizer, "vectorizer.joblib")
print("\nModel and vectorizer saved!")

new_transactions = [
    "Cred rent payment",
    "Zomato dinner order",
    "Airbnb booking payment",
    "Nykaa cosmetics order",
]

cleaned_new = [clean_text(t) for t in new_transactions]
new_vec = vectorizer.transform(cleaned_new)
predictions = model.predict(new_vec)

print("\nPredictions on brand-new transactions:")
for txn, pred in zip(new_transactions, predictions):
    print(f"  {txn} -> {pred}")

   