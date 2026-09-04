import joblib
import re

def clean_text(text):
    return re.sub(r"\d+", "", text).strip()

model = joblib.load("expense_model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

print("Smart Expense Categorizer")
print("Type a transaction (or 'quit' to exit)\n")

while True:
    txn = input("Transaction: ")
    if txn.lower() == "quit":
        break
    cleaned = clean_text(txn)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]
    print(f"  -> Predicted category: {prediction}\n")