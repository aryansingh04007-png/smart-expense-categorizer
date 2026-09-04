# 💰 Smart Expense Categorizer (Machine Learning)

A beginner‑friendly **machine learning project** that classifies transaction texts (e.g.,  
Swiggy 250 → Food, Uber 300 → Travel) into categories like **Food, Travel,E-Commerce,Finance and Entertainment**.

---

## ✨ Features
- Preprocessing of transaction text (removes numbers, cleans input)  
- TF‑IDF vectorization for text representation  
- Logistic Regression for classification  
- Evaluation using **accuracy score**, **classification report**  
- Predictions on brand‑new transactions  

---

## 🛠️ Tech Stack
- **Python**  
- **Scikit‑learn** (TF‑IDF, Logistic Regression, metrics)    
- **Joblib** (model persistence)  

---

## 🚀 How to Run
1. Clone the repository  
2. Install dependencies:  
   ```bash
   pip install scikit-learn matplotlib seaborn joblib
 Example Output-
Zomato dinner order → Food
Airbnb booking payment → Travel
Nykaa cosmetics order → Shopping

