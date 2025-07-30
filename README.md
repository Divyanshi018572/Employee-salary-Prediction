# Employee-salary-Prediction
# 💼 Employee Salary Prediction using Machine Learning

This project predicts whether an employee earns **more than 50K or not** based on various demographic and employment features. It uses the **Adult Income dataset** and is implemented using **XGBoost Classifier**, with full preprocessing and a **Streamlit web app** for live predictions.

---

## 📊 Dataset

- Source: [UCI Adult Income Dataset](https://archive.ics.uci.edu/ml/datasets/adult)
- Rows: 48,842
- Features: 14 (both numerical and categorical)

---

## 📌 Problem Statement

Build a machine learning model to classify individuals as:
- `<=50K` (low income)
- `>50K` (high income)

based on features like age, education, occupation, hours-per-week, etc.

---

## 🛠️ Features Used

- `age`
- `fnlwgt`
- `education-num`
- `capital-gain`
- `capital-loss`
- `hours-per-week`
- `workclass` (One-Hot Encoded)
- `occupation` (One-Hot Encoded)
- and other preprocessed categorical features (total ~86 after encoding)

---

## ✅ Workflow

### 1. **Data Preprocessing**
- Missing value handling
- Outlier capping using IQR
- Log transformation on skewed columns
- Encoding:
  - One-Hot Encoding (for nominal categorical features)
  - Label/Ordinal Encoding (for binary/ordered features)
- Feature scaling with `StandardScaler`

### 2. **Model Training**
- Model: `XGBoostClassifier`
- Hyperparameter tuning with `GridSearchCV`
- Final Accuracy:
  - **Train:** 93.2%
  - **Test:** 87.6%
- Confusion matrix and classification report for evaluation

### 3. **Model Saving**
```python
import joblib
joblib.dump(model, 'xgboost_salary_classifier.pkl')
