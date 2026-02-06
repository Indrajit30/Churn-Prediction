
# Customer Churn Prediction (Telco)

## 🔗 Project Demo
Live demo (may take time to load on first run):  
https://churn-prediction-indrajitandswayam.streamlit.app/

---

## 📌 Abstract
Customer churn prediction is a key problem in subscription-based businesses. This project implements an end-to-end machine learning pipeline on the **Telco Customer Churn** dataset, covering data cleaning, feature engineering, model training, evaluation, and deployment. Multiple models are evaluated, with the final system using a **soft-voting ensemble of XGBoost and Random Forest**. A Streamlit application provides interactive churn prediction with **SHAP-based explanations**.

---

## 🎯 Objectives
- Predict customer churn with high recall and balanced precision  
- Compare baseline and advanced machine learning models  
- Build a soft-voting ensemble with threshold tuning  
- Provide explainability using SHAP  
- Deploy an interactive Streamlit application  

---

## 🛠️ Tech Stack
- **Core:** Python, Pandas, NumPy  
- **Modeling:** scikit-learn (Pipeline, ColumnTransformer, OneHotEncoder, StandardScaler)  
- **Imbalance Handling:** SMOTE (training split only)  
- **Models:** Logistic Regression, Random Forest, XGBoost, Neural Network  
- **Explainability:** SHAP  
- **Visualization:** Matplotlib, Seaborn  
- **App:** Streamlit  

---

## 📂 Dataset
**Dataset:** Telco Customer Churn  
**Target Variable:** `Churn`  
- `1` = Customer churned  
- `0` = Customer retained  

---

## 🧠 Methodology
- Data cleaning and validation  
- Feature engineering and categorical encoding  
- SMOTE for class imbalance (no data leakage)  
- Model training and comparison  
- Soft-voting ensemble with threshold optimization  
- SHAP-based explainability  

### Ensemble Formula
```
p_ensemble = 0.5 * p_XGB + 0.5 * p_RF
```

---

## 📊 Model Performance (F1 Scores)

| Model | F1 Score |
|------|----------|
| Logistic Regression | 0.826 |
| Random Forest | 0.825 |
| XGBoost | 0.831 |
| Neural Network | 0.830 |
| **Ensemble (RF + XGB)** | **0.853** |

---

## 📈 Results
- **Validation F1-score:** 0.85  
- **Test Recall (Churn):** 0.89  
- **Selected Threshold:** 0.44  

---

## 🚀 Streamlit Application
The app allows users to:
- Enter customer details  
- View churn probability and decision  
- Understand predictions using SHAP explanations  

---

## ▶️ How to Run

### Option 1: Use Deployed App (Recommended)
Open the demo link above and test both churn and non-churn cases.

### Option 2: Run Locally
1. Clone the repository  
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run utils/app.py
   ```

---

## 🗂️ Project Structure
```
Customer_Churn_Prediction/
├── data/
│   ├── raw/
│   │   └── Telco-Customer-Churn.csv
│   └── processed/
│       ├── Telco-Customer-Churn-Cleaned.csv
│       └── Telco-Customer-Churn-Final.csv
│
├── data_manipulation/
│   ├── data_audit.ipynb
│   ├── eda.ipynb
│   └── feature_engineering.ipynb
│
├── models/
│   ├── model_logisticReg.ipynb
│   ├── model_neuralNetwork.ipynb
│   ├── model_randomForest.ipynb
│   ├── model_xgboost.ipynb
│   └── model_RF+XGB_ensemble_voting.ipynb
│
├── testing/
│   └── model_testing.ipynb
│
├── final_model/
│   ├── final_rf_model.pkl
│   ├── final_xgb_model.pkl
│   └── final_ensemble_config.pkl
│
├── images/
│   ├── example_churn_inputs.jpeg
│   ├── example_churn_outputs.jpeg
│   ├── example_churn_shap.jpeg
│   ├── example_nochurn_inputs.jpeg
│   ├── example_nochurn_outputs.jpeg
│   └── example_nochurn_shap.jpeg
│
├── utils/
│   ├── app.py
│   ├── error_analysis.py
│   └── split.py
│
└── README.md
```

---

## 🔮 Future Work
- Unified preprocessing pipeline for training & inference  
- Experiment tracking (MLflow)  
- Calibration, robustness, and fairness checks  
- Cloud deployment (AWS / Render / Streamlit Cloud)  

---

## 👤 Authors

**Swayam Mestry**  
MS in Data Science, Rutgers University  
GitHub: https://github.com/SwayamMestry  
LinkedIn: https://linkedin.com/in/swayammestry  

**Indrajit Dalvi**  
MS in Data Science, Rutgers University  
GitHub: https://github.com/Indrajit30  
LinkedIn: https://linkedin.com/in/indrajitdalvi  
