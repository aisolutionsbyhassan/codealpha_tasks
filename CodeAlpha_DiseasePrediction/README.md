# ❤️ Heart Disease Prediction

## Overview
This project is an AI-powered heart disease prediction system using **Support Vector Machine (SVM)**. The goal was to classify whether a patient is at risk of heart disease based on clinical parameters.

---

## 🔗 Live Web App  
Try the interactive app here:  
**https://aisolutionsbyhassan-codealpha-diseaseprediction-app-gzwulo.streamlit.app/**

---

## Model Selection
Multiple models were tested, including **Logistic Regression, Random Forest, XGBoost**, and **SVM**. While Random Forest and XGBoost achieved perfect accuracy on the test set, they showed signs of **overfitting**, and Logistic Regression did not perform as well.  

The **SVM model** was selected because it provided **high accuracy, excellent recall, and robust generalization** without overfitting.

---

## Original SVM Model (Full Features)
The **original SVM model** trained on all **14 features** achieved the following metrics:  

- **Train Accuracy:** 94.76%  
- **Test Accuracy:** 92.19%  
- **ROC-AUC:** 0.977  
- **Class 0 (No Heart Disease):** Precision 0.94, Recall 0.90, F1-score 0.92  
- **Class 1 (Heart Disease):** Precision 0.91, Recall 0.94, F1-score 0.93  

> **Key Insight:** These are the **true, real results** of the model. This represents the best predictive performance achievable.

---

## Optimized SVM Model (10 Features for App)
For a **user-friendly interactive app**, the model was reduced to **10 key features** using a Decision Tree feature selection approach.  

**Performance Metrics:**

- **Train Accuracy:** 94.63%  
- **Test Accuracy:** 91.71%  
- **ROC-AUC:** 0.968  
- **Class 0 (No Heart Disease):** Precision 0.95, Recall 0.88  
- **Class 1 (Heart Disease):** Precision 0.89, Recall 0.95  

> **Note:** Slight drop in performance is intentional to **simplify the app interface** while keeping predictions highly reliable.

---

## Features Used in App
`['cp', 'ca', 'age', 'chol', 'thal', 'oldpeak', 'trestbps', 'thalach', 'slope', 'sex']`  

These features allow for **quick, interactive predictions** without losing much accuracy.

---

The **original SVM model** represents the real strength of this project, while the app leverages a simplified version for a smooth user experience.  

Try the app, input your health metrics, and explore **how AI can help predict heart disease risk accurately and interactively**.  

> **Experience it yourself and see the predictive power of SVM in action!**


