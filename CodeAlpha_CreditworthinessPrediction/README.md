# 🎗️ **Creditworthiness Prediction Web App**

A machine learning project that predicts whether a customer is **GOOD** or **BAD** in terms of credit risk.  
Multiple models were trained and compared (**Logistic Regression, KNN, SVM, Decision Tree, Random Forest**, etc.), and the **Tuned Random Forest** model was selected due to its strong recall on high-risk borrowers.

This project includes full preprocessing, balancing, model evaluation, and a **Streamlit web app** for real-time predictions.

---

## 🌐 **Try the Web App**  
👉 https://aisolutionsbyhassan-codealpha-creditworthinesspredic-app-quwycn.streamlit.app/

---

## 📌 Features  
- End-to-end ML pipeline (cleaning → encoding → scaling → EDA → model training → deployment)  
- **Multiple ML models tested & compared**  
- Class imbalance handled using **SMOTE/Oversampling**  
- Tuned Random Forest optimized to identify **BAD (risky)** customers  
- Real-time prediction interface built with **Streamlit**  
- Probability-based output for better interpretability  
- Safe default-value handling in prediction function  

---

## 📊 Model Performance  

**Final Model:** Tuned Random Forest Classifier  

**Performance Metrics:**  
- **F1 Score:** 0.7684  
- **ROC-AUC:** 0.7909  
- **Train Accuracy:** 86.01%  
- **Test Accuracy:** 70.67%  
- **BAD Class Recall (priority metric): ~80%**  
  - Critical for detecting high-risk borrowers

**Why Random Forest?**  
It gave the best balance between catching high-risk borrowers and overall generalization compared to other models.

---
