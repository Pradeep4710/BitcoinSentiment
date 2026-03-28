# 📊 Trader Behavior Insights using Market Sentiment

## 🚀 Overview

This project analyzes the relationship between **Bitcoin market sentiment (Fear & Greed Index)** and **trader performance** using real trading data.

We combine:

* 📉 Market sentiment data
* 💰 Trader-level transaction data

To uncover:

* Profitability patterns
* Trader behavior under different sentiments
* Predictive insights using Machine Learning

---

## 📁 Project Structure

```
├── model.ipynb
├── historical_data.csv
├── fear_greed_index.csv
├── random_forest.pkl
├── XGBoost.pkl
├── kmeans_model.pkl
├── kmeans_scaler.pkl
├── cluster_features.pkl
├── features.pkl
├── app.py
├── approach.txt
└── README.md
```

---

## 🧠 Models Used

### 🔹 Supervised Learning

* Random Forest Classifier
* XGBoost Classifier

**Goal:** Predict whether a trade is profitable

---

### 🔹 Unsupervised Learning

* KMeans Clustering

**Goal:** Identify trader types (behavioral segmentation)

---

## 📊 Key Insights

* 📈 Highest profits occur during **Extreme Greed**
* 💡 **Fear markets** also show strong profitability (dip buying)
* 😐 Neutral sentiment → lowest performance
* 💸 High fees significantly reduce profitability
* 🧑‍💻 Distinct trader clusters exist (high-risk vs consistent traders)

---

## ⚙️ How to Run
### 0️⃣ Run the model.ipynb file in order to train and export PKL files
```
open in jupyter notebook and run all the cells
```
### 1️⃣ Install dependencies

```
pip install pandas numpy scikit-learn xgboost streamlit joblib
```

---

### 2️⃣ Run Streamlit App

```
streamlit run app.py
```

---

## 🧪 How to Use Models

### 🔹 Load models

```python
import joblib

rf = joblib.load("random_forest.pkl")
xgb = joblib.load("XGBoost.pkl")
kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("kmeans_scaler.pkl")
```

---

### 🔹 Predict Profitability

```python
prediction = xgb.predict([input_features])
```

---

### 🔹 Predict Trader Cluster

```python
scaled = scaler.transform([cluster_input])
cluster = kmeans.predict(scaled)
```

---

## 🎯 Conclusion

This project demonstrates how:

* Market sentiment impacts trading outcomes
* Machine learning can model trader behavior
* Data-driven strategies can improve trading decisions

---

## 👤 Author

Pradeep Singh****
