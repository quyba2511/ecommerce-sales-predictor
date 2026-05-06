# 🛍️ E-commerce Customer Sales Predictor

> **End-to-end Machine Learning project**: Predicting yearly customer spending for an e-commerce company using Linear Regression, deployed with Streamlit.

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31-red.svg)](https://streamlit.io/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange.svg)](https://scikit-learn.org/)

---

## 🌐 Live Demo

**👉 [Click here to try the app](https://github.com/quyba2511/ecommerce-sales-predictor.git)** *(Pô sẽ thay link sau khi deploy)*

---

## 🎯 Problem Statement

An e-commerce company sells clothing online via both their **mobile app** and **website**. They have a key business question:

> **Should we invest more in the mobile app or the website to maximize revenue?**

This project analyzes customer behavior data and builds a predictive model to answer that question with data-driven insights.

---

## 📊 Dataset

The dataset contains **500 customer records** with 5 numerical features:

| Feature | Description |
|---------|-------------|
| Avg. Session Length | Average time per in-store advisory session (minutes) |
| Time on App | Average time spent on mobile app (minutes) |
| Time on Website | Average time spent on website (minutes) |
| Length of Membership | Years as a member |
| **Yearly Amount Spent** | **🎯 Target: Annual spending (USD)** |

---

## 🛠️ Tech Stack

- **Python 3.12** — Core language
- **Pandas, NumPy** — Data manipulation
- **Scikit-learn** — Linear Regression model
- **Statsmodels** — VIF analysis for multicollinearity
- **Plotly, Matplotlib, Seaborn** — Data visualization
- **Streamlit** — Web app deployment
- **Git, GitHub** — Version control

---

## 📈 Project Pipeline

```
Raw Data → EDA → Feature Engineering → Train/Test Split 
       → Scaling → Model Training → Evaluation → Deployment
```

### 1️⃣ Exploratory Data Analysis (EDA)
- Correlation heatmap, pairplots, distribution analysis
- Identified key business insights

### 2️⃣ Data Preprocessing
- Train/Test Split (70/30)
- StandardScaler for feature normalization
- VIF Analysis: all features < 1.05 (no multicollinearity)

### 3️⃣ Model Training
- Linear Regression with scikit-learn
- Cross-validation for robustness

### 4️⃣ Model Evaluation

| Metric | Value |
|--------|-------|
| **R² Score** | 0.98 |
| **MAE** | ~$8 |
| **RMSE** | ~$10 |
| **MAPE** | ~1.6% |

### 5️⃣ Deployment
- Streamlit web app with interactive sliders
- 3 tabs: Predict | Explore Data | About
- Hosted on Streamlit Cloud

---

## 💡 Key Business Insights

1. **Length of Membership** is the strongest predictor (corr = 0.81) → Invest in **loyalty programs**

2. **Time on App** has 200x more impact than **Time on Website** ($38.6/min vs $0.19/min) → **Prioritize mobile app development**

3. Customer with 5+ years membership spend on average **$578/year** vs overall mean of **$499/year**

---

## 🚀 How to Run Locally

### Prerequisites
- Python 3.10+
- Git

### Installation

```bash
# Clone repo
git clone https://github.com/YOUR-USERNAME/ecommerce-sales-predictor.git
cd ecommerce-sales-predictor

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

App will open at `http://localhost:8501`

---

## 📁 Project Structure

```
ecommerce_sales_predictor/
├── app.py                          # Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── data/
│   └── Ecommerce_Customers.csv     # Dataset
├── models/
│   ├── linear_regression_model.pkl # Trained model
│   └── scaler.pkl                  # Feature scaler
└── notebooks/
    └── analysis.ipynb              # EDA & training notebook
```

---

## 📸 App Preview

*(Pô sẽ thêm screenshot sau khi deploy)*

---

## 👤 Author

** (NGUYEN QUY BA)**
- 📧 Email: nba204953@gmail.com
- 💼 LinkedIn: [your-profile](https://linkedin.com/in/your-profile)
- 🐙 GitHub: [quyba2511](https://github.com/quyba2511/ecommerce-sales-predictor.git)

---

## 📝 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- Dataset inspired by Kaggle's "Ecommerce Customers" dataset
- Built as part of my journey learning Machine Learning from scratch

---

⭐ **If you found this project helpful, please give it a star!**
