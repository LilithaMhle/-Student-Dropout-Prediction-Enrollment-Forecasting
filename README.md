# 🎓 Machine Learning for Education Performance: Dropout Prediction & Enrollment Forecasting

This repository presents a comprehensive **machine learning project** focused on predicting student dropouts and forecasting enrollment trends in the education domain.  
It combines exploratory data analysis, predictive modeling, and an interactive **Streamlit dashboard** for visual insights.

---

## 🧭 Project Overview

The goal of this project is to:
- **Predict student dropout risk** using academic, demographic, and socio-economic indicators.  
- **Forecast future enrollments** for improved academic planning and resource allocation.  
- **Explore datasets interactively** through an intuitive dashboard.  

The repository contains both a **Jupyter Notebook** for step-by-step experimentation and a **Streamlit dashboard** for real-time visualization and interaction.

---

## 📘 Included Files

| File | Description |
|------|--------------|
| `Machine_Learning_for_Education_Performance_Classification_&_Enrollment_Forecasting.ipynb` | Jupyter Notebook containing full data exploration, preprocessing, feature engineering, model training, and evaluation steps. |
| `Practiseapp.py` | Streamlit application implementing the dashboard with four tabs: Home, Dropout Prediction, Enrollment Forecasting, and Key Insights. |
| `data/Student_Performance_Data.xlsx` | Dataset with 3 sheets containing student demographics, academic data, and socio-economic factors. |
| `data/Enrollment_Trends_Data.xlsx` | Dataset with 2 sheets providing historical enrollment and forecast input data. |
| `README.md` | Project documentation (this file). |

---

## 📊 Datasets

Two datasets are included in this repository under the `data/` directory:

1. **Student_Performance_Data.xlsx**  
   - Contains **3 sheets**:  
     - `Demographics` — student background details  
     - `Academics` — grades, attendance, and participation metrics  
     - `SocioEconomic` — parental education and income levels  

2. **Enrollment_Trends_Data.xlsx**  
   - Contains **2 sheets**:  
     - `Historical_Enrollment` — enrollment numbers by year/semester  
     - `Forecast_Inputs` — external factors such as economic indicators or regional data  

🗂️ These datasets are used in both the notebook and Streamlit app to perform:
- Dropout classification modeling  
- Enrollment forecasting and trend visualization  
- Key performance analysis and insights generation  

---

## 🚀 Streamlit Dashboard Tabs

### 🏠 Home
- Project overview and navigation guide  
- Dataset information and objectives summary  

### 🎯 Dropout Prediction
- ML model predicts whether a student is at risk of dropping out  
- Interactive input form for testing new samples  
- Displays accuracy, confusion matrix, and feature importance  

### 📈 Enrollment Forecasting
- Predicts future enrollment using regression or time-series models  
- Visualizes trends and projections interactively  
- Supports customizable forecast horizons  

### 💡 Key Insights
- Presents summarized findings from both datasets  
- Highlights critical factors influencing dropout and enrollment  
- Displays correlations and actionable recommendations  

---

## 🧩 Tech Stack

- **Language:** Python 🐍  
- **Framework:** Streamlit  
- **Core Libraries:**  
  - `pandas`, `numpy` — Data preprocessing  
  - `matplotlib`, `seaborn`, `plotly` — Visualization  
  - `scikit-learn` — Model building and evaluation  
  - `prophet` or `statsmodels` — Enrollment forecasting  
  - `joblib` — Model saving/loading  

---

## 📂 Project Structure




