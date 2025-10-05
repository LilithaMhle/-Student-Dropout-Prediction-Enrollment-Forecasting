import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from prophet import Prophet
import plotly.express as px

# --- Page config ---
st.set_page_config(page_title="Student Retention Dashboard", layout="wide")

# --- Teal theme ---
def apply_theme():
    st.markdown("""
    <style>
    .stApp {background: rgba(0,150,180,0.4); backdrop-filter: blur(20px);}
    [data-testid="stSidebar"] {background: rgba(0,100,150,0.4); backdrop-filter: blur(20px);}
    h1, h2, h3 {color: #ffffff;}
    .stButton>button {background-color: #008080; color: white; font-weight: bold;}
    </style>
    """, unsafe_allow_html=True)

apply_theme()

st.sidebar.title("Navigation")
menu = st.sidebar.selectbox("Select Page", ["Home", "Dropout Prediction", "Enrollment Forecasting", "Key Insights"])

# -----------------------------
# HOME: Upload CSVs
# -----------------------------
if menu == "Home":
    st.title("🎓 Student Retention Dashboard")
    st.subheader("Upload your datasets (Ctrl+Click to select both)")

    uploaded_files = st.file_uploader(
        "Select two CSV files (Dropout & Enrollment)", type="csv", accept_multiple_files=True
    )

    if uploaded_files:
        for f in uploaded_files:
            try:
                df = pd.read_csv(f)
                df.columns = df.columns.str.strip().str.lower()  # normalize column names
            except Exception as e:
                st.error(f"Failed to read {f.name}: {e}")
                continue

            # Detect and transform dropout dataset
            if 'g3' in df.columns:
                df['dropout'] = (df['g3'] < 10).astype(int)
                st.session_state['df_dropout'] = df
                st.success(f"Dropout dataset loaded and processed from {f.name}")

            # Detect and transform enrollment dataset
            elif any(col for col in df.columns if 'total men' in col.lower()) and any(col for col in df.columns if 'total women' in col.lower()):
                df['enrollment'] = df.filter(like='men').sum(axis=1) + df.filter(like='women').sum(axis=1)
                if 'level of student' in df.columns:
                    df_enroll = df[['level of student', 'enrollment']].rename(columns={'level of student':'ds','enrollment':'y'})
                    # Convert to datetime for Prophet
                    df_enroll['ds'] = pd.to_datetime(df_enroll['ds'].astype(str) + '-01-01')
                    st.session_state['df_enroll'] = df_enroll
                    st.success(f"Enrollment dataset loaded and processed from {f.name}")
                else:
                    st.warning(f"{f.name} missing 'level of student' column for enrollment")
            else:
                st.warning(f"File {f.name} doesn't match expected Dropout or Enrollment dataset structure. Columns: {list(df.columns)}")

    # Show previews
    if 'df_dropout' in st.session_state:
        st.subheader("Dropout Dataset Preview")
        st.dataframe(st.session_state['df_dropout'].head())
    if 'df_enroll' in st.session_state:
        st.subheader("Enrollment Dataset Preview")
        st.dataframe(st.session_state['df_enroll'].head())

# -----------------------------
# Dropout Prediction
# -----------------------------
elif menu == "Dropout Prediction":
    st.header("Dropout Prediction")
    if 'df_dropout' not in st.session_state:
        st.warning("Please upload Dropout dataset first on Home tab.")
    else:
        df_filtered = st.session_state['df_dropout'].copy()

        # Filters
        if 'faculty' in df_filtered.columns:
            faculty = st.sidebar.multiselect("Faculty", df_filtered['faculty'].unique(), default=df_filtered['faculty'].unique())
            df_filtered = df_filtered[df_filtered['faculty'].isin(faculty)]
        if 'financial_aid' in df_filtered.columns:
            aid = st.sidebar.multiselect("Financial Aid", df_filtered['financial_aid'].unique(), default=df_filtered['financial_aid'].unique())
            df_filtered = df_filtered[df_filtered['financial_aid'].isin(aid)]
        if 'first_year_gpa' in df_filtered.columns:
            gpa_range = st.sidebar.slider("GPA Range", float(df_filtered['first_year_gpa'].min()), float(df_filtered['first_year_gpa'].max()), (0.0, 4.0))
            df_filtered = df_filtered[(df_filtered['first_year_gpa'] >= gpa_range[0]) & (df_filtered['first_year_gpa'] <= gpa_range[1])]

        st.subheader("Filtered Dataset Preview")
        st.dataframe(df_filtered.head())

        # Encode categorical columns
        df_class = df_filtered.copy()
        for col in df_class.select_dtypes(include='object').columns:
            df_class[col] = LabelEncoder().fit_transform(df_class[col])

        if 'dropout' in df_class.columns:
            X = df_class.drop('dropout', axis=1)
            y = df_class['dropout']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            # Metrics
            st.subheader("Metrics")
            st.write("Accuracy:", round(accuracy_score(y_test, y_pred), 2))
            st.write("Precision:", round(precision_score(y_test, y_pred), 2))
            st.write("Recall:", round(recall_score(y_test, y_pred), 2))
            st.write("F1 Score:", round(f1_score(y_test, y_pred), 2))

            # Feature importance
            importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': clf.feature_importances_}).sort_values(by='Importance', ascending=False)
            fig = px.bar(importance_df, x='Feature', y='Importance', color_discrete_sequence=['#00FFFF'], title="Feature Importance")
            st.plotly_chart(fig)

# -----------------------------
# Enrollment Forecasting
# -----------------------------
elif menu == "Enrollment Forecasting":
    st.header("Enrollment Forecasting")
    if 'df_enroll' not in st.session_state:
        st.warning("Please upload Enrollment dataset first on Home tab.")
    else:
        df_forecast = st.session_state['df_enroll'].copy()
        model = Prophet(yearly_seasonality=True)
        model.fit(df_forecast)

        horizon = st.sidebar.slider("Forecast Years", 1, 10, 5)
        future = model.make_future_dataframe(periods=horizon, freq='Y')
        forecast = model.predict(future)

        fig = px.line(forecast, x='ds', y='yhat', title="Projected Enrollment", color_discrete_sequence=['#00FFFF'])
        st.plotly_chart(fig)

# -----------------------------
# Key Insights
# -----------------------------
elif menu == "Key Insights":
    st.header("Key Insights")
    st.markdown("""
    <ul>
    <li>Dropout prediction identifies at-risk students based on grades.</li>
    <li>Enrollment forecasting predicts total students per year.</li>
    <li>Interactive filters allow exploration by faculty, GPA, and financial aid.</li>
    </ul>
    """, unsafe_allow_html=True)








