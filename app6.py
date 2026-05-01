import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

st.set_page_config(page_title="Customer Churn Prediction", page_icon="🔄", layout="centered")

st.title("🔄 Customer Churn Prediction System")
st.markdown("Predict whether a customer will **Churn** or **Stay**!")
st.markdown("---")

@st.cache_resource
def load_and_train():
    df = pd.read_csv("churn_prediction.csv")
    df.drop_duplicates(inplace=True)

    # Fill missing values
    for col in df.columns:
        if df[col].dtype == "object":
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].mean(), inplace=True)

    # Drop irrelevant columns
    drop_cols = ['CustomerID', 'LastPurchaseDate']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # Encode target
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Encode categorical columns
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col].astype(str))

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return model, scaler, acc, X.columns.tolist()

model, scaler, acc, feature_names = load_and_train()

# Show accuracy
st.markdown(f"### 🎯 Model Accuracy: **{acc*100:.2f}%**")
st.markdown("---")

# Sidebar inputs
st.sidebar.header("👤 Enter Customer Details")

age = st.sidebar.slider("Age", 18, 80, 35)
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
income = st.sidebar.slider("Income (₹)", 10000, 200000, 50000)
spending_score = st.sidebar.slider("Spending Score", 1, 100, 50)
purchase_amount = st.sidebar.slider("Purchase Amount (₹)", 100, 10000, 2000)
product_category = st.sidebar.selectbox("Product Category", ["Beauty", "Clothing", "Electronics", "Food", "Sports"])
payment_method = st.sidebar.selectbox("Payment Method", ["UPI", "Credit Card", "Debit Card", "Net Banking", "Cash"])
city = st.sidebar.selectbox("City", ["Delhi", "Mumbai", "Bangalore", "Chennai", "Hyderabad"])
state = st.sidebar.selectbox("State", ["DL", "MH", "KA", "TN", "TS"])
country = st.sidebar.selectbox("Country", ["IND"])
is_active = st.sidebar.selectbox("Is Active", ["Y", "N"])
returns = st.sidebar.slider("Returns", 0, 10, 1)
discount_used = st.sidebar.selectbox("Discount Used", ["True", "False"])
review_score = st.sidebar.slider("Review Score", 1.0, 5.0, 3.0)
browser = st.sidebar.selectbox("Browser", ["Chrome", "Firefox", "Safari", "Edge"])
device = st.sidebar.selectbox("Device", ["Desktop", "Mobile", "Tablet"])
session_time = st.sidebar.slider("Session Time (mins)", 1, 500, 100)

# Encode maps
gender_map = {"Female": 0, "Male": 1}
category_map = {"Beauty": 0, "Clothing": 1, "Electronics": 2, "Food": 3, "Sports": 4}
payment_map = {"Cash": 0, "Credit Card": 1, "Debit Card": 2, "Net Banking": 3, "UPI": 4}
city_map = {"Bangalore": 0, "Chennai": 1, "Delhi": 2, "Hyderabad": 3, "Mumbai": 4}
state_map = {"DL": 0, "KA": 1, "MH": 2, "TN": 3, "TS": 4}
country_map = {"IND": 0}
active_map = {"N": 0, "Y": 1}
discount_map = {"False": 0, "True": 1}
browser_map = {"Chrome": 0, "Edge": 1, "Firefox": 2, "Safari": 3}
device_map = {"Desktop": 0, "Mobile": 1, "Tablet": 2}

if st.sidebar.button("🔮 Predict Churn"):
    input_data = np.array([[
        age, gender_map[gender], income, spending_score,
        purchase_amount, category_map[product_category],
        payment_map[payment_method], city_map[city],
        state_map[state], country_map[country],
        active_map[is_active], returns,
        discount_map[discount_used], review_score,
        browser_map[browser], device_map[device], session_time
    ]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    if prediction == 1:
        st.error(f"### ⚠️ Customer is likely to CHURN!")
        st.error(f"Churn Probability: **{probability[1]*100:.1f}%**")
        st.warning("💡 Consider offering a discount or loyalty reward!")
    else:
        st.success(f"### ✅ Customer will NOT Churn!")
        st.success(f"Retention Probability: **{probability[0]*100:.1f}%**")

st.markdown("---")
st.markdown("Made with ❤️ using Streamlit & Random Forest")
