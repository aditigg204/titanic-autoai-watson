
import streamlit as st
import pandas as pd
import requests
import json

# ----------------------
# IBM Cloud API Settings
# ----------------------

API_KEY = "1i1OiNvnZbtLlGayLrMtK_rT5FmhWO14BRFH76rqc4Xg"  # Replace with your IBM Cloud API key
DEPLOYMENT_URL = "https://us-south.ml.cloud.ibm.com/ml/v4/deployments/36502499-32ce-4289-a23a-1345ebffc824/predictions?version=2021-05-01" # Replace with your deployment endpoint

# ----------------------
# Get IBM Access Token
# ----------------------
def get_access_token(api_key):
    url = "https://iam.cloud.ibm.com/identity/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = f"apikey={api_key}&grant_type=urn:ibm:params:oauth:grant-type:apikey"
    response = requests.post(url, headers=headers, data=data)
    return response.json()["access_token"]

# ----------------------
# Preprocess Input (Only fill missing values)
# ----------------------
FEATURE_NAMES = [
    "PassengerId", "Pclass", "Name", "Sex", "Age",
    "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"
]

def preprocess_input(df):
    df = df.copy()  # Ensure we're working with a full copy
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df["Cabin"] = df["Cabin"].fillna("Unknown")
    df["Ticket"] = df["Ticket"].fillna("Unknown")
    df["Name"] = df["Name"].fillna("Unknown")
    return df[FEATURE_NAMES]

# ----------------------
# Prediction Function
# ----------------------
def predict(data):
    token = get_access_token(API_KEY)
    headers = {
        "Authorization": "Bearer " + token,
        "Content-Type": "application/json"
    }

    payload = {
        "input_data": [{
            "fields": FEATURE_NAMES,
            "values": data
        }]
    }

    # Debug payload
    st.write("📦 Final Payload to IBM model:")
    st.json(payload)

    response = requests.post(DEPLOYMENT_URL, headers=headers, json=payload)
    
    if response.status_code == 200:
        prediction = response.json()
        return prediction['predictions'][0]['values'][0][0]
    else:
        st.error(f"API Error {response.status_code}: {response.text}")
        return None

# ----------------------
# Streamlit UI
# ----------------------
st.title("🚢 Titanic Survival Prediction (IBM Cloud)")

uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    st.write("📝 Uploaded Data:")
    st.dataframe(input_df)

    cleaned_df = preprocess_input(input_df)

    if st.button("🔍 Predict"):
        values = cleaned_df.values.tolist()
        prediction = predict(values)
        if prediction is not None:
            st.success(f"🎯 Prediction: {'Survived' if prediction == 1 else 'Did Not Survive'}")
