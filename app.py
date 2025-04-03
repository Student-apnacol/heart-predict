import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open('heart_attack_pipeline.pkl', 'rb') as file:
    best_model = pickle.load(file)

st.title("Heart Attack Prediction")

# Upload CSV file
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:")
    st.write(df.head())

    # Ensure 'target' column is not used for prediction
    if "target" in df.columns:
        X = df.drop(columns=["target"])
    else:
        X = df

    # Predict
    predictions = best_model.predict(X)
    df["Prediction"] = predictions

    # Display predictions
    st.write("Predictions:")
    st.write(df.head(400))

     
