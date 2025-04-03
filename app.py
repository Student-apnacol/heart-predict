import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import RobustScaler
import sklearn
from sklearn.preprocessing import RobustScaler
print(sklearn.__version__)


# Load the trained model
with open('rm_best_model.pkl', 'rb') as file:  # Ensure this is the correct filename
    best_model = pickle.load(file)

st.title("Heart Attack Prediction")

# Upload CSV file
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:")
    st.write(df.head())

    # Ensure 'output' column is not used for prediction
    if "output" in df.columns:
        X = df.drop(columns=["output"])
    else:
        X = df

    # Identify continuous features (excluding categorical/discrete features)
    num_features = [col for col in X.columns if X[col].dtype != 'O']
    discrete_features = [col for col in num_features if len(X[col].unique()) <= 25]
    continuous_features = [col for col in num_features if col not in discrete_features]

    # Apply RobustScaler only to continuous features
    if continuous_features:
        scaler = RobustScaler()
        X[continuous_features] = scaler.fit_transform(X[continuous_features])

    # Predict
    predictions = best_model.predict(X)
    df["Prediction"] = predictions

    # Display predictions
    st.write("Predictions:")
    st.write(df.head(400))


     
