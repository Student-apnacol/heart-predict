import streamlit as st


# Ignore warnings
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import pickle
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import pickle
import pandas as pd
from sklearn.preprocessing import RobustScaler
import streamlit as st

# Load the trained model from the pickle file
with open('rm_best_model.pkl', 'rb') as file:
    best_model = pickle.load(file)

# Function to preprocess the data
def preprocess_data(df):
    # Get all numerical features
    num_features = [feature for feature in df.columns if df[feature].dtype != 'O']

    # Identify discrete features (features with ≤ 25 unique values)
    discrete_features = [feature for feature in num_features if len(df[feature].unique()) <= 25]

    # Get continuous features (numerical features that are not discrete)
    continuous_features = [feature for feature in num_features if feature not in discrete_features]
    
    # Apply RobustScaler to continuous features
    scaler = RobustScaler()
    df[continuous_features] = scaler.fit_transform(df[continuous_features])

    # Return preprocessed data
    return df

# Streamlit app code
st.title("Heart Attack Prediction")

# Upload CSV file
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file is not None:
    # Read the uploaded CSV file into a DataFrame
    df = pd.read_csv(uploaded_file)

    # Display the uploaded data
    st.write("Uploaded Data:")
    st.write(df.head())

    # Preprocess the data
    df = preprocess_data(df)

    # Prepare features and target
    if "output" in df.columns:
        X = df.drop(columns=["output"])  # Features
        y = df["output"]  # Target

        # Make predictions using the loaded model
        predictions = best_model.predict(X)
        
        # Display predictions
        df["Prediction"] = predictions
        st.write("Predictions:")
        st.write(df.head(400))

     
