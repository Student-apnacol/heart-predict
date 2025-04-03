import streamlit as st
import pandas as pd
import pickle

# Load preprocessor and model
with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

with open("rm_best_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("💓 Heart Attack Risk Prediction App")

st.markdown("Enter the patient's medical info below:")

# Input form
age = st.number_input("Age", min_value=1, max_value=120, step=1)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
trtbps = st.number_input("Resting Blood Pressure (trtbps)", min_value=50, max_value=250)
chol = st.number_input("Cholesterol (chol)", min_value=50, max_value=600)
fbs = st.selectbox("Fasting Blood Sugar > 120 (fbs)", [0, 1])
restecg = st.selectbox("Resting ECG (restecg)", [0, 1, 2])
thalachh = st.number_input("Max Heart Rate (thalachh)", min_value=50, max_value=250)
exng = st.selectbox("Exercise-induced Angina (exng)", [0, 1])
oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=10.0, step=0.1)
slp = st.selectbox("Slope of ST segment (slp)", [0, 1, 2])
caa = st.selectbox("No. of Major Vessels (caa)", [0, 1, 2, 3, 4])
thall = st.selectbox("Thalassemia (thall)", [0, 1, 2])

# Convert sex to numeric
sex_num = 1 if sex == "Male" else 0

# Collect into DataFrame
input_data = pd.DataFrame([{
    'age': age, 'sex': sex_num, 'cp': cp, 'trtbps': trtbps, 'chol': chol,
    'fbs': fbs, 'restecg': restecg, 'thalachh': thalachh, 'exng': exng,
    'oldpeak': oldpeak, 'slp': slp, 'caa': caa, 'thall': thall
}])

# Predict button
if st.button("Predict Heart Attack Risk"):
    try:
        transformed_input = preprocessor.transform(input_data)
        prediction = model.predict(transformed_input)

        if prediction[0] == 1:
            st.error("⚠️ High Risk of Heart Attack")
        else:
            st.success("✅ Low Risk of Heart Attack")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
