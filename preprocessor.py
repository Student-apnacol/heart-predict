import pickle
import pandas as pd
from sklearn.preprocessing import RobustScaler

# Load dataset (Replace 'dataset.csv' with actual file)
df = pd.read_csv("dataset.csv")

# Identify continuous features
num_features = [col for col in df.columns if df[col].dtype != "O"]
discrete_features = [col for col in num_features if len(df[col].unique()) <= 25]
continuous_features = [col for col in num_features if col not in discrete_features]

# Apply RobustScaler to continuous features
scaler = RobustScaler()
df[continuous_features] = scaler.fit_transform(df[continuous_features])

# Save the preprocessor
with open("preprocessor.pkl", "wb") as file:
    pickle.dump(scaler, file)

print("✅ Preprocessing pipeline saved successfully!")

