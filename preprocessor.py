import pickle
import pandas as pd
from sklearn.preprocessing import RobustScaler

class Preprocessor:
    def __init__(self):
        self.scaler = RobustScaler()
        self.num_features = []
        self.cat_features = []
        self.discrete_features = []
        self.continuous_features = []

    def fit(self, df):
        """Identify feature types and fit the scaler."""
        # Identify numeric and categorical features
        self.num_features = [col for col in df.columns if df[col].dtype != 'O']
        self.cat_features = [col for col in df.columns if df[col].dtype == 'O']

        # Identify discrete and continuous features
        self.discrete_features = [col for col in self.num_features if len(df[col].unique()) <= 25]
        self.continuous_features = [col for col in self.num_features if col not in self.discrete_features]

        # Fit scaler on continuous features
        if self.continuous_features:
            self.scaler.fit(df[self.continuous_features])

    def transform(self, df):
        """Apply transformations to the dataset."""
        df = df.copy()

        # Apply RobustScaler to continuous features
        if self.continuous_features:
            df[self.continuous_features] = self.scaler.transform(df[self.continuous_features])

        return df

    def fit_transform(self, df):
        """Fit and transform in one step."""
        self.fit(df)
        return self.transform(df)

