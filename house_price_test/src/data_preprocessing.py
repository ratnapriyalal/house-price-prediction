import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

class HousingDataPreprocessor:
    def __init__(self):
        self.housing_data = None
        self.X = None
        self.y = None
        self.preprocessor = None
        self.feature_names = None

    def load_data(self):
        """Load California Housing dataset"""
        housing = fetch_california_housing()
        self.feature_names = housing.feature_names
        self.housing_data = pd.DataFrame(
            data=np.c_[housing.data, housing.target],
            columns=list(housing.feature_names) + ['price']
        )
        return self.housing_data

    def preprocess_data(self, test_size=0.2, random_state=42):
        """Preprocess the housing data"""
        # Ensure data is loaded
        if self.housing_data is None:
            self.load_data()

        # Separate features and target
        self.X = self.housing_data.drop('price', axis=1)
        self.y = self.housing_data['price']

        # Create preprocessing pipeline
        numeric_features = self.X.columns.tolist()
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]), numeric_features)
            ])

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )

        return X_train, X_test, y_train, y_test