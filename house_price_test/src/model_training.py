import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from data_preprocessing import HousingDataPreprocessor

class HousePriceModel:
    def __init__(self):
        self.preprocessor = None
        self.model = None
        self.best_params = None

    def train_model(self):
        # Load and preprocess data
        preprocessor = HousingDataPreprocessor()
        preprocessor.load_data()
        X_train, X_test, y_train, y_test = preprocessor.preprocess_data()

        # Prepare model with GridSearchCV
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
        
        rf = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(
            estimator=rf, 
            param_grid=param_grid, 
            cv=5, 
            scoring='neg_mean_squared_error'
        )
        
        # Fit the model
        X_train_transformed = preprocessor.preprocessor.fit_transform(X_train)
        grid_search.fit(X_train_transformed, y_train)
        
        # Store best model and preprocessor
        self.model = grid_search.best_estimator_
        self.preprocessor = preprocessor.preprocessor
        self.best_params = grid_search.best_params_
        
        # Evaluate model
        X_test_transformed = self.preprocessor.transform(X_test)
        y_pred = self.model.predict(X_test_transformed)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Best Parameters: {self.best_params}")
        print(f"RMSE: {rmse}")
        print(f"MAE: {mae}")
        print(f"R2 Score: {r2}")
        
        # Save model and preprocessor
        joblib.dump(self.model, 'models/house_price_model.pkl')
        joblib.dump(self.preprocessor, 'models/preprocessor.pkl')
        
        return self.model

# Train model when script is run directly
if __name__ == "__main__":
    model_trainer = HousePriceModel()
    model_trainer.train_model()