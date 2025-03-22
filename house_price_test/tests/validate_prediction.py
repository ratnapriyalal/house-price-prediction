import requests
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

class PredictionValidator:
    def __init__(self, api_url='http://localhost:5001/predict'):
        self.api_url = api_url
        self.load_reference_data()

    def load_reference_data(self):
        housing = fetch_california_housing()
        self.X = housing.data
        self.feature_names = housing.feature_names
        self.scaler = StandardScaler()
        self.scaled_X = self.scaler.fit_transform(self.X)

    def validate_predictions(self, num_samples=10):
        print("Starting Prediction Validation")
        print("-" * 40)

        sample_indices = np.random.choice(
            len(self.X), 
            size=num_samples, 
            replace=False
        )

        validation_results = []

        for idx in sample_indices:
            sample_features = dict(zip(self.feature_names, self.X[idx].tolist()))
            
            try:
                response = requests.post(
                    self.api_url, 
                    json={'features': list(sample_features.values())},
                    headers={'Content-Type': 'application/json'}
                )
                
                if response.status_code == 200:
                    api_prediction = response.json().get('predicted_price')
                    
                    validation_result = {
                        'features': sample_features,
                        'api_prediction': api_prediction,
                        'status': 'validated'
                    }
                    
                    validation_results.append(validation_result)
                    
                    print(f"\nFeatures: {sample_features}")
                    print(f"API Prediction: ${api_prediction:.2f}")
                
                else:
                    print(f"API Error for sample {idx}: {response.text}")
            
            except Exception as e:
                print(f"Validation error for sample {idx}: {e}")

        self.generate_validation_summary(validation_results)

    def generate_validation_summary(self, validation_results):
        print("\n" + "=" * 50)
        print("PREDICTION VALIDATION SUMMARY")
        print("=" * 50)
        
        total_validations = len(validation_results)
        print(f"Total Validations: {total_validations}")

def main():
    validator = PredictionValidator()
    validator.validate_predictions()

if __name__ == "__main__":
    main()