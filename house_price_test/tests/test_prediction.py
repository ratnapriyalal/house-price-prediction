import requests
import json
from sklearn.datasets import fetch_california_housing

class HousePricePredictionTest:
    def __init__(self, api_url='http://localhost:5001/predict'):
        self.api_url = api_url
        # Get feature names from the original dataset
        housing = fetch_california_housing()
        self.feature_names = housing.feature_names

    def generate_test_cases(self):
        # Generate test cases that match the feature names
        return [
            dict(zip(self.feature_names, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])),
            dict(zip(self.feature_names, [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5])),
            dict(zip(self.feature_names, [0, 0, 0, 0, 0, 0, 0, 0])),
            dict(zip(self.feature_names, [10, 10, 10, 10, 10, 10, 10, 10]))
        ]

    def test_prediction(self, features):
        try:
            payload = {'features': list(features.values())}
            
            response = requests.post(
                self.api_url, 
                json=payload,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                return {
                    'features': features,
                    'status': 'success',
                    'prediction': response.json()
                }
            else:
                return {
                    'features': features,
                    'status': 'error',
                    'message': response.text
                }
        
        except requests.exceptions.RequestException as e:
            return {
                'features': features,
                'status': 'connection_error',
                'message': str(e)
            }

    def run_comprehensive_test(self):
        print("Starting Comprehensive House Price Prediction API Test")
        print("-" * 50)
        
        test_cases = self.generate_test_cases()
        test_results = []
        
        for case in test_cases:
            print(f"\nTesting Features: {case}")
            result = self.test_prediction(case)
            test_results.append(result)
            
            if result['status'] == 'success':
                print(f"Prediction: {result['prediction']}")
                print(f"Status: {result['status']}")
            else:
                print(f"Error: {result.get('message', 'Unknown error')}")
        
        self.generate_test_summary(test_results)

    def generate_test_summary(self, test_results):
        print("\n" + "=" * 50)
        print("TEST SUMMARY")
        print("=" * 50)
        
        total_tests = len(test_results)
        successful_tests = sum(1 for result in test_results if result['status'] == 'success')
        failed_tests = total_tests - successful_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"Successful Tests: {successful_tests}")
        print(f"Failed Tests: {failed_tests}")
        
        if failed_tests > 0:
            print("\nFailed Test Cases:")
            for result in test_results:
                if result['status'] != 'success':
                    print(f"Features: {result['features']}")
                    print(f"Error: {result.get('message', 'Unknown error')}")

def main():
    predictor_test = HousePricePredictionTest()
    predictor_test.run_comprehensive_test()

if __name__ == "__main__":
    main()