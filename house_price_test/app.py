from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from src.data_preprocessing import HousingDataPreprocessor


app = Flask(__name__)

# Load model and preprocessor
model = joblib.load('models/house_price_model.pkl')
preprocessor = joblib.load('models/preprocessor.pkl')

# Get feature names from the original dataset
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
feature_names = housing.feature_names

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data
        input_data = request.json['features']
        
        # Convert to pandas DataFrame with correct column names
        input_df = pd.DataFrame([input_data], columns=feature_names)
        
        # Preprocess input
        processed_input = preprocessor.transform(input_df)
        
        # Make prediction
        prediction = model.predict(processed_input)[0]
        
        return jsonify({
            'predicted_price': float(prediction * 100000),
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)