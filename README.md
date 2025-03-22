# California House Price Prediction Project

## Project Overview
This project demonstrates a complete Machine Learning workflow for predicting house prices using the sklearn California Housing dataset. It showcases data preprocessing, model training, API deployment, and comprehensive testing.

## Features
- Machine Learning Model for House Price Prediction
- RandomForest Regression Algorithm
- Flask REST API
- Comprehensive Testing Suite
- Modular Project Structure
- Detailed Logging and Error Handling

## Tech Stack
- Python 3.8+
- Scikit-learn
- Pandas
- NumPy
- Flask
- Joblib
- Matplotlib
- Seaborn

## 📋 Prerequisites
- Anaconda
- Python 3.8+

## 🔧 Setup and Installation

### 1. Clone the Repository
```bash
git clone https://github.com/ratnapriyalal/house-price-prediction.git
cd house-price-prediction
```

### 2. Create and Activate Conda Environment
```bash
conda create -n house_price_env python=3.8 -y
conda activate house_price_env
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Project Workflow

### 1. Data Preprocessing
- Loads California Housing dataset
- Handles missing values
- Scales features
- Splits data into training and testing sets

### 2. Model Training
- Trains RandomForest Regression model
- Performs hyperparameter tuning
- Saves best model and preprocessor

### 3. Run API
```bash
python app.py
```
- Starts Flask API
- Serves model predictions
- Runs on http://localhost:5001

### 4. Run Predictions
```bash
# In another terminal with the same conda environment and working directory as the one where we trained the model,
# Run prediction tests
python tests/test_prediction.py

# Validate predictions
python tests/validate_prediction.py
```
