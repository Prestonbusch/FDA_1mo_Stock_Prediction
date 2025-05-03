# FDA_1mo_Stock_Prediction
  # Used machine learning models to predict a 1 month future stock price of the user's choice
Stock Price Predictor
A machine learning application for predicting future stock prices based on WRDS (Wharton Research Data Services) data.

Overview
Stock Price Predictor is a comprehensive machine learning solution that leverages historical stock data from WRDS to predict future stock prices. The application uses advanced feature engineering techniques and various machine learning algorithms to identify patterns and make predictions.
This project includes:

Data processing pipeline for WRDS stock data
Feature engineering module with technical, fundamental, and market indicators
Multiple machine learning models (XGBoost, Random Forest, Linear Regression, etc.)
Model ensemble for improved prediction accuracy
Interactive Streamlit web application for visualization and prediction

Table of Contents

Features
Installation
Usage
Data Requirements
Model Training
Web Application
Project Structure
Contributing
License

Features
Data Processing

Automated cleaning and preprocessing of WRDS stock data
Handling of missing values and outliers
Time series data preparation

Feature Engineering

Technical indicators (Moving Averages, RSI, Bollinger Bands, etc.)
Market features (Market returns, volatility, etc.)
Fundamental analysis metrics
Advanced feature selection

Machine Learning Models

XGBoost for high accuracy prediction
Random Forest for robust ensemble learning
Linear models (Linear Regression, Ridge, Lasso)
Model ensemble for improved performance

Streamlit Web Application

Interactive visualization of stock data
Prediction configuration and execution
Model insights and performance metrics
Downloadable prediction results

Installation
Requirements

Python 3.8+
pip or conda for package management

Setup

Clone the repository:

bashgit clone https://github.com/yourusername/stock-price-predictor.git
cd stock-price-predictor

Create a virtual environment (recommended):

bashpython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies:

bashpip install -r requirements.txt
Usage
Data Preparation

Download monthly stock data from WRDS database
Place the CSV file in the data/ directory
Process the data using the data processing module:

bashpython -m src.data_processing --input data/wrds_stock_data.csv --output data/processed/
Training Models
Train machine learning models using the processed data:
bashpython -m src.model_training --data data/processed/processed_stock_data.csv --models xgboost,random_forest,linear --output models/
Running the Web Application
Launch the Streamlit web application:
bashcd app
streamlit run app.py
The application will be available at http://localhost:8501 in your web browser.
Data Requirements
The application requires WRDS stock data with the following variables:

ticker: Stock ticker symbol
date or similar: Date of observation
prc: Stock price
vol: Trading volume
Additional variables like returns, market data, etc. (optional but recommended)

For a complete list of supported variables, see the data documentation.
Model Training
Default Training
The repository includes scripts to train models with default parameters:
bashpython scripts/train_models.py
Custom Training
For custom model training with specific parameters, use:
bashpython -m src.model_training --data data/processed/processed_stock_data.csv --model xgboost --params scripts/params/xgboost_params.json
Hyperparameter Tuning
To perform hyperparameter tuning:
bashpython -m src.model_training --tune --data data/processed/processed_stock_data.csv --model xgboost --output models/tuned/
Web Application
The Streamlit web application provides an intuitive interface for:

Data Exploration: View and explore the loaded stock data
Stock Selection: Choose specific stocks to analyze
Prediction Configuration: Set prediction horizon and model parameters
Results Visualization: View prediction results with confidence intervals
Model Insights: Understand model performance and feature importance

Project Structure
stock-price-predictor/
├── .gitignore
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── xgboost/
│   ├── random_forest/
│   └── ensemble/
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── model_evaluation.py
├── notebooks/
│   └── exploratory_analysis.ipynb
├── app/
│   ├── app.py
│   └── pages/
│       ├── home.py
│       ├── prediction.py
│       └── model_insights.py
├── scripts/
│   ├── train_models.py
│   └── params/
│       ├── xgboost_params.json
│       └── random_forest_params.json
└── docs/
    ├── data_documentation.md
    └── model_documentation.md
Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

Fork the repository
Create your feature branch (git checkout -b feature/amazing-feature)
Commit your changes (git commit -m 'Add some amazing feature')
Push to the branch (git push origin feature/amazing-feature)
Open a Pull Request

License
This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgements

WRDS (Wharton Research Data Services) for providing the financial data.
Streamlit for the amazing web application framework.
scikit-learn, XGBoost, and other machine learning libraries.


Disclaimer: This application is for educational and research purposes only. Stock prediction involves risk, and investors should always do their own research and possibly consult with financial advisors before making investment decisions.
