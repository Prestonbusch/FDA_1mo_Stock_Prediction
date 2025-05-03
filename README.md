# Stock Price Predictor

A machine learning application for predicting future stock prices based on WRDS (Wharton Research Data Services) data.

## Overview

Stock Price Predictor is a comprehensive machine learning solution that leverages historical stock data from WRDS to predict future stock prices. The application uses advanced feature engineering techniques and various machine learning algorithms to identify patterns and make predictions. This project includes:

- Data processing pipeline for WRDS stock data  
- Feature engineering module with technical, fundamental, and market indicators  
- Multiple machine learning models (XGBoost, Random Forest, Linear Regression, etc.)  
- Model ensemble for improved prediction accuracy  
- Interactive Streamlit web application for visualization and prediction  

## Table of Contents

- [Features](#features)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Data Requirements](#data-requirements)  
- [Model Training](#model-training)  
- [Web Application](#web-application)  
- [Project Structure](#project-structure)  
- [Contributing](#contributing)  
- [License](#license)  

## Features

### Data Processing

- Automated cleaning and preprocessing of WRDS stock data  
- Handling of missing values and outliers  
- Time series data preparation  

### Feature Engineering

- Technical indicators (Moving Averages, RSI, Bollinger Bands, etc.)  
- Market features (Market returns, volatility, etc.)  
- Fundamental analysis metrics  
- Advanced feature selection  

### Machine Learning Models

- XGBoost for high accuracy prediction  
- Random Forest for robust ensemble learning  
- Linear models (Linear Regression, Ridge, Lasso)  
- Model ensemble for improved performance  

### Streamlit Web Application

- Interactive visualization of stock data  
- Prediction configuration and execution  
- Model insights and performance metrics  
- Downloadable prediction results  

## Installation

### Requirements

- Python 3.8+  
- pip or conda for package management  

### Setup

1. Clone the repository:  
```bash
git clone https://github.com/yourusername/stock-price-predictor.git  
cd stock-price-predictor  
```  

2. Create a virtual environment (recommended):  
```bash
python -m venv venv  
source venv/bin/activate  # On Windows: venv\Scripts\activate  
```  

3. Install dependencies:  
```bash
pip install -r requirements.txt  
```  

## Usage

### Data Preparation

1. Download monthly stock data from WRDS database  
2. Place the CSV file in the `data/` directory  
3. Process the data using the data processing module:  
```bash
python -m src.data_processing --input data/wrds_stock_data.csv --output data/processed/  
```  

### Training Models

Train machine learning models using the processed data:  
```bash
python -m src.model_training --data data/processed/processed_stock_data.csv --models xgboost,random_forest,linear --output models/  
```  

### Running the Web Application

Launch the Streamlit web application:  
```bash
cd app  
streamlit run app.py  
```  
The application will be available at [http://localhost:8501](http://localhost:8501) in your web browser.  

## Data Requirements

### Recommended WRDS Fields

For best performance, include the following variables from WRDS in your data pull:

#### 🎯 Target Variables  
- `prc`: Stock price (used as the prediction target)  
- `ret`: Return (with dividends)  
- `retx`: Return (without dividends)  

#### 📊 Predictive Features  
- `vol`: Trading volume  
- `spread`: Bid-ask spread  
- `askhi`, `bidlo`: High/low prices of the month  
- `divamt`: Dividend amount  
- `vwretd`, `ewretd`: Value-weighted and equal-weighted market returns  
- `sprtrn`: S&P 500 return  
- `shrout`: Shares outstanding  

#### 🏷 Identifiers and Metadata  
- `permco`, `cusip`, `ticker`, `comnam`: Unique company identifiers  
- `shrcd`: Share code (use to filter common stocks, e.g., codes 10 or 11)  
- `exchcd`: Exchange code  
- `siccd`, `naics`: Industry classification  

#### ⚙️ Optional Adjustments  
- `cfacpr`, `cfacshr`: Cumulative price/share adjustments (for stock splits/dividends)  

## Model Training

### Default Training  
```bash
python scripts/train_models.py  
```  

### Custom Training  
```bash
python -m src.model_training --data data/processed/processed_stock_data.csv --model xgboost --params scripts/params/xgboost_params.json  
```  

### Hyperparameter Tuning  
```bash
python -m src.model_training --tune --data data/processed/processed_stock_data.csv --model xgboost --output models/tuned/  
```  

## Web Application

The Streamlit web application provides an intuitive interface for:

1. **Data Exploration** – Visualize and understand historical trends  
2. **Stock Selection** – Choose which stocks to analyze  
3. **Prediction Configuration** – Customize prediction settings  
4. **Results Visualization** – View results with confidence intervals  
5. **Model Insights** – Explore model diagnostics and feature importance  

## Project Structure

```
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
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.  

1. Fork the repository  
2. Create your feature branch (`git checkout -b feature/amazing-feature`)  
3. Commit your changes (`git commit -m 'Add some amazing feature'`)  
4. Push to the branch (`git push origin feature/amazing-feature`)  
5. Open a Pull Request  

## License

This project is licensed under the MIT License - see the LICENSE file for details.  

---

**Disclaimer**: This application is for educational and research purposes only. Stock prediction involves risk, and investors should always do their own research and possibly consult with financial advisors before making investment decisions.
