"""
Stock Price Predictor App (Using yfinance data)

This Streamlit app fetches stock data from Yahoo Finance and uses machine learning 
to predict stock prices at the end of 2025.
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
START_DATE = "2020-01-01"
END_DATE = datetime.now().strftime('%Y-%m-%d')
PREDICTION_HORIZON = 12  # Predict 12 months ahead (end of 2025)

# Set page config
st.set_page_config(
    page_title="Stock Price Predictor for End of 2025",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("Stock Price Predictor for End of 2025")
st.markdown("""
This application uses machine learning to predict stock prices at the end of 2025.
Models are trained on data from January 2020 to the present.
""")

# Sidebar
st.sidebar.title("Configuration")

# Stock selection
st.sidebar.header("Stock Selection")
stock_options = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", 
    "JNJ", "V", "PG", "DIS", "NFLX", "INTC", "CSCO", "KO", "PEP", "MCD", 
    "WMT", "HD", "BA", "CAT", "GE", "IBM", "XOM", "CVX"
]

selected_ticker = st.sidebar.selectbox("Select a stock", stock_options)

# Model selection
st.sidebar.header("Model Selection")
model_option = st.sidebar.selectbox(
    "Select prediction model",
    ["Random Forest", "Linear Regression", "Ensemble"],
    index=0
)

# Fetch data from yfinance
@st.cache_data
def load_stock_data(ticker):
    """Fetch stock data from Yahoo Finance."""
    try:
        # Get stock data
        stock = yf.Ticker(ticker)
        df = stock.history(start=START_DATE, end=END_DATE)
        
        # Reset index to make Date a column
        df = df.reset_index()
        
        # Rename columns to match expected format
        df = df.rename(columns={
            'Date': 'date',
            'Open': 'OPEN',
            'High': 'HIGH',
            'Low': 'LOW',
            'Close': 'PRC',  # Use Close as the price column
            'Volume': 'VOL'
        })
        
        # Add ticker column
        df['TICKER'] = ticker
        
        # Calculate returns
        df['RET'] = df['PRC'].pct_change()
        
        # Add a simple PERMNO (just use hash of ticker)
        df['PERMNO'] = hash(ticker) % 100000
        
        logger.info(f"Loaded data for {ticker}: {df.shape[0]} rows")
        return df
    except Exception as e:
        logger.error(f"Error loading data for {ticker}: {str(e)}")
        return None

# Main content
with st.spinner(f"Loading data for {selected_ticker}..."):
    data = load_stock_data(selected_ticker)

if data is not None:
    # Feature engineering
    @st.cache_data
    def engineer_features(df):
        """Create technical and fundamental features for stock prediction."""
        result = df.copy()
        
        # Group by stock
        if 'PERMNO' in result.columns:
            grouped = result.groupby('PERMNO')
            
            # Create moving averages
            for window in [3, 6, 12]:
                result[f'sma_{window}'] = grouped['PRC'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
            
            # Create momentum features
            for window in [1, 3, 6]:
                result[f'momentum_{window}'] = grouped['PRC'].transform(
                    lambda x: x.pct_change(periods=window)
                )
            
            # Create volatility features
            for window in [3, 6, 12]:
                result[f'volatility_{window}'] = grouped['PRC'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).std()
                )
            
            # Create price to SMA ratios
            for window in [3, 6, 12]:
                result[f'price_to_sma_{window}'] = result['PRC'] / result[f'sma_{window}'].replace(0, 1e-6)
                
            # Volume indicators if volume is available
            if 'VOL' in result.columns:
                # Volume moving averages
                for window in [3, 6, 12]:
                    result[f'vol_sma_{window}'] = grouped['VOL'].transform(
                        lambda x: x.rolling(window=window, min_periods=1).mean()
                    )
                
                # Volume to moving average ratio
                for window in [3, 6, 12]:
                    result[f'vol_ratio_{window}'] = result['VOL'] / result[f'vol_sma_{window}'].replace(0, 1e-6)
            
            # Relative Strength Index (RSI) if returns data is available
            if 'RET' in result.columns:
                for window in [14, 30]:
                    # Calculate up and down moves
                    result['up_move'] = result['RET'].apply(lambda x: x if x > 0 else 0)
                    result['down_move'] = result['RET'].apply(lambda x: abs(x) if x < 0 else 0)
                    
                    # Calculate rolling averages of up and down moves
                    result[f'avg_up_{window}'] = grouped['up_move'].transform(
                        lambda x: x.rolling(window=window, min_periods=1).mean()
                    )
                    result[f'avg_down_{window}'] = grouped['down_move'].transform(
                        lambda x: x.rolling(window=window, min_periods=1).mean()
                    )
                    
                    # Calculate RSI
                    result[f'rsi_{window}'] = 100 - (100 / (1 + (result[f'avg_up_{window}'] / result[f'avg_down_{window}'].replace(0, 1e-6))))
                    
                    # Drop intermediate columns
                    result = result.drop(['up_move', 'down_move', f'avg_up_{window}', f'avg_down_{window}'], axis=1)
        
        # Fill NaNs created during feature engineering
        result = result.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return result
    
    # Create features
    with st.spinner("Creating prediction features..."):
        featured_data = engineer_features(data)
    
    # Sort by date
    ticker_data = featured_data.sort_values('date')
    
    # Create train/test split based on dates
    @st.cache_resource
    def train_predict_model(ticker_df, model_name):
        """Train model and make prediction for selected ticker."""
        # Prepare data for modeling
        # Create target variable: price 12 months ahead
        df = ticker_df.copy()
        df['future_price'] = df['PRC'].shift(-PREDICTION_HORIZON)
        
        # Drop rows where future price is NA
        modeling_data = df.dropna(subset=['future_price'])
        
        # Keep latest data point for prediction
        prediction_data = df.iloc[-1:].copy()
        
        # Prepare features
        exclude_cols = ['date', 'TICKER', 'PERMNO', 'future_price', 'OPEN', 'HIGH', 'LOW']
        feature_cols = [col for col in modeling_data.columns 
                      if col not in exclude_cols]
        
        X = modeling_data[feature_cols]
        y = modeling_data['future_price']
        
        # Scale features
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
        
        # Train model
        if model_name == "Random Forest":
            model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
            model.fit(X_scaled_df, y)
            
            # Make prediction
            X_pred = prediction_data[feature_cols]
            X_pred_scaled = scaler.transform(X_pred)
            prediction = model.predict(X_pred_scaled)[0]
            
            # Feature importance
            importances = model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
        elif model_name == "Linear Regression":
            model = LinearRegression()
            model.fit(X_scaled_df, y)
            
            # Make prediction
            X_pred = prediction_data[feature_cols]
            X_pred_scaled = scaler.transform(X_pred)
            prediction = model.predict(X_pred_scaled)[0]
            
            # Feature importance
            importances = np.abs(model.coef_)
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
        else:  # Ensemble
            rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
            linear_model = LinearRegression()
            
            rf_model.fit(X_scaled_df, y)
            linear_model.fit(X_scaled_df, y)
            
            # Make prediction
            X_pred = prediction_data[feature_cols]
            X_pred_scaled = scaler.transform(X_pred)
            
            rf_pred = rf_model.predict(X_pred_scaled)[0]
            linear_pred = linear_model.predict(X_pred_scaled)[0]
            
            # Ensemble prediction (average)
            prediction = (rf_pred + linear_pred) / 2
            
            # Feature importance from Random Forest
            importances = rf_model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': importances
            }).sort_values('importance', ascending=False)
        
        # Calculate metrics on training data
        train_pred = model.predict(X_scaled_df) if model_name != "Ensemble" else \
                    (rf_model.predict(X_scaled_df) + linear_model.predict(X_scaled_df)) / 2
        
        rmse = np.sqrt(mean_squared_error(y, train_pred))
        mae = mean_absolute_error(y, train_pred)
        r2 = r2_score(y, train_pred)
        
        # Calculate MAPE
        y_nonzero = np.where(y == 0, 1e-10, y)
        mape = np.mean(np.abs((y - train_pred) / y_nonzero)) * 100
        
        return {
            'current_price': prediction_data['PRC'].values[0],
            'predicted_price': prediction,
            'feature_importance': feature_importance,
            'metrics': {
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'mape': mape
            }
        }
    
    # Run prediction
    with st.spinner(f"Predicting future price for {selected_ticker}..."):
        result = train_predict_model(ticker_data, model_option)
    
    # Display results
    st.header("Price Prediction for End of 2025")
    
    current_price = result['current_price']
    predicted_price = result['predicted_price']
    change_pct = (predicted_price / current_price - 1) * 100
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Current Price", f"${current_price:.2f}")
    
    with col2:
        st.metric("Predicted Price (End of 2025)", f"${predicted_price:.2f}")
    
    with col3:
        st.metric("Predicted Change", f"{change_pct:.2f}%", 
                delta=f"{change_pct:.2f}%", 
                delta_color="normal")
    
    # Plot price history and prediction
    st.header("Price History and Prediction")
    
    # Prepare data for plotting
    dates = ticker_data['date'].tolist()
    prices = ticker_data['PRC'].tolist()
    
    # Add prediction point
    last_date = dates[-1]
    # Add approximately one year
    pred_date = last_date + pd.DateOffset(months=12)
    
    # Create interactive plot
    fig = go.Figure()
    
    # Add historical prices
    fig.add_trace(go.Scatter(
        x=dates,
        y=prices,
        mode='lines+markers',
        name='Historical Price',
        line=dict(color='blue')
    ))
    
    # Add prediction line
    fig.add_trace(go.Scatter(
        x=[last_date, pred_date],
        y=[current_price, predicted_price],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Prediction'
    ))
    
    # Add prediction point
    fig.add_trace(go.Scatter(
        x=[pred_date],
        y=[predicted_price],
        mode='markers',
        marker=dict(size=12, color='red', symbol='star'),
        name='End of 2025 Prediction'
    ))
    
    fig.update_layout(
        title=f"{selected_ticker} Stock Price Prediction",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        legend_title="Legend",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Model information
    st.header("Model Information")
    
    if model_option == "Ensemble":
        st.write("Using an ensemble of Random Forest and Linear Regression models")
    else:
        st.write(f"Using a {model_option} model")
        
    # Show feature importance
    st.subheader("Feature Importance")
    
    feature_imp = result['feature_importance']
    if feature_imp is not None:
        # Show top 15 features
        top_features = feature_imp.head(15)
        
        fig = px.bar(
            top_features, 
            x='importance', 
            y='feature',
            orientation='h',
            title='Top 15 Features by Importance',
            labels={'importance': 'Importance', 'feature': 'Feature'},
            height=500
        )
        
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance metrics
    st.header("Model Performance")
    
    metrics = result['metrics']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("RMSE", f"${metrics['rmse']:.2f}")
    
    with col2:
        st.metric("MAE", f"${metrics['mae']:.2f}")
    
    with col3:
        st.metric("R²", f"{metrics['r2']:.2f}")
    
    with col4:
        st.metric("MAPE", f"{metrics['mape']:.2f}%")
    
    # Interpretation and disclaimer
    st.header("Interpretation")
    
    st.markdown(f"""
    Based on the {model_option} model's analysis of historical data, {selected_ticker}'s price is predicted to be 
    **${predicted_price:.2f}** at the end of 2025, representing a **{change_pct:.2f}%** change from the current price.
    
    **Disclaimer:** This prediction is based on historical patterns and technical indicators. 
    Stock markets are influenced by many factors including economic events, company performance, and market sentiment 
    that may not be captured in this model. This prediction should not be used as the sole basis for investment decisions.
    """)
else:
    st.error(f"Failed to load data for {selected_ticker}. Please try another stock.")

# Footer
st.markdown("---")
st.markdown("**Stock Price Predictor for End of 2025** | FDA_1mo_Stock_Prediction")
st.markdown("Created with Streamlit and Machine Learning")
