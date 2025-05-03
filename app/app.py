"""
Stock Price Prediction Streamlit App

This Streamlit app allows users to:
1. Load processed stock data
2. Select stocks to analyze
3. Run predictions using trained models
4. Visualize results
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

# Import local modules
import sys
sys.path.append('..')
from src.data_processing import StockDataProcessor
from src.feature_engineering import StockFeatureEngineer
from src.model_training import StockPriceModel, StockPriceModelEnsemble

# Set page config
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("1-Month Stock Price Predictor")
st.markdown("""
This application uses machine learning to predict stock prices one month in the future.
Upload processed WRDS stock data, select stocks to analyze, and get predictions.
""")

# Sidebar
st.sidebar.title("Configuration")

# Data upload
st.sidebar.header("Data Input")
uploaded_file = st.sidebar.file_uploader("Upload processed stock data CSV", type=["csv"])

# Model selection
st.sidebar.header("Model Selection")
model_option = st.sidebar.selectbox(
    "Select prediction model",
    ["XGBoost", "Random Forest", "Ensemble"]
)

# Available models
models_dir = os.path.join('..', 'models')
model_paths = {
    "XGBoost": os.path.join(models_dir, "xgboost", "xgboost_model.joblib"),
    "Random Forest": os.path.join(models_dir, "random_forest", "random_forest_model.joblib"),
    "Ensemble": os.path.join(models_dir, "ensemble")
}

# Main content
if uploaded_file is not None:
    # Load data
    data = pd.read_csv(uploaded_file)
    st.write(f"Data loaded: {data.shape[0]} rows, {data.shape[1]} columns")
    
    # Display sample data
    with st.expander("Preview data"):
        st.dataframe(data.head())
    
    # Stock selection
    if 'TICKER' in data.columns or 'ticker' in data.columns:
        ticker_col = 'TICKER' if 'TICKER' in data.columns else 'ticker'
        stock_id_col = 'PERMNO' if 'PERMNO' in data.columns else 'permno'
        
        # Get unique tickers
        unique_tickers = sorted(data[ticker_col].unique())
        
        st.header("Stock Selection")
        selected_tickers = st.multiselect(
            "Select stocks to analyze",
            unique_tickers,
            max_selections=5
        )
        
        if selected_tickers:
            st.write(f"Selected {len(selected_tickers)} stocks")
            
            # Load model
            model_path = model_paths[model_option]
            
            try:
                if model_option == "Ensemble":
                    model = StockPriceModelEnsemble.load_ensemble(model_path)
                else:
                    model = StockPriceModel.load_model(model_path)
                    
                st.success(f"Loaded {model_option} model")
                
                # Process data for selected stocks
                filtered_data = data[data[ticker_col].isin(selected_tickers)]
                
                # Process and prepare features
                processor = StockDataProcessor()
                engineer = StockFeatureEngineer()
                
                # Show progress
                with st.spinner("Processing data..."):
                    processed_data = processor.clean_data(filtered_data)
                    
                with st.spinner("Creating technical indicators..."):
                    with_tech = engineer.create_technical_indicators(processed_data)
                    
                with st.spinner("Creating market features..."):
                    with_market = engineer.create_market_features(with_tech)
                    
                with st.spinner("Creating fundamental features..."):
                    with_fundamental = engineer.create_fundamental_features(with_market)
                
                # Prepare features
                date_col = 'date' if 'date' in with_fundamental.columns else None
                
                if date_col:
                    # Sort data by date
                    with_fundamental = with_fundamental.sort_values([stock_id_col, date_col])
                    
                    # Keep the most recent data for each stock for prediction
                    most_recent = with_fundamental.groupby(stock_id_col).tail(13)  # Last 13 months for each stock
                    
                    # Separate data for visualization
                    viz_data = most_recent.copy()
                    
                    # Prepare features for prediction
                    X, y = processor.prepare_features_targets(most_recent, prediction_window=1)
                    
                    # Get the most recent data point for each stock
                    most_recent_indices = most_recent.groupby(stock_id_col)[date_col].idxmax()
                    prediction_data = most_recent.loc[most_recent_indices]
                    
                    # Prepare features for the most recent data points
                    X_recent, _ = processor.prepare_features_targets(prediction_data, prediction_window=1)
                    
                    # Scale features
                    X_scaled = engineer.scale_features(X_recent)[0]
                    
                    # Make predictions
                    with st.spinner("Making predictions..."):
                        if model_option == "Ensemble":
                            predictions = model.predict(X_scaled, method='mean')
                        else:
                            predictions = model.predict(X_scaled)
                    
                    # Create results DataFrame
                    results = pd.DataFrame({
                        'Stock': prediction_data[ticker_col].values,
                        'Current Price': prediction_data['PRC'].values,
                        'Predicted Price (1 Month)': predictions,
                        'Predicted Change (%)': (predictions / prediction_data['PRC'].values - 1) * 100
                    })
                    
                    # Display predictions
                    st.header("Price Predictions")
                    st.dataframe(results.style.format({
                        'Current Price': '${:.2f}',
                        'Predicted Price (1 Month)': '${:.2f}',
                        'Predicted Change (%)': '{:.2f}%'
                    }))
                    
                    # Create visualization
                    st.header("Visualization")
                    
                    # Prepare data for plotting
                    plot_data = []
                    
                    for ticker in selected_tickers:
                        stock_data = viz_data[viz_data[ticker_col] == ticker].sort_values(date_col)
                        
                        if len(stock_data) > 0:
                            # Get prices and dates
                            dates = stock_data[date_col].tolist()
                            prices = stock_data['PRC'].tolist()
                            
                            # Get prediction for this stock
                            stock_pred = results[results['Stock'] == ticker]
                            
                            if len(stock_pred) > 0:
                                pred_price = stock_pred['Predicted Price (1 Month)'].values[0]
                                last_date = dates[-1]
                                
                                # Add prediction point (1 month in the future)
                                if isinstance(last_date, pd.Timestamp):
                                    # Add approximately one month
                                    pred_date = last_date + pd.DateOffset(months=1)
                                else:
                                    # Handle non-datetime date format
                                    pred_date = last_date + 30  # Approximate 1 month
                                
                                # Add to plot data
                                plot_data.append({
                                    'ticker': ticker,
                                    'dates': dates + [pred_date],
                                    'prices': prices + [pred_price],
                                    'prediction_index': len(dates)
                                })
                    
                    # Create interactive plot
                    fig = go.Figure()
                    
                    # Color palette
                    colors = px.colors.qualitative.Plotly
                    
                    for i, stock in enumerate(plot_data):
                        color = colors[i % len(colors)]
                        
                        # Add historical prices
                        fig.add_trace(go.Scatter(
                            x=stock['dates'][:stock['prediction_index']],
                            y=stock['prices'][:stock['prediction_index']],
                            mode='lines+markers',
                            name=f"{stock['ticker']} (Historical)",
                            line=dict(color=color)
                        ))
                        
                        # Add prediction
                        fig.add_trace(go.Scatter(
                            x=stock['dates'][stock['prediction_index']-1:],
                            y=stock['prices'][stock['prediction_index']-1:],
                            mode='lines+markers',
                            line=dict(color=color, dash='dash'),
                            marker=dict(size=[8, 12], symbol=['circle', 'star']),
                            name=f"{stock['ticker']} (Prediction)"
                        ))
                    
                    fig.update_layout(
                        title="Stock Price Prediction (1 Month)",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        legend_title="Stocks",
                        height=600
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Model information
                    st.header("Model Information")
                    
                    if model_option == "Ensemble":
                        st.write(f"Using Ensemble of {len(model.models)} models")
                        
                        # List models in ensemble
                        model_names = [m.model_type.capitalize() for m in model.models]
                        st.write(f"Models in ensemble: {', '.join(model_names)}")
                    else:
                        st.write(f"Using {model_option} model")
                        
                        # Show feature importance if available
                        if hasattr(model, 'plot_feature_importance'):
                            st.subheader("Feature Importance")
                            
                            try:
                                fig = model.plot_feature_importance(top_n=15)
                                if fig:
                                    st.pyplot(fig)
                            except Exception as e:
                                st.warning(f"Could not plot feature importance: {str(e)}")
                
                else:
                    st.error("Date column not found in data")
                
            except Exception as e:
                st.error(f"Error processing data: {str(e)}")
        
        else:
            st.info("Please select at least one stock to analyze")
    
    else:
        st.error("Ticker column not found in data. Please ensure your data includes a 'TICKER' or 'ticker' column.")

else:
    st.info("Please upload processed stock data to begin analysis.")
    
    # Example data format
    st.header("Expected Data Format")
    st.write("The uploaded CSV should contain processed WRDS stock data with the following columns:")
    
    example_data = pd.DataFrame({
        'PERMNO': [10026, 10026, 10026],
        'date': ['2020-01-31', '2020-02-28', '2020-03-31'],
        'TICKER': ['JJSF', 'JJSF', 'JJSF'],
        'PRC': [165.84, 160.82, 121.00],
        'VOL': [22433, 18648, 39302],
        'RET': [-0.10002, -0.03027, -0.24403]
    })
    
    st.dataframe(example_data)

# Footer
st.markdown("---")
st.markdown("Stock Price Predictor | FDA_1mo_Stock_Prediction")
