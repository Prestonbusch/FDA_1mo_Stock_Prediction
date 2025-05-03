"""
1-Month Stock Price Predictor

This Streamlit app uses machine learning to predict stock prices one year ahead (end of 2025).
It works with pre-loaded data, allowing users to select stocks and prediction models.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import py7zr
import joblib
from typing import List, Dict, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import logging
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DATA_PATH = "data/raw/FDA_StockData.7z"
PREDICTION_HORIZON = 12  # Predict 12 months ahead (end of 2025)

#####################################
# DATA PROCESSING MODULE
#####################################

class StockDataProcessor:
    """Class for processing stock data for machine learning."""
    
    def __init__(self, data_path: str = None):
        """
        Initialize the stock data processor.
        
        Args:
            data_path: Path to the raw data file (can be .csv or .7z)
        """
        self.data_path = data_path
        self.data = None
        self.processed_data = None
    
    def extract_7z(self, output_dir: str = 'data/raw/extracted') -> str:
        """
        Extract a 7z archive.
        
        Args:
            output_dir: Directory to extract files to
            
        Returns:
            Path to the extracted CSV file
        """
        if not self.data_path.endswith('.7z'):
            return self.data_path
            
        logger.info(f"Extracting 7z archive: {self.data_path}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract the archive
        with py7zr.SevenZipFile(self.data_path, mode='r') as z:
            z.extractall(path=output_dir)
            file_list = z.getnames()
        
        # Find CSV files in the extracted contents
        csv_files = [f for f in file_list if f.endswith('.csv')]
        
        if not csv_files:
            raise ValueError("No CSV files found in the 7z archive")
            
        extracted_file = os.path.join(output_dir, csv_files[0])
        logger.info(f"Extracted file: {extracted_file}")
        
        return extracted_file
        
    def load_data(self, data_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load stock data from CSV file or 7z archive.
        
        Args:
            data_path: Optional path to override the instance's data_path
            
        Returns:
            Pandas DataFrame containing the stock data
        """
        file_path = data_path if data_path else self.data_path
        
        if not file_path:
            raise ValueError("No data path provided.")
            
        logger.info(f"Loading data from {file_path}")
        
        # Handle 7z archives
        if file_path.endswith('.7z'):
            file_path = self.extract_7z()
        
        try:
            # Load the CSV file
            df = pd.read_csv(file_path)
            
            # Convert date column to datetime
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            # Ensure ticker column is uppercase
            if 'TICKER' in df.columns:
                df['TICKER'] = df['TICKER'].str.upper()
            
            logger.info(f"Successfully loaded data with {df.shape[0]} rows and {df.shape[1]} columns")
            
            self.data = df
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def clean_data(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Clean the stock data by handling missing values, outliers, etc.
        
        Args:
            df: Optional DataFrame to override the instance's data
            
        Returns:
            Cleaned DataFrame
        """
        data = df if df is not None else self.data
        
        if data is None:
            raise ValueError("No data loaded. Call load_data first.")
            
        logger.info("Cleaning data")
        
        # Make a copy to avoid modifying the original
        cleaned_data = data.copy()
        
        # Remove rows where price (PRC) is missing
        cleaned_data = cleaned_data.dropna(subset=['PRC'])
        
        # Handle missing returns data
        if 'RET' in cleaned_data.columns:
            # Fill missing returns with 0 (assumes no change)
            cleaned_data['RET'] = cleaned_data['RET'].fillna(0)
        
        # Handle infinite values
        numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
        cleaned_data[numeric_cols] = cleaned_data[numeric_cols].replace([np.inf, -np.inf], np.nan)
        
        # Fill remaining NaN values with appropriate strategies
        for col in numeric_cols:
            # Use appropriate fill strategy based on column
            if col in ['VOL', 'SHROUT', 'MMCNT']:
                # For volumes, counts: replace with 0
                cleaned_data[col] = cleaned_data[col].fillna(0)
            elif col == 'DIVAMT':
                # Dividend amount: replace with 0
                cleaned_data[col] = cleaned_data[col].fillna(0)
            else:
                # For other numeric columns: use median
                cleaned_data[col] = cleaned_data[col].fillna(cleaned_data[col].median())
        
        # Remove extreme outliers (values beyond 3 standard deviations)
        for col in ['PRC', 'VOL', 'RET']:
            if col in cleaned_data.columns:
                mean, std = cleaned_data[col].mean(), cleaned_data[col].std()
                lower_bound, upper_bound = mean - 3 * std, mean + 3 * std
                cleaned_data[col] = cleaned_data[col].clip(lower_bound, upper_bound)
        
        logger.info(f"Data cleaning complete. {cleaned_data.shape[0]} rows remaining")
        
        self.processed_data = cleaned_data
        return cleaned_data
    
    def prepare_features_targets(self, 
                              df: Optional[pd.DataFrame] = None,
                              target_col: str = 'PRC',
                              prediction_window: int = PREDICTION_HORIZON) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare feature and target variables for machine learning.
        
        Args:
            df: Optional DataFrame to override the instance's processed_data
            target_col: Column name to use as prediction target
            prediction_window: Number of months ahead to predict
            
        Returns:
            Tuple of (X_features, y_targets)
        """
        data = df if df is not None else self.processed_data
        
        if data is None:
            raise ValueError("No processed data available. Call clean_data first.")
            
        logger.info(f"Preparing features and targets with {prediction_window}-month prediction window")
        
        # Ensure data is sorted by date for each ticker
        if 'date' in data.columns:
            data = data.sort_values(['PERMNO', 'date'])
        else:
            raise ValueError("Could not find a date column for time-based sorting")
        
        # Create target variable: future price after prediction_window months
        data['future_price'] = data.groupby('PERMNO')[target_col].shift(-prediction_window)
        
        # Create features
        features = data.copy()
        
        # Drop rows where future price is NA (at the end of each stock's timeline)
        features = features.dropna(subset=['future_price'])
        
        # Extract target variable
        y = features['future_price']
        
        # Remove target and identifier columns from features
        cols_to_drop = [
            'future_price', 'date', 'TICKER', 'CUSIP', 'NCUSIP', 'COMNAM', 
            'TSYMBOL', 'NAMEENDT', 'SHRCLS', 'NEXTDT', 'DCLRDT', 'DLPDT',
            'PAYDT', 'RCRDDT', 'SHRENDDT', 'ALTPRCDT'
        ]
        X = features.drop([col for col in cols_to_drop if col in features.columns], axis=1)
        
        # Convert categorical columns to numeric using one-hot encoding
        cat_cols = ['EXCHCD', 'SICCD', 'NAICS', 'PRIMEXCH', 'TRDSTAT', 'SECSTAT']
        for col in cat_cols:
            if col in X.columns and X[col].dtype == 'object':
                # Convert to categorical and create dummies
                X[col] = pd.Categorical(X[col])
                dummies = pd.get_dummies(X[col], prefix=col)
                X = pd.concat([X, dummies], axis=1)
                X = X.drop(col, axis=1)
        
        logger.info(f"Features and targets prepared: {X.shape[0]} samples with {X.shape[1]} features")
        
        return X, y
    
    def split_data(self, 
                  X: pd.DataFrame, 
                  y: pd.Series,
                  train_ratio: float = 0.7,
                  val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Split data into training, validation, and test sets.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            train_ratio: Proportion of data to use for training
            val_ratio: Proportion of data to use for validation
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        assert train_ratio + val_ratio < 1.0, "Train and validation ratios must sum to less than 1.0"
        
        # Calculate split points
        n = len(X)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        # Split the data
        X_train = X.iloc[:train_end]
        X_val = X.iloc[train_end:val_end]
        X_test = X.iloc[val_end:]
        
        y_train = y.iloc[:train_end]
        y_val = y.iloc[train_end:val_end]
        y_test = y.iloc[val_end:]
        
        logger.info(f"Data split: train={X_train.shape[0]}, val={X_val.shape[0]}, test={X_test.shape[0]} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test

#####################################
# FEATURE ENGINEERING MODULE
#####################################

class StockFeatureEngineer:
    """Class for engineering features for stock price prediction."""
    
    def __init__(self):
        """Initialize the feature engineer."""
        self.scaler = None
        self.feature_selector = None
        self.pca = None
        self.feature_names = None
    
    def create_technical_indicators(self, df: pd.DataFrame, price_col: str = 'PRC') -> pd.DataFrame:
        """
        Create technical indicators for stock price prediction.
        
        Args:
            df: DataFrame with stock data, must be sorted by date for each stock
            price_col: Column name for price data
            
        Returns:
            DataFrame with added technical indicators
        """
        logger.info("Creating technical indicators")
        
        result = df.copy()
        
        # Group by stock identifier to calculate per-stock indicators
        if 'PERMNO' in result.columns:
            grouped = result.groupby('PERMNO')
            
            # Simple Moving Averages (SMA)
            for window in [3, 6, 12]:
                result[f'sma_{window}'] = grouped[price_col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
            
            # Exponential Moving Averages (EMA)
            for window in [3, 6, 12]:
                result[f'ema_{window}'] = grouped[price_col].transform(
                    lambda x: x.ewm(span=window, adjust=False).mean()
                )
            
            # Price momentum (percent change)
            for window in [1, 3, 6, 12]:
                result[f'momentum_{window}'] = grouped[price_col].transform(
                    lambda x: x.pct_change(periods=window)
                )
            
            # Volatility (rolling standard deviation)
            for window in [3, 6, 12]:
                result[f'volatility_{window}'] = grouped[price_col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).std()
                )
            
            # Relative Strength Index (RSI)
            # First calculate daily returns
            if 'RET' not in result.columns:
                result['daily_return'] = grouped[price_col].transform(lambda x: x.pct_change())
            else:
                result['daily_return'] = result['RET']
                
            for window in [14, 30]:
                # Calculate up and down moves
                result['up_move'] = result['daily_return'].apply(lambda x: x if x > 0 else 0)
                result['down_move'] = result['daily_return'].apply(lambda x: abs(x) if x < 0 else 0)
                
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
            
            # Bollinger Bands
            for window in [20]:
                # Calculate middle band (SMA)
                result[f'bb_middle_{window}'] = grouped[price_col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
                
                # Calculate standard deviation
                result[f'bb_std_{window}'] = grouped[price_col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).std()
                )
                
                # Calculate upper and lower bands
                result[f'bb_upper_{window}'] = result[f'bb_middle_{window}'] + (result[f'bb_std_{window}'] * 2)
                result[f'bb_lower_{window}'] = result[f'bb_middle_{window}'] - (result[f'bb_std_{window}'] * 2)
                
                # Calculate %B indicator
                result[f'bb_b_{window}'] = (result[price_col] - result[f'bb_lower_{window}']) / (
                    result[f'bb_upper_{window}'] - result[f'bb_lower_{window}']
                ).replace(0, 1e-6)
                
                # Calculate bandwidth
                result[f'bb_bandwidth_{window}'] = (
                    result[f'bb_upper_{window}'] - result[f'bb_lower_{window}']
                ) / result[f'bb_middle_{window}']
                
                # Drop intermediate columns
                result = result.drop([f'bb_std_{window}'], axis=1)
            
            # Price to SMA ratios
            for window in [3, 6, 12]:
                result[f'price_to_sma_{window}'] = result[price_col] / result[f'sma_{window}'].replace(0, 1e-6)
            
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
                
                # Price-volume trend
                result['pvt'] = result['daily_return'] * result['VOL']
                result['pvt_cum'] = grouped['pvt'].transform(lambda x: x.cumsum())
        
        else:
            logger.warning("No 'PERMNO' column found, creating global indicators only")
        
        # Fill any NaNs created during feature engineering
        result = result.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        logger.info(f"Created {result.shape[1] - df.shape[1]} new technical indicators")
        
        return result
    
    def create_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create market-related features based on existing data.
        
        Args:
            df: DataFrame with stock data
            
        Returns:
            DataFrame with added market features
        """
        logger.info("Creating market features")
        
        result = df.copy()
        
        # Check if market index features exist
        market_cols = ['vwretd', 'vwretx', 'ewretd', 'ewretx', 'sprtrn']
        available_market_cols = [col for col in market_cols if col in result.columns]
        
        if available_market_cols:
            logger.info(f"Found market columns: {available_market_cols}")
            
            # Calculate moving averages for market indices
            for col in available_market_cols:
                for window in [3, 6, 12]:
                    result[f'{col}_sma_{window}'] = result[col].rolling(window=window, min_periods=1).mean()
            
            # Calculate differences between stock returns and market returns
            if 'RET' in result.columns:
                for idx_col in available_market_cols:
                    if 'ret' in idx_col.lower():
                        result[f'ret_minus_{idx_col}'] = result['RET'] - result[idx_col]
        else:
            logger.warning("No market index columns found")
        
        # Fill any NaNs
        result = result.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        logger.info(f"Created {result.shape[1] - df.shape[1]} market features")
        
        return result
    
    def create_fundamental_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create fundamental analysis features if appropriate data is available.
        
        Args:
            df: DataFrame with stock data
            
        Returns:
            DataFrame with added fundamental features
        """
        logger.info("Creating fundamental features")
        
        result = df.copy()
        
        # Check for necessary columns for fundamental analysis
        if 'PRC' in result.columns and 'SHROUT' in result.columns:
            # Market capitalization
            result['market_cap'] = result['PRC'] * result['SHROUT']
            
            # Log market cap (often more normally distributed)
            result['log_market_cap'] = np.log1p(result['market_cap'])
            
            # Market cap buckets (small, mid, large)
            result['market_cap_quantile'] = result.groupby('PERMNO')['market_cap'].transform(
                lambda x: pd.qcut(x, 3, labels=False, duplicates='drop')
            )
            
        # Price features
        if 'PRC' in result.columns:
            # Log price
            result['log_price'] = np.log1p(result['PRC'])
            
            # Price bins
            if 'PERMNO' in result.columns:
                result['price_quantile'] = result.groupby('PERMNO')['PRC'].transform(
                    lambda x: pd.qcut(x, 4, labels=False, duplicates='drop')
                )
        
        # Industry features
        if 'SICCD' in result.columns:
            # First two digits of SIC code represents major industry group
            result['industry_group'] = result['SICCD'] // 100
            
            # One-hot encode major industries (top 10)
            top_industries = result['industry_group'].value_counts().nlargest(10).index
            for ind in top_industries:
                result[f'industry_{ind}'] = (result['industry_group'] == ind).astype(int)
        
        # Fill any NaNs
        result = result.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        logger.info(f"Created {result.shape[1] - df.shape[1]} fundamental features")
        
        return result
        
    def scale_features(self, X_train: pd.DataFrame, X_val: Optional[pd.DataFrame] = None, X_test: Optional[pd.DataFrame] = None) -> Tuple:
        """
        Scale features using robust scaling to handle outliers.
        
        Args:
            X_train: Training feature set
            X_val: Optional validation feature set
            X_test: Optional test feature set
            
        Returns:
            Tuple of scaled DataFrames (X_train_scaled, X_val_scaled, X_test_scaled)
        """
        logger.info("Scaling features")
        
        # Initialize scaler if not already created
        if self.scaler is None:
            self.scaler = RobustScaler()
            
        # Fit on training data only
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        result = [X_train_scaled]
        
        # Transform validation data if provided
        if X_val is not None:
            X_val_scaled = pd.DataFrame(
                self.scaler.transform(X_val),
                columns=X_val.columns,
                index=X_val.index
            )
            result.append(X_val_scaled)
        
        # Transform test data if provided
        if X_test is not None:
            X_test_scaled = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
            result.append(X_test_scaled)
        
        logger.info("Feature scaling complete")
        
        return tuple(result)
        
    def select_features(self, X_train: pd.DataFrame, y_train: pd.Series, 
                       X_val: Optional[pd.DataFrame] = None, X_test: Optional[pd.DataFrame] = None,
                       k: int = 50) -> Tuple:
        """
        Select top k features based on F-regression.
        
        Args:
            X_train: Training feature set
            y_train: Training target values
            X_val: Optional validation feature set
            X_test: Optional test feature set
            k: Number of features to select
            
        Returns:
            Tuple of DataFrames with selected features (X_train_selected, X_val_selected, X_test_selected)
        """
        logger.info(f"Selecting top {k} features")
        
        # Ensure k is not larger than the number of features
        k = min(k, X_train.shape[1])
        
        # Initialize feature selector if not already created
        if self.feature_selector is None:
            self.feature_selector = SelectKBest(f_regression, k=k)
            
        # Fit on training data
        X_train_selected = self.feature_selector.fit_transform(X_train, y_train)
        
        # Get selected feature names
        selected_indices = self.feature_selector.get_support(indices=True)
        self.feature_names = X_train.columns[selected_indices]
        
        # Convert back to DataFrame with feature names
        X_train_selected = pd.DataFrame(
            X_train_selected,
            columns=self.feature_names,
            index=X_train.index
        )
        
        result = [X_train_selected]
        
        # Transform validation data if provided
        if X_val is not None:
            X_val_selected = pd.DataFrame(
                self.feature_selector.transform(X_val),
                columns=self.feature_names,
                index=X_val.index
            )
            result.append(X_val_selected)
        
        # Transform test data if provided
        if X_test is not None:
            X_test_selected = pd.DataFrame(
                self.feature_selector.transform(X_test),
                columns=self.feature_names,
                index=X_test.index
            )
            result.append(X_test_selected)
            
        logger.info(f"Selected {len(self.feature_names)} features: {', '.join(self.feature_names[:10])}...")
        
        return tuple(result)

#####################################
# MODEL TRAINING MODULE
#####################################

class StockPriceModel:
    """Class for training and evaluating stock price prediction models."""
    
    def __init__(self, model_type: str = 'random_forest', model_params: Optional[Dict] = None):
        """
        Initialize the stock price model.
        
        Args:
            model_type: Type of model to use ('linear', 'ridge', 'lasso', 'elastic_net', 
                       'random_forest', 'gradient_boosting')
            model_params: Parameters for the model
        """
        self.model_type = model_type.lower()
        self.model_params = model_params or {}
        self.model = None
        self.feature_names = None
        
    def get_model(self) -> Any:
        """
        Get the initialized model based on model_type.
        
        Returns:
            Initialized model object
        """
        if self.model is not None:
            return self.model
            
        logger.info(f"Initializing {self.model_type} model")
        
        # Initialize model based on type
        if self.model_type == 'linear':
            model = LinearRegression(**self.model_params)
        elif self.model_type == 'ridge':
            model = Ridge(**self.model_params)
        elif self.model_type == 'lasso':
            model = Lasso(**self.model_params)
        elif self.model_type == 'elastic_net':
            model = ElasticNet(**self.model_params)
        elif self.model_type == 'random_forest':
            model = RandomForestRegressor(n_jobs=-1, **self.model_params)
        elif self.model_type == 'gradient_boosting':
            model = GradientBoostingRegressor(**self.model_params)
        else:
            logger.error(f"Unknown model type: {self.model_type}")
            raise ValueError(f"Unknown model type: {self.model_type}")
            
        self.model = model
        return model
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Trained model
        """
        model = self.get_model()
        
        logger.info(f"Training {self.model_type} model on {X_train.shape[0]} samples with {X_train.shape[1]} features")
        
        # Store feature names for later use
        self.feature_names = X_train.columns.tolist()
        
        # Train the model
        model.fit(X_train, y_train)
        
        logger.info(f"Model training complete")
        
        return model
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Feature data
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call train() first.")
            
        logger.info(f"Making predictions on {X.shape[0]} samples")
        
        return self.model.predict(X)
        
    def evaluate(self, X: pd.DataFrame, y_true: pd.Series) -> Dict[str, float]:
        """
        Evaluate the model on the given data.
        
        Args:
            X: Feature data
            y_true: True target values
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call train() first.")
            
        logger.info(f"Evaluating model on {X.shape[0]} samples")
        
        # Make predictions
        y_pred = self.predict(X)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        # Handle zeros in y_true to avoid division by zero
        y_true_nonzero = np.where(y_true == 0, 1e-10, y_true)
        mape = np.mean(np.abs((y_true - y_pred) / y_true_nonzero)) * 100
        
        metrics = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        }
        
        logger.info(f"Evaluation metrics: {metrics}")
        
        return metrics
        
    def plot_feature_importance(self, top_n: int = 20) -> plt.Figure:
        """
        Plot feature importance for tree-based models.
        
        Args:
            top_n: Number of top features to show
            
        Returns:
            Matplotlib figure
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call train() first.")
            
        # Get feature importance
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_)
            if len(importance.shape) > 1:
                importance = importance[0]
        else:
            logger.warning("Model doesn't have feature_importances_ or coef_ attribute")
            return None
            
        # Create DataFrame
        feature_names = self.feature_names or [f"Feature {i}" for i in range(len(importance))]
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False).head(top_n)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(importance_df['feature'][::-1], importance_df['importance'][::-1])
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')
        ax.set_title(f'Top {top_n} Feature Importance for {self.model_type.capitalize()}')
        
        return fig


class StockPriceModelEnsemble:
    """Ensemble of stock price prediction models."""
    
    def __init__(self, models: Optional[List[StockPriceModel]] = None):
        """
        Initialize the ensemble.
        
        Args:
            models: List of StockPriceModel instances
        """
        self.models = models or []
        
    def add_model(self, model: StockPriceModel) -> None:
        """
        Add a model to the ensemble.
        
        Args:
            model: StockPriceModel instance
        """
        self.models.append(model)
        logger.info(f"Added {model.model_type} model to ensemble (total: {len(self.models)} models)")
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train all models in the ensemble.
        
        Args:
            X_train: Training features
            y_train: Training target
        """
        logger.info(f"Training ensemble of {len(self.models)} models")
        
        for model in self.models:
            model.train(X_train, y_train)
            
    def predict(self, X: pd.DataFrame, method: str = 'mean') -> np.ndarray:
        """
        Make ensemble predictions.
        
        Args:
            X: Feature data
            method: Ensemble method ('mean', 'median', 'weighted')
            
        Returns:
            Array of predictions
        """
        if not self.models:
            raise ValueError("No models in ensemble")
            
        logger.info(f"Making ensemble predictions using {method} method")
        
        # Get predictions from each model
        predictions = [model.predict(X) for model in self.models]
        
        # Combine predictions
        if method == 'mean':
            ensemble_pred = np.mean(predictions, axis=0)
        elif method == 'median':
            ensemble_pred = np.median(predictions, axis=0)
        elif method == 'weighted':
            # Use equal weights for now
            weights = np.ones(len(self.models)) / len(self.models)
            
            # Apply weights
            ensemble_pred = np.zeros(predictions[0].shape)
            for i, pred in enumerate(predictions):
                ensemble_pred += pred * weights[i]
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
            
        return ensemble_pred
        
    def evaluate(self, X: pd.DataFrame, y_true: pd.Series, method: str = 'mean') -> Dict[str, float]:
        """
        Evaluate the ensemble.
        
        Args:
            X: Feature data
            y_true: True target values
            method: Ensemble method ('mean', 'median', 'weighted')
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Make predictions
        y_pred = self.predict(X, method=method)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        # Handle zeros in y_true
        y_true_nonzero = np.where(y_true == 0, 1e-10, y_true)
        mape = np.mean(np.abs((y_true - y_pred) / y_true_nonzero)) * 100
        
        metrics = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        }
        
        logger.info(f"Ensemble evaluation metrics ({method} method): {metrics}")
        
        return metrics

#####################################
# STREAMLIT APP
#####################################

# Set page config
st.set_page_config(
    page_title="1-Year Stock Price Predictor",
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

# Model selection
st.sidebar.header("Model Selection")
model_option = st.sidebar.selectbox(
    "Select prediction model",
    ["Random Forest", "Linear Regression", "Ensemble"],
    index=0
)

# Cache data loading
@st.cache_data
def load_stock_data():
    """Load and process the stock data."""
    try:
        # Initialize processor
        processor = StockDataProcessor(DATA_PATH)
        
        # Load data
        df = processor.load_data()
        
        # Clean data
        cleaned_data = processor.clean_data(df)
        
        return cleaned_data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Load data
with st.spinner("Loading stock data..."):
    data = load_stock_data()

# Process data and train models (cached)
@st.cache_resource
def process_and_train_models(data):
    """Process data and train prediction models."""
    try:
        # Create feature engineer
        engineer = StockFeatureEngineer()
        
        # Generate features
        with_tech = engineer.create_technical_indicators(data)
        with_market = engineer.create_market_features(with_tech)
        with_fundamental = engineer.create_fundamental_features(with_market)
        
        # Process data for modeling
        processor = StockDataProcessor()
        X, y = processor.prepare_features_targets(with_fundamental, prediction_window=PREDICTION_HORIZON)
        X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(X, y)
        
        # Scale features
        X_train_scaled, X_val_scaled, X_test_scaled = engineer.scale_features(X_train, X_val, X_test)
        
        # Train models
        rf_model = StockPriceModel(model_type='random_forest', 
                                  model_params={'n_estimators': 100, 'max_depth': 10, 'random_state': 42})
        rf_model.train(X_train_scaled, y_train)
        
        linear_model = StockPriceModel(model_type='linear')
        linear_model.train(X_train_scaled, y_train)
        
        ensemble = StockPriceModelEnsemble([
            StockPriceModel(model_type='random_forest', 
                          model_params={'n_estimators': 100, 'max_depth': 10, 'random_state': 42}),
            StockPriceModel(model_type='linear')
        ])
        ensemble.train(X_train_scaled, y_train)
        
        # Return processed data and models
        return {
            'processed_data': with_fundamental,
            'feature_engineer': engineer,
            'models': {
                'Random Forest': rf_model,
                'Linear Regression': linear_model,
                'Ensemble': ensemble
            },
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val, 
            'y_test': y_test,
            'X_train_scaled': X_train_scaled,
            'X_val_scaled': X_val_scaled,
            'X_test_scaled': X_test_scaled
        }
    except Exception as e:
        st.error(f"Error processing data and training models: {str(e)}")
        return None

# Main content
if data is not None:
    # Get unique tickers
    if 'TICKER' in data.columns:
        tickers = sorted(data['TICKER'].unique())
        
        # Display stock selection
        st.header("Stock Selection")
        selected_ticker = st.selectbox("Select a stock to predict", tickers)
        
        if selected_ticker:
            # Process data and train models (if not already done)
            with st.spinner("Processing data and training models..."):
                processed_data = process_and_train_models(data)
            
            if processed_data:
                # Filter data for selected ticker
                ticker_data = processed_data['processed_data'][processed_data['processed_data']['TICKER'] == selected_ticker].copy()
                
                # Sort by date
                ticker_data = ticker_data.sort_values('date')
                
                # Get the most recent data point
                recent_data = ticker_data.iloc[-1:]
                
                # Prepare features for prediction
                processor = StockDataProcessor()
                X_recent, _ = processor.prepare_features_targets(recent_data, prediction_window=PREDICTION_HORIZON)
                
                # Scale features
                X_recent_scaled = processed_data['feature_engineer'].scale_features(X_recent)[0]
                
                # Get selected model
                model = processed_data['models'][model_option]
                
                # Make prediction
                if isinstance(model, StockPriceModelEnsemble):
                    prediction = model.predict(X_recent_scaled, method='mean')[0]
                else:
                    prediction = model.predict(X_recent_scaled)[0]
                
                # Get current price
                current_price = recent_data['PRC'].values[0]
                
                # Calculate predicted change
                change_pct = (prediction / current_price - 1) * 100
                
                # Display results
                st.header("Price Prediction for End of 2025")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Current Price", f"${current_price:.2f}")
                
                with col2:
                    st.metric("Predicted Price (End of 2025)", f"${prediction:.2f}")
                
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
                    y=[current_price, prediction],
                    mode='lines',
                    line=dict(color='red', dash='dash'),
                    name='Prediction'
                ))
                
                # Add prediction point
                fig.add_trace(go.Scatter(
                    x=[pred_date],
                    y=[prediction],
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
                    
                    # Show feature importance if available
                    if model_option == "Random Forest":
                        st.subheader("Feature Importance")
                        
                        try:
                            fig = model.plot_feature_importance(top_n=15)
                            if fig:
                                st.pyplot(fig)
                        except Exception as e:
                            st.warning(f"Could not plot feature importance: {str(e)}")
                
                # Performance metrics
                st.header("Model Performance")
                
                # Calculate metrics on validation set
                if isinstance(model, StockPriceModelEnsemble):
                    metrics = model.evaluate(processed_data['X_val_scaled'], processed_data['y_val'])
                else:
                    metrics = model.evaluate(processed_data['X_val_scaled'], processed_data['y_val'])
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("RMSE", f"${metrics['rmse']:.2f}")
                
                with col2:
                    st.metric("MAE", f"${metrics['mae']:.2f}")
                
                with col3:
                    st.metric("R²", f"{metrics['r2']:.2f}")
                
                with col4:
                    st.metric("MAPE", f"{metrics['mape']:.2f}%")
        else:
            st.info("Please select a stock to predict")
    else:
        st.error("Ticker column not found in data")
else:
    st.error("Failed to load stock data. Please check the data path and format.")

# Footer
st.markdown("---")
st.markdown("**Stock Price Predictor for End of 2025** | FDA_1mo_Stock_Prediction")
st.markdown("Created with Streamlit and Machine Learning")
