"""
FDA 1-Month Stock Price Prediction App

A streamlined app that combines data processing, feature engineering, 
model training, and the Streamlit interface in a single file.
"""

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import os
import py7zr
import joblib
from typing import List, Dict, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import logging
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None
try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.graph_objects as go
import plotly.express as px

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#####################################
# DATA PROCESSING MODULE
#####################################

class StockDataProcessor:
    """Class for processing WRDS stock data for machine learning."""
    
    def __init__(self, data_path: str = None):
        """
        Initialize the stock data processor.
        
        Args:
            data_path: Path to the raw WRDS data file (can be .csv or .7z)
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
        
        # Try to infer date format and data types
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
                              prediction_window: int = 1) -> Tuple[pd.DataFrame, pd.Series]:
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
    
    def save_processed_data(self, output_dir: str = 'data/processed'):
        """
        Save processed data to disk.
        
        Args:
            output_dir: Directory to save processed data
        """
        if self.processed_data is None:
            raise ValueError("No processed data available. Call clean_data first.")
        
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save processed data
        output_path = os.path.join(output_dir, 'processed_stock_data.csv')
        self.processed_data.to_csv(output_path, index=False)
        
        logger.info(f"Processed data saved to {output_path}")
        
    def process_pipeline(self, data_path: str = None, output_dir: str = 'data/processed'):
        """
        Run the full data processing pipeline.
        
        Args:
            data_path: Path to raw data file
            output_dir: Directory to save processed data
        """
        if data_path:
            self.data_path = data_path
            
        self.load_data()
        self.clean_data()
        self.save_processed_data(output_dir)
        
        logger.info("Data processing pipeline completed successfully")
        
        return self.processed_data

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
    
    def get_feature_importance(self, model, X: pd.DataFrame) -> pd.DataFrame:
        """
        Extract feature importance from trained model.
        
        Args:
            model: Trained model with feature_importances_ attribute
            X: Feature DataFrame
            
        Returns:
            DataFrame with feature importance scores
        """
        # Check if model has feature_importances_ attribute
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_)
        else:
            logger.warning("Model doesn't have feature_importances_ or coef_ attribute")
            return None
        
        # Create DataFrame with feature names and importance scores
        if len(importance.shape) > 1 and importance.shape[0] > 1:
            # For multi-output models, take the mean importance
            importance = np.mean(importance, axis=0)
        
        feature_names = X.columns
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df
    
    def feature_pipeline(self, 
                        df: pd.DataFrame, 
                        target_col: str = 'PRC',
                        prediction_window: int = 1,
                        train_ratio: float = 0.7,
                        val_ratio: float = 0.15,
                        top_k_features: int = 50) -> Dict:
        """
        Complete feature engineering pipeline.
        
        Args:
            df: Raw DataFrame with stock data
            target_col: Column to predict
            prediction_window: Months ahead to predict
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            top_k_features: Number of features to select
            
        Returns:
            Dictionary with processed datasets and metadata
        """
        logger.info("Starting feature engineering pipeline")
        
        # Data processing
        processor = StockDataProcessor()
        processed_data = processor.clean_data(df)
        
        # Create features
        with_tech = self.create_technical_indicators(processed_data)
        with_market = self.create_market_features(with_tech)
        with_fundamental = self.create_fundamental_features(with_market)
        
        # Prepare features and targets
        X, y = processor.prepare_features_targets(
            with_fundamental, 
            target_col=target_col,
            prediction_window=prediction_window
        )
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(
            X, y, train_ratio, val_ratio
        )
        
        # Scale features
        X_train_scaled, X_val_scaled, X_test_scaled = self.scale_features(X_train, X_val, X_test)
        
        # Select features
        X_train_selected, X_val_selected, X_test_selected = self.select_features(
            X_train_scaled, y_train, X_val_scaled, X_test_scaled, k=top_k_features
        )
        
        logger.info("Feature engineering pipeline complete")
        
        return {
            'X_train': X_train_selected,
            'X_val': X_val_selected,
            'X_test': X_test_selected,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'feature_names': self.feature_names,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector
        }

#####################################
# MODEL TRAINING MODULE
#####################################

class StockPriceModel:
    """Class for training and evaluating stock price prediction models."""
    
    def __init__(self, model_type: str = 'xgboost', model_params: Optional[Dict] = None):
        """
        Initialize the stock price model.
        
        Args:
            model_type: Type of model to use ('linear', 'ridge', 'lasso', 'elastic_net', 
                       'random_forest', 'gradient_boosting', 'ada_boost', 'svr', 
                       'knn', 'xgboost', 'lightgbm')
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
        elif self.model_type == 'ada_boost':
            model = AdaBoostRegressor(**self.model_params)
        elif self.model_type == 'svr':
            model = SVR(**self.model_params)
        elif self.model_type == 'knn':
            model = KNeighborsRegressor(n_jobs=-1, **self.model_params)
        elif self.model_type == 'xgboost':
            if XGBRegressor is None:
                raise ImportError("XGBoost is not installed. Please install it with 'pip install xgboost'.")
            model = XGBRegressor(n_jobs=-1, **self.model_params)
        elif self.model_type == 'lightgbm':
            if LGBMRegressor is None:
                raise ImportError("LightGBM is not installed. Please install it with 'pip install lightgbm'.")
            model = LGBMRegressor(n_jobs=-1, **self.model_params)
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
        
        # Calculate direction accuracy (how often the model predicts the correct price direction)
        # For the last data point of each stock, compare with its previous price
        direction_accuracy = np.mean((y_true > 0) == (y_pred > 0)) * 100
        
        metrics = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'direction_accuracy': direction_accuracy
        }
        
        logger.info(f"Evaluation metrics: {metrics}")
        
        return metrics
        
    def save_model(self, directory: str = 'models', filename: Optional[str] = None) -> str:
        """
        Save the trained model to disk.
        
        Args:
            directory: Directory to save the model
            filename: Filename for the model (optional)
            
        Returns:
            Path to the saved model
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call train() first.")
            
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.model_type}_model_{timestamp}.joblib"
            
        # Save the model
        model_path = os.path.join(directory, filename)
        joblib.dump(self.model, model_path)
        
        # Save the feature names
        feature_path = os.path.join(directory, filename.replace(".joblib", "_features.joblib"))
        joblib.dump(self.feature_names, feature_path)
        
        logger.info(f"Model saved to {model_path}")
        
        return model_path
        
    @classmethod
    def load_model(cls, model_path: str) -> 'StockPriceModel':
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            StockPriceModel instance with loaded model
        """
        logger.info(f"Loading model from {model_path}")
        
        # Load the model
        model = joblib.load(model_path)
        
        # Determine model type
        model_type = type(model).__name__.lower()
        if 'xgb' in model_type:
            model_type = 'xgboost'
        elif 'lgbm' in model_type:
            model_type = 'lightgbm'
        elif 'randomforest' in model_type:
            model_type = 'random_forest'
        elif 'gradientboosting' in model_type:
            model_type = 'gradient_boosting'
        elif 'adaboost' in model_type:
            model_type = 'ada_boost'
        elif 'linear' in model_type:
            model_type = 'linear'
        
        # Create instance
        instance = cls(model_type=model_type)
        instance.model = model
        
        # Try to load feature names
        try:
            feature_path = model_path.replace(".joblib", "_features.joblib")
            if os.path.exists(feature_path):
                instance.feature_names = joblib.load(feature_path)
                logger.info(f"Loaded {len(instance.feature_names)} feature names")
        except Exception as e:
            logger.warning(f"Could not load feature names: {str(e)}")
        
        return instance
        
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
            # Use inverse RMSE as weights
            # Assumes each model has been evaluated on a validation set
            weights = []
            for model in self.models:
                # Default weight of 1 if no RMSE available
                weights.append(1.0)
            
            # Normalize weights
            weights = np.array(weights) / sum(weights)
            
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
        
        # Calculate direction accuracy
        direction_accuracy = np.mean((y_true > 0) == (y_pred > 0)) * 100
        
        metrics = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'direction_accuracy': direction_accuracy
        }
        
        logger.info(f"Ensemble evaluation metrics ({method} method): {metrics}")
        
        return metrics
        
    def save_models(self, directory: str = 'models/ensemble') -> List[str]:
        """
        Save all models in the ensemble.
        
        Args:
            directory: Directory to save models
            
        Returns:
            List of paths to saved models
        """
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Save each model
        paths = []
        for i, model in enumerate(self.models):
            filename = f"ensemble_model_{i}_{model.model_type}.joblib"
            path = model.save_model(directory, filename)
            paths.append(path)
            
        # Save ensemble metadata
        metadata = {
            'model_types': [model.model_type for model in self.models],
            'model_paths': paths
        }
        
        metadata_path = os.path.join(directory, 'ensemble_metadata.joblib')
        joblib.dump(metadata, metadata_path)
        
        logger.info(f"Ensemble saved to {directory}")
        
        return paths
        
    @classmethod
    def load_ensemble(cls, directory: str = 'models/ensemble') -> 'StockPriceModelEnsemble':
        """
        Load ensemble from disk.
        
        Args:
            directory: Directory containing saved models
            
        Returns:
            StockPriceModelEnsemble instance
        """
        logger.info(f"Loading ensemble from {directory}")
        
        # Load metadata
        metadata_path = os.path.join(directory, 'ensemble_metadata.joblib')
        
        try:
            metadata = joblib.load(metadata_path)
            
            # Load each model
            models = []
            for path in metadata['model_paths']:
                model = StockPriceModel.load_model(path)
                models.append(model)
                
            # Create ensemble
            ensemble = cls(models)
            
            logger.info(f"Loaded ensemble with {len(models)} models")
            
            return ensemble
        except Exception as e:
            logger.error(f"Failed to load ensemble: {str(e)}")
            
            # Attempt to create a default ensemble with available models
            logger.info("Creating default ensemble with random forest and linear regression")
            
            rf_model = StockPriceModel(model_type='random_forest')
            linear_model = StockPriceModel(model_type='linear')
            
            return cls([rf_model, linear_model])

#####################################
# STREAMLIT APP
#####################################

# Set page config
st.set_page_config(
    page_title="1-Month Stock Price Predictor",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("1-Month Stock Price Predictor")
st.markdown("""
This application uses machine learning to predict stock prices one month in the future.
Upload your stock data, select stocks to analyze, and get predictions.
""")

# Sidebar
st.sidebar.title("Configuration")

# File upload
st.sidebar.header("Data Input")
uploaded_file = st.sidebar.file_uploader("Upload stock data CSV", type=["csv"])

# Sample data option
use_sample_data = st.sidebar.checkbox("Use sample data", value=False)

# Model selection
st.sidebar.header("Model Selection")
model_option = st.sidebar.selectbox(
    "Select prediction model",
    ["Random Forest", "Linear Regression", "Ensemble"],
    index=0
)

# Define default models - will be initialized only when needed
default_models = {
    "Random Forest": None,
    "Linear Regression": None,
    "Ensemble": None
}

def load_or_create_model(model_name):
    """Load or create a model based on selection."""
    if default_models[model_name] is not None:
        return default_models[model_name]
    
    if model_name == "Random Forest":
        model = StockPriceModel(model_type='random_forest', 
                              model_params={'n_estimators': 100, 'max_depth': 10, 'random_state': 42})
        default_models[model_name] = model
    elif model_name == "Linear Regression":
        model = StockPriceModel(model_type='linear')
        default_models[model_name] = model
    elif model_name == "Ensemble":
        rf_model = StockPriceModel(model_type='random_forest', 
                                 model_params={'n_estimators': 100, 'max_depth': 10, 'random_state': 42})
        linear_model = StockPriceModel(model_type='linear')
        model = StockPriceModelEnsemble([rf_model, linear_model])
        default_models[model_name] = model
    
    return default_models[model_name]

# Load sample data if requested
@st.cache_data
def load_sample_data():
    """Load sample stock data for demonstration."""
    # Create sample data with a few stocks and dates
    dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='M')
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'FB']
    
    data_rows = []
    
    for ticker in tickers:
        # Generate realistic price series
        base_price = np.random.uniform(50, 500)
        for i, date in enumerate(dates):
            price = base_price * (1 + 0.05 * np.sin(i/3) + np.random.normal(0, 0.03))
            volume = int(np.random.uniform(1000000, 10000000))
            ret = np.random.normal(0.01, 0.05)
            
            # Create a row
            data_rows.append({
                'PERMNO': hash(ticker) % 10000,  # Create a stable PERMNO from ticker
                'date': date,
                'TICKER': ticker,
                'PRC': price,
                'VOL': volume,
                'RET': ret,
                'SHROUT': int(np.random.uniform(500000, 5000000)),
                'SICCD': np.random.randint(1000, 9999),
                'vwretd': np.random.normal(0.01, 0.02),
                'sprtrn': np.random.normal(0.01, 0.02)
            })
    
    # Create DataFrame
    df = pd.DataFrame(data_rows)
    return df

# Main content
if uploaded_file is not None or use_sample_data:
    # Load data
    if use_sample_data:
        data = load_sample_data()
        st.success("Loaded sample data for demonstration")
    else:
        data = pd.read_csv(uploaded_file)
        st.success(f"Data loaded: {data.shape[0]} rows, {data.shape[1]} columns")
    
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
            with st.spinner("Loading model..."):
                model = load_or_create_model(model_option)
                
            st.success(f"Loaded {model_option} model")
            
            # Process data for selected stocks
            filtered_data = data[data[ticker_col].isin(selected_tickers)]
            
            # Convert date column to datetime if needed
            if 'date' in filtered_data.columns and not pd.api.types.is_datetime64_any_dtype(filtered_data['date']):
                filtered_data['date'] = pd.to_datetime(filtered_data['date'])
            
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
                
                # Handle the case where we need to train the model
                if isinstance(model, StockPriceModel) and model.model is None:
                    with st.spinner("Training model on your data..."):
                        # Create train/val split
                        X_train, X_val, y_train, y_val = X[:-len(prediction_data)], X[-len(prediction_data):], y[:-len(prediction_data)], y[-len(prediction_data):]
                        
                        # Scale features
                        X_train_scaled, X_val_scaled = engineer.scale_features(X_train, X_val)
                        
                        # Train model
                        model.train(X_train_scaled, y_train)
                        
                        # Evaluate
                        metrics = model.evaluate(X_val_scaled, y_val)
                        st.info(f"Model trained with validation RMSE: {metrics['rmse']:.2f}")
                elif isinstance(model, StockPriceModelEnsemble):
                    # Train ensemble models if needed
                    models_to_train = [m for m in model.models if m.model is None]
                    if models_to_train:
                        with st.spinner("Training ensemble models on your data..."):
                            # Create train/val split
                            X_train, X_val, y_train, y_val = X[:-len(prediction_data)], X[-len(prediction_data):], y[:-len(prediction_data)], y[-len(prediction_data):]
                            
                            # Scale features
                            X_train_scaled, X_val_scaled = engineer.scale_features(X_train, X_val)
                            
                            # Train each model
                            for m in models_to_train:
                                m.train(X_train_scaled, y_train)
                            
                            # Evaluate ensemble
                            metrics = model.evaluate(X_val_scaled, y_val)
                            st.info(f"Ensemble trained with validation RMSE: {metrics['rmse']:.2f}")
                
                # Scale the recent data for prediction
                X_recent_scaled = engineer.scale_features(X_recent)[0]
                
                # Make predictions
                with st.spinner("Making predictions..."):
                    if isinstance(model, StockPriceModelEnsemble):
                        predictions = model.predict(X_recent_scaled, method='mean')
                    else:
                        predictions = model.predict(X_recent_scaled)
                
                # Create results DataFrame
                results = pd.DataFrame({
                    'Stock': prediction_data[ticker_col].values,
                    'Current Price': prediction_data['PRC'].values,
                    'Predicted Price (1 Month)': predictions,
                    'Predicted Change (%)': (predictions / prediction_data['PRC'].values - 1) * 100
                })
                
                # Display predictions
                st.header("Price Predictions")
                
                # Format the results for display
                formatted_results = results.copy()
                formatted_results['Current Price'] = formatted_results['Current Price'].map('${:.2f}'.format)
                formatted_results['Predicted Price (1 Month)'] = formatted_results['Predicted Price (1 Month)'].map('${:.2f}'.format)
                formatted_results['Predicted Change (%)'] = formatted_results['Predicted Change (%)'].map('{:.2f}%'.format)
                
                st.dataframe(formatted_results)
                
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
                
                if isinstance(model, StockPriceModelEnsemble):
                    st.write(f"Using Ensemble of {len(model.models)} models")
                    
                    # List models in ensemble
                    model_names = [m.model_type.capitalize() for m in model.models]
                    st.write(f"Models in ensemble: {', '.join(model_names)}")
                else:
                    st.write(f"Using {model_option} model")
                    
                    # Show feature importance if available and model is trained
                    if model.model is not None and hasattr(model, 'plot_feature_importance'):
                        st.subheader("Feature Importance")
                        
                        try:
                            fig = model.plot_feature_importance(top_n=15)
                            if fig:
                                st.pyplot(fig)
                        except Exception as e:
                            st.warning(f"Could not plot feature importance: {str(e)}")
            
            else:
                st.error("Date column not found in data")
            
        else:
            st.info("Please select at least one stock to analyze")
    
    else:
        st.error("Ticker column not found in data. Please ensure your data includes a 'TICKER' or 'ticker' column.")

else:
    st.info("Please upload stock data or use the sample data to begin analysis.")
    
    # Information about expected data format
    st.header("Expected Data Format")
    st.write("The uploaded CSV should contain stock data with the following columns:")
    
    example_data = pd.DataFrame({
        'PERMNO': [10026, 10026, 10026],
        'date': ['2020-01-31', '2020-02-28', '2020-03-31'],
        'TICKER': ['JJSF', 'JJSF', 'JJSF'],
        'PRC': [165.84, 160.82, 121.00],
        'VOL': [22433, 18648, 39302],
        'RET': [-0.10002, -0.03027, -0.24403]
    })
    
    st.dataframe(example_data)
    
    st.markdown("""
    ### Key Columns:
    - **PERMNO**: Stock identifier
    - **date**: Date of observation
    - **TICKER**: Stock ticker symbol
    - **PRC**: Stock price
    - **VOL**: Trading volume
    - **RET**: Returns
    
    The app will automatically generate technical indicators and other features for prediction.
    """)
    
    # Sample data button
    if st.button("Use Sample Data"):
        st.session_state.use_sample_data = True
        st.experimental_rerun()

# Footer
st.markdown("---")
st.markdown("**1-Month Stock Price Predictor** | FDA_1mo_Stock_Prediction")
st.markdown("Created with Streamlit and Machine Learning")
