"""
Feature Engineering Module for Stock Price Prediction

This module handles feature creation, transformation, and selection for stock price prediction.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        from src.data_processing import StockDataProcessor
        
        logger.info("Starting feature engineering pipeline")
        
        # Data processing
        processor = StockDataProcessor()
        processed_data = processor.clean_data(df)
        
        # Create features
        with_tech = self.create_technical_indicators(processed_data)
        with_market = self.create_market_features(with_tech)
        with_fundamental = self.create_fundamental_features(with_market)
        
        # Prepare features
