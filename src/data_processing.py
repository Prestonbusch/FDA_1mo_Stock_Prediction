"""
Stock Price Data Processing Module

This module handles the preprocessing of WRDS stock data for machine learning model training.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import os
import py7zr  # For handling 7z archives
from datetime import datetime, timedelta
import logging
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process WRDS stock data for ML prediction')
    parser.add_argument('--input', type=str, required=True, help='Path to input CSV file or 7z archive')
    parser.add_argument('--output', type=str, default='data/processed', help='Directory to save processed data')
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Run data processing pipeline
    processor = StockDataProcessor(args.input)
    processed_data = processor.process_pipeline(output_dir=args.output)
    
    # Prepare features and targets
    X, y = processor.prepare_features_targets(prediction_window=1)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(X, y)
    
    # Print summary
    print(f"\nData Processing Summary:")
    print(f"Input data: {args.input}")
    print(f"Processed data: {args.output}/processed_stock_data.csv")
    print(f"Number of samples: {len(X)}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
