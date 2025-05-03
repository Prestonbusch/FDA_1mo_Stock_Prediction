"""
Stock Data Cleaning Script

This script extracts the 7z archive, cleans the data, and saves it to a CSV file.
"""

import pandas as pd
import numpy as np
import os
import py7zr
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
RAW_DATA_PATH = "data/raw/FDA_StockData.7z"
EXTRACTED_DIR = "data/raw/extracted"
CLEANED_DATA_PATH = "data/processed/cleaned_stock_data.csv"

def extract_7z(data_path, output_dir):
    """Extract a 7z archive."""
    logger.info(f"Extracting 7z archive: {data_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract the archive
    with py7zr.SevenZipFile(data_path, mode='r') as z:
        z.extractall(path=output_dir)
        file_list = z.getnames()
    
    # Find CSV files in the extracted contents
    csv_files = [f for f in file_list if f.endswith('.csv')]
    
    if not csv_files:
        raise ValueError("No CSV files found in the 7z archive")
        
    extracted_file = os.path.join(output_dir, csv_files[0])
    logger.info(f"Extracted file: {extracted_file}")
    
    return extracted_file

def clean_data(file_path):
    """Clean the stock data."""
    logger.info(f"Loading data from {file_path}")
    
    # Load the CSV file
    df = pd.read_csv(file_path)
    
    # Convert date column to datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    # Ensure ticker column is uppercase
    if 'TICKER' in df.columns:
        df['TICKER'] = df['TICKER'].str.upper()
    
    logger.info(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Remove rows with missing prices
    if 'PRC' in df.columns:
        df = df.dropna(subset=['PRC'])
    
    # Handle missing returns data
    if 'RET' in df.columns:
        df['RET'] = df['RET'].fillna(0)
    
    # Handle infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    
    # Fill remaining NaN values
    for col in numeric_cols:
        if col in ['VOL', 'SHROUT', 'MMCNT']:
            df[col] = df[col].fillna(0)
        elif col == 'DIVAMT':
            df[col] = df[col].fillna(0)
        else:
            df[col] = df[col].fillna(df[col].median())
    
    # Remove extreme outliers
    for col in ['PRC', 'VOL', 'RET']:
        if col in df.columns:
            mean, std = df[col].mean(), df[col].std()
            lower_bound, upper_bound = mean - 3 * std, mean + 3 * std
            df[col] = df[col].clip(lower_bound, upper_bound)
    
    logger.info(f"Data cleaning complete: {df.shape[0]} rows")
    
    return df

def main():
    """Run the data cleaning process."""
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(CLEANED_DATA_PATH), exist_ok=True)
    
    # Extract the 7z archive
    extracted_file = extract_7z(RAW_DATA_PATH, EXTRACTED_DIR)
    
    # Clean the data
    cleaned_data = clean_data(extracted_file)
    
    # Save the cleaned data
    cleaned_data.to_csv(CLEANED_DATA_PATH, index=False)
    logger.info(f"Cleaned data saved to {CLEANED_DATA_PATH}")

if __name__ == "__main__":
    main()
