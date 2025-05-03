"""
Model Evaluation Module for Stock Price Prediction

This module handles evaluation metrics, visualizations, and performance analysis for stock price prediction models.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from datetime import datetime

# Import local modules
from src.model_training import StockPriceModel, StockPriceModelEnsemble
from src.data_processing import StockDataProcessor
from src.feature_engineering import StockFeatureEngineer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_model(model, X, y, model_name="Model"):
    """
    Evaluate a model on given data and print/return metrics.
    
    Args:
        model: Trained model
        X: Feature data
        y: Target values
        model_name: Name to identify the model in outputs
        
    Returns:
        Dictionary of metrics
    """
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    y_true_nonzero = np.where(y == 0, 1e-10, y)
    mape = np.mean(np.abs((y - y_pred) / y_true_nonzero)) * 100
    
    # Calculate direction accuracy
    direction_accuracy = np.mean((y > 0) == (y_pred > 0)) * 100
    
    metrics = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'direction_accuracy': direction_accuracy
    }
    
    print(f"\n{model_name} Evaluation Metrics:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R²: {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  Direction Accuracy: {direction_accuracy:.2f}%")
    
    return metrics

def plot_predictions(y_true, y_pred, model_name="Model", save_path=None):
    """
    Create scatter plot of predicted vs actual values.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        model_name: Name of the model
        save_path: Path to save the plot (optional)
        
    Returns:
        Matplotlib figure
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    
    # Add perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.title(f"{model_name}: Predicted vs Actual Stock Prices")
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.grid(True, alpha=0.3)
    
    # Add metrics as text
    from sklearn.metrics import mean_squared_error, r2_score
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    plt.text(0.05, 0.95, f"RMSE: {rmse:.2f}\nR²: {r2:.2f}", 
              transform=plt.gca().transAxes, 
              verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    return plt.gcf()

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate stock price prediction models')
    parser.add_argument('--data', type=str, required=True, help='Path to processed data CSV file')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--output', type=str, default='reports', help='Directory to save reports')
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load data
    logger.info(f"Loading data from {args.data}")
    data =
