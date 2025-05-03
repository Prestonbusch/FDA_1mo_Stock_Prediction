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
    data =if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load data
    logger.info(f"Loading data from {args.data}")
    data = pd.read_csv(args.data)
    
    # Process and prepare features
    from src.data_processing import StockDataProcessor
    from src.feature_engineering import StockFeatureEngineer
    
    processor = StockDataProcessor()
    engineer = StockFeatureEngineer()
    
    # Feature engineering
    processed_data = processor.clean_data(data)
    with_tech = engineer.create_technical_indicators(processed_data)
    with_market = engineer.create_market_features(with_tech)
    with_fundamental = engineer.create_fundamental_features(with_market)
    
    # Prepare features and split data
    X, y = processor.prepare_features_targets(with_fundamental, prediction_window=1)
    X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(X, y)
    
    # Scale features
    X_train_scaled, X_val_scaled, X_test_scaled = engineer.scale_features(X_train, X_val, X_test)
    
    # Load model
    logger.info(f"Loading model from {args.model}")
    if 'ensemble' in args.model.lower():
        model = StockPriceModelEnsemble.load_ensemble(args.model)
        model_name = "Ensemble Model"
    else:
        model = StockPriceModel.load_model(args.model)
        model_name = f"{model.model_type.capitalize()} Model"
    
    # Evaluate model on validation set
    logger.info("Evaluating model on validation set")
    val_metrics = evaluate_model(model, X_val_scaled, y_val, f"{model_name} (Validation)")
    
    # Evaluate model on test set
    logger.info("Evaluating model on test set")
    test_metrics = evaluate_model(model, X_test_scaled, y_test, f"{model_name} (Test)")
    
    # Plot predictions
    if isinstance(model, StockPriceModel):
        y_pred_test = model.predict(X_test_scaled)
    else:
        y_pred_test = model.predict(X_test_scaled, method='mean')
        
    plot_path = os.path.join(args.output, f"{model_name.lower().replace(' ', '_')}_predictions.png")
    plot_predictions(y_test, y_pred_test, model_name, plot_path)
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'Metric': ['RMSE', 'MAE', 'R2', 'MAPE', 'Direction Accuracy'],
        'Validation': [val_metrics['rmse'], val_metrics['mae'], val_metrics['r2'], 
                      val_metrics['mape'], val_metrics['direction_accuracy']],
        'Test': [test_metrics['rmse'], test_metrics['mae'], test_metrics['r2'], 
                test_metrics['mape'], test_metrics['direction_accuracy']]
    })
    
    metrics_path = os.path.join(args.output, f"{model_name.lower().replace(' ', '_')}_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    
    # If model is tree-based, plot feature importance
    if hasattr(model, 'plot_feature_importance'):
        try:
            fig = model.plot_feature_importance(top_n=20)
            if fig:
                importance_path = os.path.join(args.output, f"{model_name.lower().replace(' ', '_')}_feature_importance.png")
                fig.savefig(importance_path, dpi=300, bbox_inches='tight')
                logger.info(f"Feature importance saved to {importance_path}")
        except Exception as e:
            logger.warning(f"Could not plot feature importance: {str(e)}")
    
    # Print summary
    print("\nEvaluation Summary:")
    print(f"Model: {model_name}")
    print(f"Data: {args.data}")
    print(f"Test samples: {len(X_test)}")
    print(f"RMSE (Test): {test_metrics['rmse']:.4f}")
    print(f"R² (Test): {test_metrics['r2']:.4f}")
    print(f"Direction Accuracy (Test): {test_metrics['direction_accuracy']:.2f}%")
    print(f"Reports saved to: {args.output}")
