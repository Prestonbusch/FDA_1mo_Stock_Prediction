"""
Stock Price Prediction Pipeline

This script runs the complete stock price prediction pipeline:
1. Data processing
2. Feature engineering
3. Model training
4. Model evaluation
"""

import os
import argparse
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run stock price prediction pipeline')
    parser.add_argument('--data', type=str, required=True, help='Path to raw data file')
    parser.add_argument('--models', type=str, default='xgboost,random_forest', 
                        help='Comma-separated list of models to train')
    parser.add_argument('--tune', action='store_true', help='Perform hyperparameter tuning')
    parser.add_argument('--output', type=str, default='output', help='Directory for all outputs')
    return parser.parse_args()

def main():
    """Run the complete pipeline."""
    # Parse arguments
    args = parse_arguments()
    
    # Create directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output, timestamp)
    processed_dir = os.path.join(output_dir, "processed")
    models_dir = os.path.join(output_dir, "models")
    reports_dir = os.path.join(output_dir, "reports")
    
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    # Step 1: Data Processing
    logger.info("Step 1: Data Processing")
    processed_data_path = os.path.join(processed_dir, "processed_stock_data.csv")
    
    data_processing_cmd = f"python -m src.data_processing --input {args.data} --output {processed_dir}"
    logger.info(f"Running: {data_processing_cmd}")
    os.system(data_processing_cmd)
    
    # Step 2 & 3: Feature Engineering & Model Training
    logger.info("Step 2 & 3: Feature Engineering & Model Training")
    
    tune_arg = "--tune" if args.tune else ""
    train_cmd = f"python -m src.model_training --data {processed_data_path} --models {args.models} --output {models_dir} {tune_arg}"
    logger.info(f"Running: {train_cmd}")
    os.system(train_cmd)
    
    # Step 4: Model Evaluation
    logger.info("Step 4: Model Evaluation")
    
    # Evaluate each model
    for model_type in args.models.split(','):
        model_path = os.path.join(models_dir, model_type, f"{model_type}_model.joblib")
        eval_cmd = f"python -m src.model_evaluation --data {processed_data_path} --model {model_path} --output {reports_dir}"
        logger.info(f"Running: {eval_cmd}")
        os.system(eval_cmd)
    
    # Evaluate ensemble if multiple models were trained
    if len(args.models.split(',')) > 1:
        ensemble_path = os.path.join(models_dir, "ensemble")
        eval_cmd = f"python -m src.model_evaluation --data {processed_data_path} --model {ensemble_path} --output {reports_dir}"
        logger.info(f"Running: {eval_cmd}")
        os.system(eval_cmd)
    
    logger.info(f"Pipeline complete. All outputs saved to {output_dir}")
    logger.info(f"  - Processed data: {processed_dir}")
    logger.info(f"  - Trained models: {models_dir}")
    logger.info(f"  - Evaluation reports: {reports_dir}")

if __name__ == "__main__":
    main()
