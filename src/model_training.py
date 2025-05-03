"""
Model Training Module for Stock Price Prediction

This module handles training and tuning of various machine learning models for stock price prediction.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
import logging
import os
import joblib
from datetime import datetime
import argparse

# Machine learning models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
            model = XGBRegressor(n_jobs=-1, **self.model_params)
        elif self.model_type == 'lightgbm':
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
        
    def tune_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series, 
                           param_grid: Dict, cv: int = 5, 
                           search_type: str = 'grid', n_iter: int = 10) -> Any:
        """
        Tune model hyperparameters using grid or random search.
        
        Args:
            X_train: Training features
            y_train: Training target
            param_grid: Dictionary of parameters to search
            cv: Number of cross-validation folds
            search_type: Type of search ('grid' or 'random')
            n_iter: Number of parameter settings for random search
            
        Returns:
            Best model found
        """
        logger.info(f"Tuning hyperparameters using {search_type} search with {cv}-fold CV")
        
        model = self.get_model()
        
        # Initialize search
        if search_type.lower() == 'grid':
            search = GridSearchCV(
                model, param_grid, cv=cv, n_jobs=-1,
                scoring='neg_mean_squared_error',
                verbose=1
            )
        elif search_type.lower() == 'random':
            search = RandomizedSearchCV(
                model, param_grid, n_iter=n_iter, cv=cv, n_jobs=-1,
                scoring='neg_mean_squared_error',
                verbose=1, random_state=42
            )
        else:
            raise ValueError(f"Unknown search type: {search_type}")
            
        # Run search
        search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {search.best_params_}")
        logger.info(f"Best score: {search.best_score_}")
        
        # Update model with best parameters
        self.model = search.best_estimator_
        self.model_params = search.best_params_
        
        return self.model
        
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


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train stock price prediction models')
    parser.add_argument('--data', type=str, required=True, help='Path to processed data CSV file')
    parser.add_argument('--models', type=str, default='xgboost,random_forest', 
                        help='Comma-separated list of models to train')
    parser.add_argument('--output', type=str, default='models', help='Directory to save models')
    parser.add_argument('--tune', action='store_true', help='Perform hyperparameter tuning')
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Load processed data
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
    
    # Train models
    model_types = args.models.split(',')
    trained_models = {}
    
    for model_type in model_types:
        logger.info(f"Training {model_type} model")
        
        # Create and train model
        model = StockPriceModel(model_type=model_type)
        
        if args.tune:
            # Define hyperparameter grid based on model type
            if model_type == 'xgboost':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'colsample_bytree': [0.6, 0.8, 1.0],
                    'subsample': [0.6, 0.8, 1.0]
                }
            elif model_type == 'random_forest':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            elif model_type in ['ridge', 'lasso', 'elastic_net']:
                param_grid = {
                    'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
                }
            else:
                logger.warning(f"No tuning parameters defined for {model_type}, using default parameters")
                param_grid = {}
            
            if param_grid:
                model.tune_hyperparameters(X_train_scaled, y_train, param_grid, search_type='random')
            else:
                model.train(X_train_scaled, y_train)
        else:
            model.train(X_train_scaled, y_train)
        
        # Evaluate on validation set
        metrics = model.evaluate(X_val_scaled, y_val)
        
        # Save model
        model_dir = os.path.join(args.output, model_type)
        os.makedirs(model_dir, exist_ok=True)
        model_path = model.save_model(model_dir, f"{model_type}_model.joblib")
        
        trained_models[model_type] = (model, metrics)
    
    # Create and train ensemble model
    logger.info("Creating model ensemble")
    ensemble = StockPriceModelEnsemble([model for model, _ in trained_models.values()])
    
    # Evaluate ensemble
    ensemble_metrics = ensemble.evaluate(X_val_scaled, y_val)
    
    # Save ensemble
    ensemble_dir = os.path.join(args.output, "ensemble")
    ensemble.save_models(ensemble_dir)
    
    # Print summary
    print("\nModel Training Summary:")
    print(f"Data: {args.data}")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    print("\nModel Performance (RMSE, lower is better):")
    
    for model_type, (_, metrics) in trained_models.items():
        print(f"  {model_type}: {metrics['rmse']:.4f}")
    
    print(f"  ensemble: {ensemble_metrics['rmse']:.4f}")
    
    print("\nModel Paths:")
    for model_type in trained_models.keys():
        print(f"  {model_type}: {args.output}/{model_type}/{model_type}_model.joblib")
    
    print(f"  ensemble: {args.output}/ensemble/")
