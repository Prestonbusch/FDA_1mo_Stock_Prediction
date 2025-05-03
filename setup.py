#!/usr/bin/env python
"""
Setup script for FDA_1mo_Stock_Prediction repository.
This script creates the directory structure and initial files for the project.
"""

import os
import json
import shutil
from pathlib import Path

def create_directory(path):
    """Create a directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)
    print(f"Created directory: {path}")

def create_file(path, content=""):
    """Create a file with the given content."""
    with open(path, 'w') as f:
        f.write(content)
    print(f"Created file: {path}")

def create_gitkeep(path):
    """Create .gitkeep file in the directory."""
    create_file(os.path.join(path, '.gitkeep'))

def setup_repository():
    """Set up the repository structure for FDA_1mo_Stock_Prediction."""
    # Project root
    project_root = "FDA_1mo_Stock_Prediction"
    
    # Create main project directory
    create_directory(project_root)
    
    # Create subdirectories
    directories = [
        os.path.join(project_root, "data/raw"),
        os.path.join(project_root, "data/processed"),
        os.path.join(project_root, "models/xgboost"),
        os.path.join(project_root, "models/random_forest"),
        os.path.join(project_root, "models/ensemble"),
        os.path.join(project_root, "src"),
        os.path.join(project_root, "notebooks"),
        os.path.join(project_root, "app/pages"),
        os.path.join(project_root, "scripts/params"),
        os.path.join(project_root, "docs")
    ]
    
    for directory in directories:
        create_directory(directory)
        create_gitkeep(directory)
    
    # Create README.md
    readme_content = """# FDA_1mo_Stock_Prediction

A machine learning application for predicting future stock prices based on WRDS (Wharton Research Data Services) data.

## Overview

FDA_1mo_Stock_Prediction is a comprehensive machine learning solution that leverages historical stock data from WRDS to predict future stock prices. The application uses advanced feature engineering techniques and various machine learning algorithms to identify patterns and make predictions. This project includes:

- Data processing pipeline for WRDS stock data
- Feature engineering module with technical, fundamental, and market indicators
- Multiple machine learning models (XGBoost, Random Forest, Linear Regression, etc.)
- Model ensemble for improved prediction accuracy
- Interactive Streamlit web application for visualization and prediction

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Data Requirements](#data-requirements)
- [Model Training](#model-training)
- [Web Application](#web-application)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Features

### Data Processing
- Automated cleaning and preprocessing of WRDS stock data
- Handling of missing values and outliers
- Time series data preparation

### Feature Engineering
- Technical indicators (Moving Averages, RSI, Bollinger Bands, etc.)
- Market features (Market returns, volatility, etc.)
- Fundamental analysis metrics
- Advanced feature selection

### Machine Learning Models
- XGBoost for high accuracy prediction
- Random Forest for robust ensemble learning
- Linear models (Linear Regression, Ridge, Lasso)
- Model ensemble for improved performance

### Streamlit Web Application
- Interactive visualization of stock data
- Prediction configuration and execution
- Model insights and performance metrics
- Downloadable prediction results

## Installation

### Requirements
- Python 3.8+
- pip or conda for package management

### Setup

1. Clone the repository:
```bash
git clone https://github.com/prestonbusch/FDA_1mo_Stock_Prediction.git
cd FDA_1mo_Stock_Prediction
