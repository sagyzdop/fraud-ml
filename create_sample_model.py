"""
Script to generate a sample ML model for testing the fraud detection application.

This creates a simple Random Forest classifier trained on synthetic data
that mimics the expected input format.
"""

import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def create_sample_model():
    """Create a sample Random Forest model for fraud detection testing."""
    # Create synthetic training data matching the preprocessed feature set
    np.random.seed(42)
    n_samples = 1000
    
    # Features after preprocessing:
    # cst_dim_id, amount, target, trans_day, trans_month, trans_year, 
    # trans_dayofweek, trans_hour, trans_minute, direction
    
    X = np.random.rand(n_samples, 10)
    
    # Create synthetic labels (10% fraud rate)
    y = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])
    
    # Train a simple Random Forest model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
    model.fit(X, y)
    
    # Save the model
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    print("Sample model created and saved as 'model.pkl'")
    return model


if __name__ == "__main__":
    create_sample_model()
