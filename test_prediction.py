"""
Test script to verify the prediction pipeline works correctly
"""

import pickle
import pandas as pd
from datetime import datetime
import sys

# Import preprocessing function from app
sys.path.insert(0, '.')
from app import preprocess_for_model, predict_fraud, load_model

def test_prediction():
    print("="*60)
    print("TESTING PREDICTION PIPELINE")
    print("="*60)
    
    # Load model
    print("\n1. Loading model...")
    model_bundle = load_model('model.pkl')
    print(f"   Model loaded successfully!")
    print(f"   Model type: {type(model_bundle['model'])}")
    print(f"   Number of features: {len(model_bundle['feature_names'])}")
    
    # Create test data (legitimate transaction)
    print("\n2. Creating test data...")
    test_data = pd.DataFrame({
        'cst_dim_id': [453024373],
        'transdate': [datetime(2025, 3, 5)],
        'transdatetime': [datetime(2025, 3, 5, 16, 30)],
        'amount': [5000.0],
        'docno': [1234],
        'direction': ['8406e407421ec28bd5f445793ef64fd1'],  # Encrypted text value
    })
    print(f"   Test data created: {test_data.shape}")
    
    # Run prediction
    print("\n3. Running prediction...")
    result = predict_fraud(model_bundle, test_data)
    print(f"   Prediction complete!")
    print(f"\n   Result:")
    print(f"   - Amount: {result['amount'].iloc[0]}")
    print(f"   - Direction: {result['direction'].iloc[0]}")
    print(f"   - Fraud Prediction: {result['fraud_prediction'].iloc[0]}")
    print(f"   - Is Fraudulent: {result['is_fraudulent'].iloc[0]}")
    
    # Create suspicious transaction
    print("\n4. Testing with suspicious transaction...")
    suspicious_data = pd.DataFrame({
        'cst_dim_id': [999999999],
        'transdate': [datetime(2025, 11, 26)],
        'transdatetime': [datetime(2025, 11, 26, 3, 15)],  # Late night
        'amount': [150000.0],  # Large amount
        'docno': [9999],
        'direction': ['b3a3d4a6006293195d998957d4f01e42'],  # Encrypted text value
    })
    
    result2 = predict_fraud(model_bundle, suspicious_data)
    print(f"   Result:")
    print(f"   - Amount: {result2['amount'].iloc[0]}")
    print(f"   - Direction: {result2['direction'].iloc[0]}")
    print(f"   - Transaction Hour: 3 AM")
    print(f"   - Fraud Prediction: {result2['fraud_prediction'].iloc[0]}")
    print(f"   - Is Fraudulent: {result2['is_fraudulent'].iloc[0]}")
    
    print("\n" + "="*60)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("="*60)

if __name__ == "__main__":
    test_prediction()
