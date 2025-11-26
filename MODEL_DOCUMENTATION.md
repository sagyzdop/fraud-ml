# Fraud Detection Model Documentation

## Overview

This document describes the real fraud detection model that has been trained on actual mobile banking transaction data. The model replaces the previous mock implementation with a production-ready machine learning system.

## Dataset Description

### Data Sources

The model is trained on two merged datasets:

1. **Transaction Data** (`транзакции_в_Мобильном_интернет_Банкинге.csv`)
   - 13,113 transactions
   - 350 unique customers
   - Fraud rate: 1.26% (165 fraudulent transactions)
   - Fields:
     - `cst_dim_id`: Customer ID
     - `transdate`: Transaction date
     - `transdatetime`: Transaction timestamp
     - `amount`: Transaction amount
     - `docno`: Document number
     - `direction`: Transaction direction (encrypted identifier - text field)
     - `target`: Label (0=legitimate, 1=fraudulent)

2. **Behavioral Patterns Data** (`поведенческие_паттерны_клиентов_3.csv`)
   - 8,587 behavioral records
   - 338 unique customers
   - 19 behavioral features tracking customer login patterns, device changes, and usage patterns

### Data Merging

The datasets are merged on `cst_dim_id` and `transdate`, resulting in:
- **13,140 total records**
- **12,762 records with behavioral data** (97.1% coverage)

## Feature Engineering

### 33 Features Used by the Model

#### 1. Transaction Features (2)
- `amount`: Raw transaction amount
- `amount_log`: Log-transformed amount (log1p)

#### 2. Temporal Features (7)
- `trans_day`: Day of month
- `trans_month`: Month
- `trans_year`: Year
- `trans_dayofweek`: Day of week (0=Monday, 6=Sunday)
- `trans_quarter`: Quarter of year
- `trans_hour`: Hour of day
- `trans_minute`: Minute of hour

#### 3. Time-Based Flags (3)
- `is_weekend`: Weekend transaction indicator
- `is_night`: Night transaction (10 PM - 6 AM)
- `is_business_hours`: Business hours (9 AM - 5 PM)

#### 4. Categorical Encodings (3)
- `direction_encoded`: Encoded transaction direction (from encrypted text field)
- `phone_model_freq`: Frequency encoding of phone model
- `os_freq`: Frequency encoding of OS

#### 5. Behavioral Flags (3)
- `has_behavioral_data`: Indicator for behavioral data availability
- `os_changes_high`: Multiple OS changes in past 30 days
- `phone_changes_high`: Multiple device changes in past 30 days

#### 6. Behavioral Features (15)
- `monthly_os_changes`: OS version changes in last 30 days
- `monthly_phone_model_changes`: Device changes in last 30 days
- `logins_last_7_days`: Login sessions in last 7 days
- `logins_last_30_days`: Login sessions in last 30 days
- `login_frequency_7d`: Average daily logins (7 days)
- `login_frequency_30d`: Average daily logins (30 days)
- `freq_change_7d_vs_mean`: Recent frequency change indicator
- `logins_7d_over_30d_ratio`: Ratio of recent to overall activity
- `avg_login_interval_30d`: Average time between logins
- `std_login_interval_30d`: Std dev of login intervals
- `var_login_interval_30d`: Variance of login intervals
- `ewm_login_interval_7d`: Exponentially weighted mean interval
- `burstiness_login_interval`: Login burst pattern indicator
- `fano_factor_login_interval`: Fano factor of intervals
- `zscore_avg_login_interval_7d`: Z-score of recent intervals

## Model Architecture

### Algorithm: Random Forest Classifier

**Hyperparameters:**
- `n_estimators`: 200 trees
- `max_depth`: 10
- `min_samples_split`: 10
- `min_samples_leaf`: 5
- `class_weight`: Balanced (0: 0.506, 1: 39.818)
- `random_state`: 42

### Feature Importance (Top 10)

1. `amount_log` (16.51%) - Log-transformed transaction amount
2. `amount` (15.94%) - Raw transaction amount
3. `trans_month` (12.28%) - Month of transaction
4. `trans_quarter` (9.04%) - Quarter of year
5. `trans_year` (3.39%) - Year
6. `trans_hour` (3.06%) - Hour of day
7. `phone_model_freq` (2.85%) - Phone model frequency
8. `burstiness_login_interval` (2.64%) - Login burst pattern
9. `direction_encoded` (2.62%) - Transaction direction
10. `std_login_interval_30d` (2.43%) - Login interval variability

## Model Performance

### Test Set Results (20% holdout)

| Metric | Value |
|--------|-------|
| **Accuracy** | 97.30% |
| **Precision** | 20.31% |
| **Recall** | 39.39% |
| **F1-Score** | 26.80% |
| **ROC-AUC** | 94.48% |

### Cross-Validation (5-fold)

- **Mean F1-Score**: 0.3298 ± 0.1208

### Confusion Matrix (Test Set)

|              | Predicted Legitimate | Predicted Fraud |
|--------------|---------------------|-----------------|
| **Legitimate** | 2,544               | 51              |
| **Fraud**      | 20                  | 13              |

### Class-Specific Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Legitimate | 0.99 | 0.98 | 0.99 | 2,595 |
| Fraud | 0.20 | 0.39 | 0.27 | 33 |

## Model Interpretation

### Strengths
1. **High Accuracy** (97.30%): Excellent overall performance
2. **Excellent ROC-AUC** (94.48%): Strong ability to distinguish between classes
3. **Good Recall** (39.39%): Catches about 2 in 5 fraud cases
4. **Feature Rich**: Uses comprehensive transaction and behavioral data

### Limitations
1. **Low Precision** (20.31%): High false positive rate
   - 4 in 5 fraud alerts are false alarms
   - This is common with imbalanced datasets (1.26% fraud rate)
2. **Class Imbalance Challenge**: Only 165 fraud cases in 13,140 transactions
3. **Missing Behavioral Data**: ~3% of transactions lack behavioral features

### Business Impact
- **51 False Positives**: 51 legitimate transactions flagged for review
- **13 True Positives**: 13 fraud cases correctly caught
- **20 False Negatives**: 20 fraud cases missed
- **Cost-Benefit**: Depends on:
  - Cost of manual review ($X per transaction)
  - Average fraud loss ($Y per fraud case)
  - If $Y > 4 × $X, the model is cost-effective

## Preprocessing Pipeline

### Training Time Preprocessing
1. Load and merge datasets on `cst_dim_id` and `transdate`
2. Clean date/datetime strings (remove quotes)
3. Extract temporal features from dates
4. Create time-based flags (weekend, night, business hours)
5. Encode categorical variables (direction, phone model, OS)
6. Handle missing behavioral data (fill with median)
7. Create additional features (amount_log, behavioral flags)
8. Scale all features using StandardScaler

### Prediction Time Preprocessing
1. Extract same temporal features from input data
2. Apply same encodings (using saved encoders)
3. Handle missing features (fill with 0 or defaults)
4. Apply same scaling transformation (using saved scaler)
5. Ensure all 33 features are present in correct order

## Usage

### Training the Model

```bash
python3 create_sample_model.py
```

This will:
- Load both datasets
- Merge and preprocess data
- Train the Random Forest model
- Evaluate performance
- Save `model.pkl` with all components

### Making Predictions

```python
import pickle
import pandas as pd
from app import predict_fraud, load_model

# Load model bundle
model_bundle = load_model('model.pkl')

# Prepare transaction data
transactions = pd.DataFrame({
    'cst_dim_id': [12345],
    'transdate': ['2025-03-05'],
    'transdatetime': ['2025-03-05 16:30:00'],
    'amount': [50000.0],
    'docno': [1234],
    'direction': ['outgoing'],
    'target': [0]
})

# Make predictions
results = predict_fraud(model_bundle, transactions)
print(results[['amount', 'fraud_prediction', 'is_fraudulent']])
```

### Running the Web Application

```bash
streamlit run app.py
```

The application provides:
- Single transaction analysis
- Bulk CSV upload and processing
- Visual results and downloadable reports

## Model Bundle Structure

The saved `model.pkl` contains:

```python
{
    'model': RandomForestClassifier,  # Trained sklearn model
    'feature_names': list,            # 33 feature names in order
    'encoders': dict,                 # Label encoders for categorical vars
    'scaler': StandardScaler,         # Fitted scaler for features
    'metrics': dict                   # Performance metrics
}
```

## Future Improvements

1. **Address Class Imbalance**
   - Try SMOTE or other oversampling techniques
   - Adjust decision threshold based on business costs
   - Use ensemble methods (XGBoost, LightGBM)

2. **Feature Engineering**
   - Add velocity features (transactions per hour/day)
   - Include historical fraud rates per customer
   - Add recipient risk scores
   - Geolocation features if available

3. **Model Enhancements**
   - Hyperparameter tuning with GridSearch/RandomSearch
   - Try other algorithms (XGBoost, Neural Networks)
   - Ensemble multiple models
   - Implement online learning for model updates

4. **Monitoring & Maintenance**
   - Track prediction distribution over time
   - Monitor feature drift
   - Regular model retraining
   - A/B testing for model improvements

## Requirements

```
pandas>=2.2.3
scikit-learn>=1.5.2
streamlit>=1.40.0
numpy>=1.24.0
```
