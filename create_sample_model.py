"""
Real Fraud Detection Model Training Script

This script trains a machine learning model for fraud detection using real transaction
and behavioral pattern data from mobile banking transactions.
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score, 
    precision_recall_curve,
    f1_score,
    fbeta_score,
    accuracy_score,
    precision_score,
    recall_score
)
from sklearn.utils import class_weight
import warnings
warnings.filterwarnings('ignore')


def load_and_merge_datasets(
    transactions_path: str = "транзакции_в_Мобильном_интернет_Банкинге.csv",
    behavioral_path: str = "поведенческие_паттерны_клиентов_3.csv"
) -> pd.DataFrame:
    """
    Load and merge transaction and behavioral pattern datasets.
    
    Args:
        transactions_path: Path to transactions CSV file
        behavioral_path: Path to behavioral patterns CSV file
    
    Returns:
        Merged DataFrame with all features
    """
    print("Loading datasets...")
    
    # Load transactions data
    df_transactions = pd.read_csv(transactions_path, sep=';')
    print(f"Loaded {len(df_transactions)} transactions")
    
    # Load behavioral patterns data
    df_behavioral = pd.read_csv(behavioral_path, sep=';')
    print(f"Loaded {len(df_behavioral)} behavioral records")
    
    # Clean transdate column (remove quotes if present)
    for df in [df_transactions, df_behavioral]:
        if 'transdate' in df.columns:
            df['transdate'] = df['transdate'].astype(str).str.strip("'")
            df['transdate'] = pd.to_datetime(df['transdate'], errors='coerce')
    
    # Clean transdatetime column
    if 'transdatetime' in df_transactions.columns:
        df_transactions['transdatetime'] = df_transactions['transdatetime'].astype(str).str.strip("'")
        df_transactions['transdatetime'] = pd.to_datetime(df_transactions['transdatetime'], errors='coerce')
    
    # Merge datasets on customer ID and transaction date
    print("Merging datasets on cst_dim_id and transdate...")
    df_merged = pd.merge(
        df_transactions,
        df_behavioral,
        on=['cst_dim_id', 'transdate'],
        how='left'
    )
    
    print(f"Merged dataset size: {len(df_merged)} records")
    print(f"Records with behavioral data: {df_merged['monthly_os_changes'].notna().sum()}")
    
    return df_merged


def preprocess_data(df: pd.DataFrame) -> tuple:
    """
    Preprocess the merged dataset for model training.
    
    Args:
        df: Merged DataFrame
    
    Returns:
        Tuple of (X, y, feature_names, encoders, scaler)
    """
    print("\nPreprocessing data...")
    
    # Create a copy to avoid modifying original
    data = df.copy()
    
    # Remove records with missing target
    data = data[data['target'].notna()].copy()
    print(f"Records after removing missing targets: {len(data)}")
    
    # Extract datetime features from transdate
    data['trans_day'] = data['transdate'].dt.day
    data['trans_month'] = data['transdate'].dt.month
    data['trans_year'] = data['transdate'].dt.year
    data['trans_dayofweek'] = data['transdate'].dt.dayofweek
    data['trans_quarter'] = data['transdate'].dt.quarter
    
    # Extract datetime features from transdatetime
    data['trans_hour'] = data['transdatetime'].dt.hour
    data['trans_minute'] = data['transdatetime'].dt.minute
    
    # Create time-based features
    data['is_weekend'] = data['trans_dayofweek'].isin([5, 6]).astype(int)
    data['is_night'] = ((data['trans_hour'] >= 22) | (data['trans_hour'] <= 6)).astype(int)
    data['is_business_hours'] = ((data['trans_hour'] >= 9) & (data['trans_hour'] <= 17)).astype(int)
    
    # Handle categorical variables
    encoders = {}
    
    # Encode direction
    if 'direction' in data.columns:
        le_direction = LabelEncoder()
        data['direction_encoded'] = le_direction.fit_transform(data['direction'].fillna('unknown'))
        encoders['direction'] = le_direction
    
    # Encode phone model and OS (high cardinality - use frequency encoding)
    if 'last_phone_model_categorical' in data.columns:
        phone_freq = data['last_phone_model_categorical'].value_counts(normalize=True)
        data['phone_model_freq'] = data['last_phone_model_categorical'].map(phone_freq).fillna(0)
    
    if 'last_os_categorical' in data.columns:
        os_freq = data['last_os_categorical'].value_counts(normalize=True)
        data['os_freq'] = data['last_os_categorical'].map(os_freq).fillna(0)
    
    # Fill missing behavioral features with median or 0
    behavioral_features = [
        'monthly_os_changes', 'monthly_phone_model_changes',
        'logins_last_7_days', 'logins_last_30_days',
        'login_frequency_7d', 'login_frequency_30d',
        'freq_change_7d_vs_mean', 'logins_7d_over_30d_ratio',
        'avg_login_interval_30d', 'std_login_interval_30d',
        'var_login_interval_30d', 'ewm_login_interval_7d',
        'burstiness_login_interval', 'fano_factor_login_interval',
        'zscore_avg_login_interval_7d'
    ]
    
    for feature in behavioral_features:
        if feature in data.columns:
            # Convert to numeric first (handles object types)
            data[feature] = pd.to_numeric(data[feature], errors='coerce')
            # Then fill missing values with median
            data[feature] = data[feature].fillna(data[feature].median())
    
    # Create additional features
    data['amount_log'] = np.log1p(data['amount'])
    
    # Create behavioral pattern flags
    if 'monthly_os_changes' in data.columns:
        data['has_behavioral_data'] = data['monthly_os_changes'].notna().astype(int)
        data['os_changes_high'] = (data['monthly_os_changes'] > 1).astype(int)
        data['phone_changes_high'] = (data['monthly_phone_model_changes'] > 1).astype(int)
    else:
        data['has_behavioral_data'] = 0
        data['os_changes_high'] = 0
        data['phone_changes_high'] = 0
    
    # Select features for model
    feature_columns = [
        'amount', 'amount_log',
        'trans_day', 'trans_month', 'trans_year', 'trans_dayofweek', 'trans_quarter',
        'trans_hour', 'trans_minute',
        'is_weekend', 'is_night', 'is_business_hours',
        'direction_encoded',
        'has_behavioral_data',
    ]
    
    # Add phone and OS frequency if available
    if 'phone_model_freq' in data.columns:
        feature_columns.append('phone_model_freq')
    if 'os_freq' in data.columns:
        feature_columns.append('os_freq')
    
    # Add behavioral features if available
    if 'os_changes_high' in data.columns:
        feature_columns.extend(['os_changes_high', 'phone_changes_high'])
    
    for feature in behavioral_features:
        if feature in data.columns:
            feature_columns.append(feature)
    
    # Prepare X and y
    X = data[feature_columns].copy()
    y = data['target'].copy()
    
    # Handle any remaining NaN values
    X = X.fillna(0)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_columns, index=X.index)
    
    print(f"Final feature set: {len(feature_columns)} features")
    print(f"Feature names: {feature_columns}")
    print(f"\nTarget distribution:")
    print(y.value_counts())
    print(f"Fraud rate: {(y.sum() / len(y) * 100):.2f}%")
    
    return X_scaled, y, feature_columns, encoders, scaler


def train_model(X: pd.DataFrame, y: pd.Series, model_type: str = 'random_forest'):
    """
    Train a fraud detection model.
    
    Args:
        X: Feature matrix
        y: Target vector
        model_type: Type of model to train ('random_forest' or 'gradient_boosting')
    
    Returns:
        Trained model and evaluation metrics
    """
    print(f"\nTraining {model_type} model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Training set fraud rate: {(y_train.sum() / len(y_train) * 100):.2f}%")
    print(f"Test set fraud rate: {(y_test.sum() / len(y_test) * 100):.2f}%")
    
    # Calculate class weights for imbalanced dataset
    classes = np.unique(y_train)
    weights = class_weight.compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))
    print(f"Class weights: {class_weights}")
    
    # Initialize model
    if model_type == 'gradient_boosting':
        model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
            verbose=0
        )
    else:  # random_forest
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight=class_weights,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
    
    # Train model
    print("Training in progress...")
    model.fit(X_train, y_train)
    
    # Evaluate on training set
    y_train_pred = model.predict(X_train)
    print("\n=== Training Set Performance ===")
    print(f"Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
    print(f"Precision: {precision_score(y_train, y_train_pred):.4f}")
    print(f"Recall: {recall_score(y_train, y_train_pred):.4f}")
    print(f"F1-Score: {f1_score(y_train, y_train_pred):.4f}")
    
    # Evaluate on test set
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate F_β scores with different beta values
    beta_values = [0.5, 1.0, 2.0]  # β=0.5 (precision-focused), β=1 (F1), β=2 (recall-focused)
    f_beta_scores = {}
    for beta in beta_values:
        f_beta = fbeta_score(y_test, y_pred, beta=beta)
        f_beta_scores[f"f{beta}_score"] = f_beta
    
    print("\n=== Test Set Performance ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
    
    # Display F_β scores
    print(f"\n=== F_β Scores (Different β values) ===")
    print(f"F_0.5-Score (precision-focused): {f_beta_scores['f0.5_score']:.4f}")
    print(f"F_1.0-Score (balanced):          {f_beta_scores['f1.0_score']:.4f}")
    print(f"F_2.0-Score (recall-focused):    {f_beta_scores['f2.0_score']:.4f}")
    print(f"\nβ interpretation:")
    print(f"  β < 1: Emphasizes precision (fewer false positives)")
    print(f"  β = 1: Balanced F1 score")
    print(f"  β > 1: Emphasizes recall (catch more fraud cases)")
    
    print(f"\nROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))
    
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n=== Top 10 Most Important Features ===")
        print(feature_importance.head(10))
    
    # Cross-validation score
    print("\n=== Cross-Validation (5-fold) ===")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
    print(f"F1 Scores: {cv_scores}")
    print(f"Mean F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'cv_f1_mean': cv_scores.mean(),
        'cv_f1_std': cv_scores.std(),
        **f_beta_scores  # Include all F_β scores
    }
    
    return model, metrics, (X_test, y_test, y_pred, y_pred_proba)


def create_model_pipeline():
    """
    Complete pipeline to create and save the fraud detection model.
    """
    print("="*60)
    print("FRAUD DETECTION MODEL TRAINING PIPELINE")
    print("="*60)
    
    # Load and merge data
    df_merged = load_and_merge_datasets()
    
    # Preprocess data
    X, y, feature_names, encoders, scaler = preprocess_data(df_merged)
    
    # Train model
    model, metrics, test_data = train_model(X, y, model_type='random_forest')
    
    # Create model bundle with preprocessing components
    model_bundle = {
        'model': model,
        'feature_names': feature_names,
        'encoders': encoders,
        'scaler': scaler,
        'metrics': metrics
    }
    
    # Save model
    model_path = "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model_bundle, f)
    
    print(f"\n{'='*60}")
    print(f"Model successfully trained and saved to '{model_path}'")
    print(f"{'='*60}")
    print("\nModel Performance Summary:")
    print(f"  - Accuracy: {metrics['accuracy']:.4f}")
    print(f"  - Precision: {metrics['precision']:.4f}")
    print(f"  - Recall: {metrics['recall']:.4f}")
    print(f"  - F1-Score: {metrics['f1_score']:.4f}")
    print(f"  - ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"\nThe model is ready to use for fraud detection!")
    
    return model_bundle


# Backwards compatibility
def create_sample_model():
    """Alias for create_model_pipeline for backwards compatibility."""
    return create_model_pipeline()


if __name__ == "__main__":
    model_bundle = create_model_pipeline()
