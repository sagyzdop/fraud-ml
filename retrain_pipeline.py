"""
Automated Model Retraining Pipeline

This script provides automated retraining functionality for the fraud detection model.
It can be scheduled to run periodically or triggered when new data becomes available.
"""

import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
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
import json
warnings.filterwarnings('ignore')


class ModelRetrainingPipeline:
    """
    Automated pipeline for retraining fraud detection models with new data.
    """
    
    def __init__(
        self, 
        config_path: str = "retrain_config.json",
        model_path: str = "model.pkl",
        backup_dir: str = "model_backups"
    ):
        """
        Initialize the retraining pipeline.
        
        Args:
            config_path: Path to configuration file
            model_path: Path to current model file
            backup_dir: Directory for model backups
        """
        self.config_path = config_path
        self.model_path = model_path
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
        
        # Load configuration
        self.config = self._load_config()
        
        # Tracking metrics
        self.training_history = []
        
    def _load_config(self) -> Dict:
        """Load retraining configuration."""
        default_config = {
            "transactions_path": "транзакции_в_Мобильном_интернет_Банкинге.csv",
            "behavioral_path": "поведенческие_паттерны_клиентов_3.csv",
            "model_type": "random_forest",
            "test_size": 0.2,
            "random_state": 42,
            "n_estimators": 200,
            "max_depth": 10,
            "min_samples_split": 10,
            "min_samples_leaf": 5,
            "beta_for_fbeta": 2.0,
            "performance_threshold": {
                "min_roc_auc": 0.85,
                "min_recall": 0.30,
                "min_precision": 0.15
            },
            "auto_deploy": False,
            "backup_old_model": True
        }
        
        if Path(self.config_path).exists():
            with open(self.config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def save_config(self):
        """Save current configuration to file."""
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        print(f"Configuration saved to {self.config_path}")
    
    def load_and_merge_datasets(self) -> pd.DataFrame:
        """Load and merge transaction and behavioral data."""
        print("Loading datasets...")
        
        # Load transactions
        df_transactions = pd.read_csv(
            self.config['transactions_path'], 
            sep=';'
        )
        print(f"  - Loaded {len(df_transactions)} transactions")
        
        # Load behavioral patterns
        df_behavioral = pd.read_csv(
            self.config['behavioral_path'], 
            sep=';'
        )
        print(f"  - Loaded {len(df_behavioral)} behavioral records")
        
        # Clean and merge
        for df in [df_transactions, df_behavioral]:
            if 'transdate' in df.columns:
                df['transdate'] = df['transdate'].astype(str).str.strip("'")
                df['transdate'] = pd.to_datetime(df['transdate'], errors='coerce')
        
        if 'transdatetime' in df_transactions.columns:
            df_transactions['transdatetime'] = df_transactions['transdatetime'].astype(str).str.strip("'")
            df_transactions['transdatetime'] = pd.to_datetime(df_transactions['transdatetime'], errors='coerce')
        
        df_merged = pd.merge(
            df_transactions,
            df_behavioral,
            on=['cst_dim_id', 'transdate'],
            how='left'
        )
        
        print(f"  - Merged dataset: {len(df_merged)} records")
        return df_merged
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple:
        """Preprocess data for training."""
        print("\nPreprocessing data...")
        
        data = df.copy()
        data = data[data['target'].notna()].copy()
        
        # Extract temporal features
        data['trans_day'] = data['transdate'].dt.day
        data['trans_month'] = data['transdate'].dt.month
        data['trans_year'] = data['transdate'].dt.year
        data['trans_dayofweek'] = data['transdate'].dt.dayofweek
        data['trans_quarter'] = data['transdate'].dt.quarter
        data['trans_hour'] = data['transdatetime'].dt.hour
        data['trans_minute'] = data['transdatetime'].dt.minute
        
        # Time-based features
        data['is_weekend'] = data['trans_dayofweek'].isin([5, 6]).astype(int)
        data['is_night'] = ((data['trans_hour'] >= 22) | (data['trans_hour'] <= 6)).astype(int)
        data['is_business_hours'] = ((data['trans_hour'] >= 9) & (data['trans_hour'] <= 17)).astype(int)
        
        # Encode categoricals
        encoders = {}
        if 'direction' in data.columns:
            # Convert to string type to handle encrypted text values
            data['direction'] = data['direction'].fillna('unknown').astype(str)
            le_direction = LabelEncoder()
            data['direction_encoded'] = le_direction.fit_transform(data['direction'])
            encoders['direction'] = le_direction
        
        # Frequency encoding
        if 'last_phone_model_categorical' in data.columns:
            phone_freq = data['last_phone_model_categorical'].value_counts(normalize=True)
            data['phone_model_freq'] = data['last_phone_model_categorical'].map(phone_freq).fillna(0)
        
        if 'last_os_categorical' in data.columns:
            os_freq = data['last_os_categorical'].value_counts(normalize=True)
            data['os_freq'] = data['last_os_categorical'].map(os_freq).fillna(0)
        
        # Fill behavioral features
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
                data[feature] = pd.to_numeric(data[feature], errors='coerce')
                data[feature] = data[feature].fillna(data[feature].median())
        
        # Additional features
        data['amount_log'] = np.log1p(data['amount'])
        
        if 'monthly_os_changes' in data.columns:
            data['has_behavioral_data'] = data['monthly_os_changes'].notna().astype(int)
            data['os_changes_high'] = (data['monthly_os_changes'] > 1).astype(int)
            data['phone_changes_high'] = (data['monthly_phone_model_changes'] > 1).astype(int)
        else:
            data['has_behavioral_data'] = 0
            data['os_changes_high'] = 0
            data['phone_changes_high'] = 0
        
        # Select features
        feature_columns = [
            'amount', 'amount_log',
            'trans_day', 'trans_month', 'trans_year', 'trans_dayofweek', 'trans_quarter',
            'trans_hour', 'trans_minute',
            'is_weekend', 'is_night', 'is_business_hours',
            'direction_encoded',
            'has_behavioral_data',
        ]
        
        if 'phone_model_freq' in data.columns:
            feature_columns.append('phone_model_freq')
        if 'os_freq' in data.columns:
            feature_columns.append('os_freq')
        
        feature_columns.extend(['os_changes_high', 'phone_changes_high'])
        
        for feature in behavioral_features:
            if feature in data.columns:
                feature_columns.append(feature)
        
        X = data[feature_columns].copy().fillna(0)
        y = data['target'].copy()
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_columns, index=X.index)
        
        print(f"  - Features: {len(feature_columns)}")
        print(f"  - Samples: {len(X_scaled)}")
        print(f"  - Fraud rate: {(y.sum() / len(y) * 100):.2f}%")
        
        return X_scaled, y, feature_columns, encoders, scaler
    
    def train_new_model(self, X: pd.DataFrame, y: pd.Series) -> Tuple:
        """Train a new model."""
        print("\nTraining new model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config['test_size'], 
            random_state=self.config['random_state'], 
            stratify=y
        )
        
        print(f"  - Training set: {len(X_train)} samples")
        print(f"  - Test set: {len(X_test)} samples")
        
        # Calculate class weights
        classes = np.unique(y_train)
        weights = class_weight.compute_class_weight('balanced', classes=classes, y=y_train)
        class_weights = dict(zip(classes, weights))
        
        # Initialize and train model
        model = RandomForestClassifier(
            n_estimators=self.config['n_estimators'],
            max_depth=self.config['max_depth'],
            min_samples_split=self.config['min_samples_split'],
            min_samples_leaf=self.config['min_samples_leaf'],
            class_weight=class_weights,
            random_state=self.config['random_state'],
            n_jobs=-1,
            verbose=0
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        beta = self.config['beta_for_fbeta']
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'f_beta_score': fbeta_score(y_test, y_pred, beta=beta, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'beta_value': beta,
            'test_size': len(X_test),
            'train_size': len(X_train),
            'fraud_rate_test': float(y_test.sum() / len(y_test))
        }
        
        print("\n=== Model Performance ===")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        print(f"  F_{beta}-Score: {metrics['f_beta_score']:.4f}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        return model, metrics, (X_test, y_test, y_pred, y_pred_proba)
    
    def validate_model_performance(self, metrics: Dict) -> Tuple[bool, List[str]]:
        """
        Validate if new model meets performance thresholds.
        
        Returns:
            (passes_validation, list_of_issues)
        """
        issues = []
        threshold = self.config['performance_threshold']
        
        if metrics['roc_auc'] < threshold['min_roc_auc']:
            issues.append(
                f"ROC-AUC {metrics['roc_auc']:.4f} below threshold {threshold['min_roc_auc']}"
            )
        
        if metrics['recall'] < threshold['min_recall']:
            issues.append(
                f"Recall {metrics['recall']:.4f} below threshold {threshold['min_recall']}"
            )
        
        if metrics['precision'] < threshold['min_precision']:
            issues.append(
                f"Precision {metrics['precision']:.4f} below threshold {threshold['min_precision']}"
            )
        
        passes = len(issues) == 0
        return passes, issues
    
    def backup_current_model(self):
        """Create backup of current model before replacing."""
        if not Path(self.model_path).exists():
            print("No existing model to backup")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"model_backup_{timestamp}.pkl"
        
        import shutil
        shutil.copy2(self.model_path, backup_path)
        print(f"  ✅ Backed up current model to {backup_path}")
    
    def save_model(self, model_bundle: Dict, metrics: Dict):
        """Save new model and update training history."""
        # Save model
        with open(self.model_path, 'wb') as f:
            pickle.dump(model_bundle, f)
        
        # Update training history
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'config': self.config.copy()
        }
        self.training_history.append(history_entry)
        
        # Save training history
        history_path = Path("training_history.json")
        if history_path.exists():
            with open(history_path, 'r') as f:
                history = json.load(f)
        else:
            history = []
        
        history.append(history_entry)
        
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"  ✅ Model saved to {self.model_path}")
        print(f"  ✅ Training history updated")
    
    def run_retraining(
        self, 
        force_deploy: bool = False,
        dry_run: bool = False
    ) -> Dict:
        """
        Execute the complete retraining pipeline.
        
        Args:
            force_deploy: Deploy even if validation fails
            dry_run: Run training but don't save/deploy
        
        Returns:
            Dictionary with retraining results
        """
        print("="*70)
        print("AUTOMATED MODEL RETRAINING PIPELINE")
        print("="*70)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Dry run: {dry_run}")
        print()
        
        try:
            # Load data
            df_merged = self.load_and_merge_datasets()
            
            # Preprocess
            X, y, feature_names, encoders, scaler = self.preprocess_data(df_merged)
            
            # Train
            model, metrics, test_data = self.train_new_model(X, y)
            
            # Validate
            print("\n" + "="*70)
            print("MODEL VALIDATION")
            print("="*70)
            passes, issues = self.validate_model_performance(metrics)
            
            if passes:
                print("  ✅ Model passes all validation checks")
            else:
                print("  ⚠️  Model validation issues:")
                for issue in issues:
                    print(f"     - {issue}")
            
            # Decide on deployment
            should_deploy = passes or force_deploy
            
            if dry_run:
                print("\n  ℹ️  DRY RUN: Model will not be saved")
                should_deploy = False
            
            # Create model bundle
            model_bundle = {
                'model': model,
                'feature_names': feature_names,
                'encoders': encoders,
                'scaler': scaler,
                'metrics': metrics,
                'trained_at': datetime.now().isoformat(),
                'config': self.config.copy()
            }
            
            # Deploy if approved
            if should_deploy:
                print("\n" + "="*70)
                print("DEPLOYING NEW MODEL")
                print("="*70)
                
                if self.config['backup_old_model']:
                    self.backup_current_model()
                
                self.save_model(model_bundle, metrics)
                
                print("\n  ✅ DEPLOYMENT SUCCESSFUL")
            else:
                if not passes and not force_deploy:
                    print("\n  ⛔ DEPLOYMENT CANCELLED: Model did not meet performance thresholds")
                    print("     Use force_deploy=True to deploy anyway")
            
            print("\n" + "="*70)
            print("RETRAINING PIPELINE COMPLETED")
            print("="*70)
            print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            return {
                'success': True,
                'deployed': should_deploy,
                'metrics': metrics,
                'validation_passed': passes,
                'validation_issues': issues,
                'model_path': self.model_path if should_deploy else None
            }
            
        except Exception as e:
            print(f"\n❌ ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            
            return {
                'success': False,
                'error': str(e),
                'deployed': False
            }


def create_default_config():
    """Create default configuration file."""
    pipeline = ModelRetrainingPipeline()
    pipeline.save_config()
    print("Default configuration created at retrain_config.json")


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    dry_run = '--dry-run' in sys.argv
    force = '--force' in sys.argv
    create_config = '--create-config' in sys.argv
    
    if create_config:
        create_default_config()
        sys.exit(0)
    
    # Run retraining pipeline
    pipeline = ModelRetrainingPipeline()
    result = pipeline.run_retraining(force_deploy=force, dry_run=dry_run)
    
    # Exit with appropriate code
    if result['success'] and result['deployed']:
        sys.exit(0)  # Success
    elif result['success'] and not result['deployed']:
        sys.exit(1)  # Ran but didn't deploy
    else:
        sys.exit(2)  # Error
