"""
Model Explainer Module

Provides interpretability and explainability for fraud detection predictions
using SHAP values and feature importance visualization.
"""

import pandas as pd
import numpy as np
import pickle
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class ModelExplainer:
    """
    Explainability layer for fraud detection model using SHAP values
    and feature importance analysis.
    """
    
    def __init__(self, model_bundle: dict):
        """
        Initialize the explainer with a trained model bundle.
        
        Args:
            model_bundle: Dictionary containing model, feature names, encoders, scaler
        """
        self.model = model_bundle['model']
        self.feature_names = model_bundle.get('feature_names', [])
        self.encoders = model_bundle.get('encoders', {})
        self.scaler = model_bundle.get('scaler', None)
        self.shap_explainer = None
        
    def initialize_shap(self, background_data: Optional[pd.DataFrame] = None):
        """
        Initialize SHAP explainer with background data.
        
        Args:
            background_data: Background dataset for SHAP (optional)
        """
        try:
            import shap
            
            if background_data is not None:
                # Use provided background data
                self.shap_explainer = shap.TreeExplainer(
                    self.model, 
                    background_data,
                    feature_perturbation='interventional'
                )
            else:
                # Use model directly without background data
                self.shap_explainer = shap.TreeExplainer(self.model)
                
            return True
        except ImportError:
            print("Warning: SHAP not installed. Install with: pip install shap")
            return False
    
    def explain_prediction(
        self, 
        transaction_data: pd.DataFrame,
        prediction: int,
        prediction_proba: float
    ) -> Dict:
        """
        Generate explanation for a single prediction.
        
        Args:
            transaction_data: Preprocessed transaction features (1 row)
            prediction: Model prediction (0 or 1)
            prediction_proba: Prediction probability
        
        Returns:
            Dictionary containing explanation details
        """
        explanation = {
            'prediction': int(prediction),
            'prediction_proba': float(prediction_proba),
            'prediction_label': 'FRAUD' if prediction == 1 else 'LEGITIMATE',
            'confidence': float(abs(prediction_proba - 0.5) * 2),  # 0 to 1 scale
            'risk_level': self._get_risk_level(prediction_proba),
            'feature_importance': self._get_feature_importance(),
            'top_contributing_features': self._get_top_features(transaction_data),
            'shap_values': None,
            'shap_available': self.shap_explainer is not None
        }
        
        # Add SHAP values if available
        if self.shap_explainer is not None:
            try:
                import shap
                shap_values = self.shap_explainer.shap_values(transaction_data)
                
                # Handle different SHAP output formats
                if isinstance(shap_values, list):
                    # Binary classification - use fraud class (index 1)
                    shap_values = shap_values[1]
                
                # Get SHAP values for this prediction
                if len(shap_values.shape) > 1:
                    shap_values_single = shap_values[0]
                else:
                    shap_values_single = shap_values
                
                # Create SHAP explanation
                shap_explanation = []
                for feature, shap_val, actual_val in zip(
                    self.feature_names, 
                    shap_values_single,
                    transaction_data.values[0]
                ):
                    shap_explanation.append({
                        'feature': feature,
                        'value': float(actual_val),
                        'shap_value': float(shap_val),
                        'impact': 'increases' if shap_val > 0 else 'decreases',
                        'magnitude': abs(float(shap_val))
                    })
                
                # Sort by magnitude
                shap_explanation.sort(key=lambda x: x['magnitude'], reverse=True)
                
                explanation['shap_values'] = shap_explanation[:10]  # Top 10
                explanation['shap_base_value'] = float(self.shap_explainer.expected_value[1] if isinstance(self.shap_explainer.expected_value, (list, np.ndarray)) else self.shap_explainer.expected_value)
                
            except Exception as e:
                print(f"Warning: Could not compute SHAP values: {e}")
                explanation['shap_error'] = str(e)
        
        return explanation
    
    def _get_risk_level(self, probability: float) -> str:
        """Classify risk level based on probability."""
        if probability < 0.3:
            return 'LOW'
        elif probability < 0.5:
            return 'MODERATE'
        elif probability < 0.7:
            return 'HIGH'
        else:
            return 'CRITICAL'
    
    def _get_feature_importance(self) -> List[Dict]:
        """
        Get feature importance from the model.
        
        Returns:
            List of dictionaries with feature names and importance scores
        """
        if not hasattr(self.model, 'feature_importances_'):
            return []
        
        importance_list = []
        for feature, importance in zip(self.feature_names, self.model.feature_importances_):
            importance_list.append({
                'feature': feature,
                'importance': float(importance)
            })
        
        # Sort by importance
        importance_list.sort(key=lambda x: x['importance'], reverse=True)
        return importance_list[:10]  # Top 10
    
    def _get_top_features(self, transaction_data: pd.DataFrame, top_n: int = 5) -> List[Dict]:
        """
        Get top contributing features based on values and importance.
        
        Args:
            transaction_data: Preprocessed transaction features
            top_n: Number of top features to return
        
        Returns:
            List of top contributing features
        """
        if not hasattr(self.model, 'feature_importances_'):
            return []
        
        # Combine feature values with importance
        features_with_values = []
        for feature, importance, value in zip(
            self.feature_names, 
            self.model.feature_importances_,
            transaction_data.values[0]
        ):
            # Calculate contribution score (importance * normalized value)
            contribution = float(importance * abs(value))
            features_with_values.append({
                'feature': feature,
                'value': float(value),
                'importance': float(importance),
                'contribution': contribution
            })
        
        # Sort by contribution
        features_with_values.sort(key=lambda x: x['contribution'], reverse=True)
        return features_with_values[:top_n]
    
    def generate_explanation_text(self, explanation: Dict) -> str:
        """
        Generate human-readable explanation text.
        
        Args:
            explanation: Explanation dictionary from explain_prediction
        
        Returns:
            Human-readable explanation string
        """
        text_parts = []
        
        # Header
        label = explanation['prediction_label']
        proba = explanation['prediction_proba']
        risk = explanation['risk_level']
        confidence = explanation['confidence']
        
        text_parts.append(f"🔍 FRAUD DETECTION ANALYSIS")
        text_parts.append(f"{'='*50}")
        text_parts.append(f"Prediction: {label}")
        text_parts.append(f"Fraud Probability: {proba:.2%}")
        text_parts.append(f"Risk Level: {risk}")
        text_parts.append(f"Model Confidence: {confidence:.2%}")
        text_parts.append("")
        
        # Top contributing features
        if explanation['top_contributing_features']:
            text_parts.append("🎯 KEY FACTORS:")
            for i, feature in enumerate(explanation['top_contributing_features'], 1):
                text_parts.append(
                    f"  {i}. {feature['feature']}: {feature['value']:.4f} "
                    f"(importance: {feature['importance']:.2%})"
                )
            text_parts.append("")
        
        # SHAP explanation
        if explanation['shap_values']:
            text_parts.append("📊 SHAP ANALYSIS (Why this prediction?):")
            for i, shap_item in enumerate(explanation['shap_values'][:5], 1):
                impact_symbol = "📈" if shap_item['impact'] == 'increases' else "📉"
                text_parts.append(
                    f"  {i}. {shap_item['feature']}: {impact_symbol} "
                    f"{shap_item['impact']} fraud likelihood by {shap_item['magnitude']:.4f}"
                )
            text_parts.append("")
        elif not explanation['shap_available']:
            text_parts.append("💡 Install SHAP for detailed explanations: pip install shap")
            text_parts.append("")
        
        # Recommendation
        text_parts.append("💡 RECOMMENDATION:")
        if explanation['prediction'] == 1:
            if risk == 'CRITICAL':
                text_parts.append("  ⛔ BLOCK transaction immediately and investigate")
            elif risk == 'HIGH':
                text_parts.append("  ⚠️  REVIEW transaction manually before processing")
            else:
                text_parts.append("  ⚠️  FLAG for review but allow processing")
        else:
            if proba > 0.3:
                text_parts.append("  ✅ ALLOW but monitor for patterns")
            else:
                text_parts.append("  ✅ ALLOW transaction to proceed")
        
        return "\n".join(text_parts)
    
    def batch_explain(
        self, 
        transactions_data: pd.DataFrame,
        predictions: np.ndarray,
        prediction_probas: np.ndarray
    ) -> List[Dict]:
        """
        Generate explanations for multiple predictions.
        
        Args:
            transactions_data: Preprocessed transaction features
            predictions: Model predictions array
            prediction_probas: Prediction probabilities array
        
        Returns:
            List of explanation dictionaries
        """
        explanations = []
        for i in range(len(transactions_data)):
            transaction = transactions_data.iloc[i:i+1]
            explanation = self.explain_prediction(
                transaction,
                predictions[i],
                prediction_probas[i]
            )
            explanations.append(explanation)
        
        return explanations
    
    def plot_feature_importance(self, save_path: Optional[str] = None):
        """
        Plot feature importance chart.
        
        Args:
            save_path: Path to save the plot (optional)
        """
        try:
            import matplotlib.pyplot as plt
            
            importance_data = self._get_feature_importance()
            if not importance_data:
                print("No feature importance available")
                return
            
            features = [item['feature'] for item in importance_data]
            importances = [item['importance'] for item in importance_data]
            
            plt.figure(figsize=(10, 6))
            plt.barh(features, importances)
            plt.xlabel('Importance')
            plt.title('Top 10 Feature Importance')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Plot saved to {save_path}")
            else:
                plt.show()
                
        except ImportError:
            print("Matplotlib not installed. Install with: pip install matplotlib")
    
    def plot_shap_waterfall(
        self, 
        transaction_data: pd.DataFrame,
        save_path: Optional[str] = None
    ):
        """
        Plot SHAP waterfall chart for a single prediction.
        
        Args:
            transaction_data: Preprocessed transaction features (1 row)
            save_path: Path to save the plot (optional)
        """
        if self.shap_explainer is None:
            print("SHAP explainer not initialized. Call initialize_shap() first.")
            return
        
        try:
            import shap
            import matplotlib.pyplot as plt
            
            shap_values = self.shap_explainer.shap_values(transaction_data)
            
            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Fraud class
            
            # Create explanation object
            if hasattr(shap, 'Explanation'):
                expected_value = self.shap_explainer.expected_value
                if isinstance(expected_value, (list, np.ndarray)):
                    expected_value = expected_value[1]
                
                explanation = shap.Explanation(
                    values=shap_values[0],
                    base_values=expected_value,
                    data=transaction_data.values[0],
                    feature_names=self.feature_names
                )
                
                shap.plots.waterfall(explanation, show=not save_path)
            else:
                # Fallback for older SHAP versions
                shap.waterfall_plot(
                    shap.Explanation(
                        values=shap_values[0],
                        base_values=self.shap_explainer.expected_value[1],
                        data=transaction_data.values[0],
                        feature_names=self.feature_names
                    ),
                    show=not save_path
                )
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"SHAP waterfall plot saved to {save_path}")
                
        except Exception as e:
            print(f"Error plotting SHAP waterfall: {e}")


def load_explainer(model_path: str = "model.pkl") -> ModelExplainer:
    """
    Load a model and create an explainer.
    
    Args:
        model_path: Path to the model pickle file
    
    Returns:
        ModelExplainer instance
    """
    with open(model_path, 'rb') as f:
        model_bundle = pickle.load(f)
    
    # Ensure model bundle has correct structure
    if not isinstance(model_bundle, dict):
        model_bundle = {
            'model': model_bundle,
            'feature_names': None,
            'encoders': None,
            'scaler': None
        }
    
    return ModelExplainer(model_bundle)


if __name__ == "__main__":
    # Demo usage
    import sys
    from app import load_model, preprocess_for_model
    from datetime import datetime
    
    print("="*60)
    print("MODEL EXPLAINER DEMO")
    print("="*60)
    
    # Load model
    print("\n1. Loading model...")
    model_bundle = load_model('model.pkl')
    explainer = ModelExplainer(model_bundle)
    
    # Initialize SHAP (optional)
    print("2. Initializing SHAP explainer...")
    shap_initialized = explainer.initialize_shap()
    if shap_initialized:
        print("   ✅ SHAP initialized successfully")
    else:
        print("   ⚠️  SHAP not available (install with: pip install shap)")
    
    # Create test transaction
    print("\n3. Creating test transaction...")
    test_transaction = pd.DataFrame({
        'cst_dim_id': [453024373],
        'transdate': [datetime(2025, 3, 5)],
        'transdatetime': [datetime(2025, 3, 5, 3, 30)],  # Night transaction
        'amount': [150000.0],  # Large amount
        'docno': [1234],
        'direction': ['outgoing'],
        'target': [0]
    })
    
    # Preprocess
    processed = preprocess_for_model(test_transaction, model_bundle)
    
    # Make prediction
    print("4. Making prediction...")
    prediction = model_bundle['model'].predict(processed)[0]
    prediction_proba = model_bundle['model'].predict_proba(processed)[0, 1]
    
    # Generate explanation
    print("\n5. Generating explanation...")
    explanation = explainer.explain_prediction(processed, prediction, prediction_proba)
    
    # Print explanation
    print("\n" + explainer.generate_explanation_text(explanation))
    
    # Plot feature importance
    print("\n6. Feature importance available:")
    for item in explanation['feature_importance'][:5]:
        print(f"   - {item['feature']}: {item['importance']:.4f}")
    
    print("\n" + "="*60)
    print("DEMO COMPLETED")
    print("="*60)
