"""
Fraud Detection Streamlit Application

A web application for detecting fraudulent bank transactions using a pre-trained ML model.
Supports both single transaction input and bulk CSV file processing.
"""

import streamlit as st
import pandas as pd
import pickle
from datetime import datetime, date
from pathlib import Path
import numpy as np
from sklearn.metrics import fbeta_score
try:
    from model_explainer import ModelExplainer
    EXPLAINER_AVAILABLE = True
except ImportError:
    EXPLAINER_AVAILABLE = False


def load_model(model_path: str):
    """Load a pickle model from the specified path."""
    with open(model_path, "rb") as f:
        model_bundle = pickle.load(f)
    
    # Check if it's the new model bundle or old simple model
    if isinstance(model_bundle, dict) and 'model' in model_bundle:
        return model_bundle
    else:
        # Old model format - wrap it for compatibility
        return {'model': model_bundle, 'feature_names': None, 'encoders': None, 'scaler': None}


def preprocess_single_transaction(
    cst_dim_id: int,
    transdate: date,
    transdatetime: datetime,
    amount: float,
    docno: str,
    direction: str,
    target: int,
) -> pd.DataFrame:
    """Convert single transaction input to DataFrame for model prediction."""
    data = {
        "cst_dim_id": [cst_dim_id],
        "transdate": [transdate],
        "transdatetime": [transdatetime],
        "amount": [amount],
        "docno": [docno],
        "direction": [direction],
        "target": [target],
    }
    return pd.DataFrame(data)


def preprocess_for_model(df: pd.DataFrame, model_bundle: dict) -> pd.DataFrame:
    """Preprocess DataFrame for model prediction.
    
    This function prepares the data for the model by:
    - Converting date/datetime columns to numeric features
    - Encoding categorical variables
    - Creating additional features
    - Selecting only the features the model expects
    
    Args:
        df: Input DataFrame with transaction data
        model_bundle: Dictionary containing model and preprocessing components
    
    Returns:
        Processed DataFrame ready for prediction
    """
    import numpy as np
    from sklearn.preprocessing import LabelEncoder
    
    processed = df.copy()
    
    # Get model components
    feature_names = model_bundle.get('feature_names')
    encoders = model_bundle.get('encoders', {})
    scaler = model_bundle.get('scaler')
    
    # If no feature names provided, use old preprocessing
    if feature_names is None:
        # Old preprocessing logic for backwards compatibility
        if "transdate" in processed.columns:
            processed["transdate"] = pd.to_datetime(processed["transdate"], errors="coerce")
            processed["trans_day"] = processed["transdate"].dt.day
            processed["trans_month"] = processed["transdate"].dt.month
            processed["trans_year"] = processed["transdate"].dt.year
            processed["trans_dayofweek"] = processed["transdate"].dt.dayofweek
            processed = processed.drop(columns=["transdate"])
        
        if "transdatetime" in processed.columns:
            processed["transdatetime"] = pd.to_datetime(processed["transdatetime"], errors="coerce")
            processed["trans_hour"] = processed["transdatetime"].dt.hour
            processed["trans_minute"] = processed["transdatetime"].dt.minute
            processed = processed.drop(columns=["transdatetime"])
        
        if "direction" in processed.columns:
            processed["direction"] = processed["direction"].map({"incoming": 0, "outgoing": 1}).fillna(-1).astype(int)
        
        if "docno" in processed.columns:
            processed = processed.drop(columns=["docno"])
        
        for col in processed.columns:
            if processed[col].dtype == "object":
                try:
                    processed[col] = pd.to_numeric(processed[col], errors="coerce")
                except Exception:
                    processed = processed.drop(columns=[col])
        
        return processed
    
    # New preprocessing logic matching training
    
    # Convert transdate to datetime features
    if "transdate" in processed.columns:
        processed["transdate"] = pd.to_datetime(processed["transdate"], errors="coerce")
        processed["trans_day"] = processed["transdate"].dt.day
        processed["trans_month"] = processed["transdate"].dt.month
        processed["trans_year"] = processed["transdate"].dt.year
        processed["trans_dayofweek"] = processed["transdate"].dt.dayofweek
        processed["trans_quarter"] = processed["transdate"].dt.quarter
    
    # Convert transdatetime to datetime features
    if "transdatetime" in processed.columns:
        processed["transdatetime"] = pd.to_datetime(processed["transdatetime"], errors="coerce")
        processed["trans_hour"] = processed["transdatetime"].dt.hour
        processed["trans_minute"] = processed["transdatetime"].dt.minute
    
    # Create time-based features
    if "trans_dayofweek" in processed.columns:
        processed['is_weekend'] = processed['trans_dayofweek'].isin([5, 6]).astype(int)
    if "trans_hour" in processed.columns:
        processed['is_night'] = ((processed['trans_hour'] >= 22) | (processed['trans_hour'] <= 6)).astype(int)
        processed['is_business_hours'] = ((processed['trans_hour'] >= 9) & (processed['trans_hour'] <= 17)).astype(int)
    
    # Encode direction
    if "direction" in processed.columns:
        if 'direction' in encoders:
            le_direction = encoders['direction']
            # Handle unknown categories
            processed['direction_encoded'] = processed['direction'].apply(
                lambda x: le_direction.transform([x])[0] if x in le_direction.classes_ else -1
            )
        else:
            processed["direction_encoded"] = processed["direction"].map({"incoming": 0, "outgoing": 1}).fillna(-1).astype(int)
    
    # Handle phone model and OS with frequency encoding
    if 'last_phone_model_categorical' in processed.columns:
        # For prediction, we won't have the exact frequency mapping, so use a default
        processed['phone_model_freq'] = 0.0
    
    if 'last_os_categorical' in processed.columns:
        processed['os_freq'] = 0.0
    
    # Create amount log feature
    if 'amount' in processed.columns:
        processed['amount_log'] = np.log1p(processed['amount'])
    
    # Handle behavioral features - fill missing with 0
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
    
    # Create behavioral flags
    if 'monthly_os_changes' in processed.columns and processed['monthly_os_changes'].notna().any():
        processed['has_behavioral_data'] = processed['monthly_os_changes'].notna().astype(int)
        processed['os_changes_high'] = (processed['monthly_os_changes'] > 1).astype(int)
        processed['phone_changes_high'] = (processed['monthly_phone_model_changes'] > 1).astype(int)
    else:
        processed['has_behavioral_data'] = 0
        processed['os_changes_high'] = 0
        processed['phone_changes_high'] = 0
    
    # Ensure all expected features exist
    for feature in feature_names:
        if feature not in processed.columns:
            processed[feature] = 0
    
    # Fill any remaining NaN values
    for feature in feature_names:
        if feature in processed.columns:
            processed[feature] = processed[feature].fillna(0)
    
    # Select only the features used in training
    processed_features = processed[feature_names].copy()
    
    # Scale features if scaler is available
    if scaler is not None:
        processed_features = scaler.transform(processed_features)
        processed_features = pd.DataFrame(processed_features, columns=feature_names, index=processed.index)
    
    return processed_features


def predict_fraud(model_bundle, df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    Run fraud detection on the DataFrame using the loaded model.
    
    Args:
        model_bundle: Dictionary containing model and preprocessing components
        df: Input DataFrame with transaction data
        threshold: Fraud probability threshold (default 0.5)
    
    Returns:
        DataFrame with predictions and probabilities added
    """
    # Extract model from bundle
    if isinstance(model_bundle, dict) and 'model' in model_bundle:
        model = model_bundle['model']
    else:
        model = model_bundle
        model_bundle = {'model': model, 'feature_names': None, 'encoders': None, 'scaler': None}
    
    # Preprocess the data for the model
    processed_df = preprocess_for_model(df, model_bundle)
    
    # Make predictions with probabilities
    fraud_probabilities = model.predict_proba(processed_df)[:, 1]
    predictions = (fraud_probabilities >= threshold).astype(int)
    
    # Add predictions to original dataframe
    result_df = df.copy()
    result_df["fraud_probability"] = fraud_probabilities
    result_df["fraud_prediction"] = predictions
    result_df["is_fraudulent"] = result_df["fraud_prediction"].apply(
        lambda x: "Yes" if x == 1 else "No"
    )
    
    return result_df


def main():
    st.set_page_config(
        page_title="Fraud Detection System",
        page_icon="🔍",
        layout="wide",
    )
    
    st.title("🔍 Bank Transaction Fraud Detection")
    st.markdown("""
    This application uses a machine learning model to detect potentially fraudulent bank transactions.
    You can analyze individual transactions or upload a CSV file for bulk processing.
    """)
    
    # Model loading section
    st.sidebar.header("⚙️ Model Configuration")
    
    # Fraud threshold configuration
    fraud_threshold = st.sidebar.slider(
        "Fraud Probability Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Transactions with fraud probability above this threshold will be flagged as fraudulent"
    )
    
    st.sidebar.markdown(f"**Current Threshold:** {fraud_threshold:.2f}")
    st.sidebar.markdown("""
    - **Lower threshold** (e.g., 0.3): More sensitive, catches more fraud but more false positives
    - **Higher threshold** (e.g., 0.7): More conservative, fewer false positives but may miss some fraud
    """)
    
    # Explainability toggle
    enable_explanations = st.sidebar.checkbox(
        "Enable Explainability",
        value=EXPLAINER_AVAILABLE,
        help="Show SHAP explanations for predictions (requires shap package)",
        disabled=not EXPLAINER_AVAILABLE
    )
    
    if not EXPLAINER_AVAILABLE:
        st.sidebar.info("💡 Install SHAP for explainability: `pip install shap`")
    
    st.sidebar.markdown("---")
    
    # Check for default model
    default_model_path = Path("model.pkl")
    model_bundle = None
    
    uploaded_model = st.sidebar.file_uploader(
        "Upload ML Model (.pkl)", 
        type=["pkl"],
        help="Upload a pre-trained pickle model file"
    )
    
    if uploaded_model is not None:
        try:
            model_bundle = pickle.load(uploaded_model)
            # Wrap old model format for compatibility
            if not isinstance(model_bundle, dict):
                model_bundle = {'model': model_bundle, 'feature_names': None, 'encoders': None, 'scaler': None}
            st.sidebar.success("✅ Model loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"❌ Error loading model: {e}")
    elif default_model_path.exists():
        try:
            model_bundle = load_model(str(default_model_path))
            st.sidebar.info("ℹ️ Using default model (model.pkl)")
        except Exception as e:
            st.sidebar.warning(f"⚠️ Could not load default model: {e}")
    else:
        st.sidebar.warning("⚠️ Please upload a model file to make predictions")
    
    # Main content tabs
    tab1, tab2 = st.tabs(["📝 Single Transaction", "📁 Bulk CSV Upload"])
    
    # Tab 1: Single Transaction Input
    with tab1:
        st.header("Single Transaction Analysis")
        st.markdown("Enter the transaction details below to check for fraud.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            cst_dim_id = st.number_input(
                "Customer Dimension ID (cst_dim_id)",
                min_value=0,
                value=0,
                step=1,
                help="Unique identifier for the customer"
            )
            
            transdate = st.date_input(
                "Transaction Date (transdate)",
                value=date.today(),
                help="Date of the transaction"
            )
            
            transdatetime = st.time_input(
                "Transaction Time",
                value=datetime.now().time(),
                help="Time of the transaction"
            )
            
            amount = st.number_input(
                "Amount",
                min_value=0.0,
                value=0.0,
                step=0.01,
                format="%.2f",
                help="Transaction amount"
            )
        
        with col2:
            docno = st.text_input(
                "Document Number (docno)",
                value="",
                help="Transaction document number"
            )
            
            direction = st.selectbox(
                "Direction",
                options=["incoming", "outgoing"],
                help="Direction of the transaction"
            )
            
            target = st.number_input(
                "Target",
                min_value=0,
                max_value=1,
                value=0,
                step=1,
                help="Target variable (0 or 1)"
            )
        
        # Combine date and time for transdatetime
        full_transdatetime = datetime.combine(transdate, transdatetime)
        
        if st.button("🔍 Analyze Transaction", type="primary"):
            if model_bundle is None:
                st.error("❌ Please upload a model file first!")
            else:
                try:
                    # Create DataFrame from input
                    transaction_df = preprocess_single_transaction(
                        cst_dim_id=cst_dim_id,
                        transdate=transdate,
                        transdatetime=full_transdatetime,
                        amount=amount,
                        docno=docno,
                        direction=direction,
                        target=target,
                    )
                    
                    # Run prediction with custom threshold
                    result = predict_fraud(model_bundle, transaction_df, threshold=fraud_threshold)
                    
                    # Display result
                    st.subheader("Analysis Result")
                    
                    is_fraud = result["fraud_prediction"].iloc[0] == 1
                    fraud_prob = result["fraud_probability"].iloc[0]
                    
                    # Show probability gauge
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Fraud Probability", f"{fraud_prob:.2%}")
                    col_b.metric("Threshold", f"{fraud_threshold:.2%}")
                    col_c.metric("Decision", "FRAUD" if is_fraud else "LEGITIMATE")
                    
                    if is_fraud:
                        st.error("⚠️ **POTENTIAL FRAUD DETECTED!**")
                        st.markdown("""
                        This transaction has been flagged as potentially fraudulent.
                        Please review the transaction details carefully.
                        """)
                    else:
                        if fraud_prob > 0.3:
                            st.warning("⚡ **Transaction appears legitimate but monitor closely**")
                        else:
                            st.success("✅ **Transaction appears legitimate**")
                        st.markdown("""
                        No fraud indicators were detected for this transaction.
                        """)
                    
                    # Show transaction details
                    st.dataframe(result, use_container_width=True)
                    
                    # Show explainability if enabled
                    if enable_explanations and EXPLAINER_AVAILABLE:
                        try:
                            st.subheader("🔍 Prediction Explanation")
                            
                            # Create explainer
                            explainer = ModelExplainer(model_bundle)
                            explainer.initialize_shap()
                            
                            # Get preprocessed data
                            processed = preprocess_for_model(transaction_df, model_bundle)
                            
                            # Generate explanation
                            explanation = explainer.explain_prediction(
                                processed,
                                result["fraud_prediction"].iloc[0],
                                fraud_prob
                            )
                            
                            # Display explanation text
                            st.text(explainer.generate_explanation_text(explanation))
                            
                        except Exception as e:
                            st.warning(f"Could not generate explanation: {e}")
                    
                except Exception as e:
                    st.error(f"❌ Error analyzing transaction: {e}")
    
    # Tab 2: Bulk CSV Upload
    with tab2:
        st.header("Bulk Transaction Analysis")
        st.markdown("""
        Upload a CSV file containing multiple transactions to analyze them all at once.
        
        **Required columns:** `cst_dim_id`, `transdate`, `transdatetime`, `amount`, `docno`, `direction`, `target`
        """)
        
        uploaded_file = st.file_uploader(
            "Upload CSV File",
            type=["csv"],
            help="Upload a CSV file with transaction data"
        )
        
        if uploaded_file is not None:
            try:
                # Read CSV
                df = pd.read_csv(uploaded_file)
                
                st.subheader("📊 Uploaded Data Preview")
                st.dataframe(df.head(10), use_container_width=True)
                st.info(f"Total transactions: {len(df)}")
                
                # Validate columns
                required_columns = ["cst_dim_id", "transdate", "transdatetime", "amount", "docno", "direction", "target"]
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    st.warning(f"⚠️ Missing columns: {', '.join(missing_columns)}")
                    st.markdown("The model will try to work with available columns.")
                
                if st.button("🔍 Analyze All Transactions", type="primary"):
                    if model_bundle is None:
                        st.error("❌ Please upload a model file first!")
                    else:
                        try:
                            with st.spinner("Analyzing transactions..."):
                                # Run predictions with custom threshold
                                results = predict_fraud(model_bundle, df, threshold=fraud_threshold)
                            
                            # Summary statistics
                            st.subheader("📈 Analysis Summary")
                            
                            total = len(results)
                            fraudulent = results["fraud_prediction"].sum()
                            legitimate = total - fraudulent
                            avg_fraud_prob = results["fraud_probability"].mean()
                            
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Total Transactions", total)
                            col2.metric("Potentially Fraudulent", int(fraudulent), delta=None)
                            col3.metric("Legitimate", int(legitimate))
                            col4.metric("Avg Fraud Probability", f"{avg_fraud_prob:.2%}")
                            
                            # Show fraudulent transactions
                            st.subheader("⚠️ Potentially Fraudulent Transactions")
                            fraud_df = results[results["fraud_prediction"] == 1]
                            
                            if len(fraud_df) > 0:
                                st.error(f"Found {len(fraud_df)} potentially fraudulent transaction(s)")
                                st.dataframe(fraud_df, use_container_width=True)
                                
                                # Download button for fraudulent transactions
                                csv = fraud_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Fraudulent Transactions CSV",
                                    data=csv,
                                    file_name="fraudulent_transactions.csv",
                                    mime="text/csv",
                                )
                            else:
                                st.success("No fraudulent transactions detected!")
                            
                            # Show all results
                            st.subheader("📋 All Results")
                            st.dataframe(results, use_container_width=True)
                            
                            # Download button for all results
                            all_csv = results.to_csv(index=False)
                            st.download_button(
                                label="📥 Download All Results CSV",
                                data=all_csv,
                                file_name="all_transaction_results.csv",
                                mime="text/csv",
                            )
                            
                        except Exception as e:
                            st.error(f"❌ Error analyzing transactions: {e}")
                            
            except Exception as e:
                st.error(f"❌ Error reading CSV file: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Fraud Detection System | Powered by Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
