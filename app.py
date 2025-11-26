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


def load_model(model_path: str):
    """Load a pickle model from the specified path."""
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


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


def preprocess_for_model(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess DataFrame for model prediction.
    
    This function prepares the data for the model by:
    - Converting date/datetime columns to numeric features
    - Encoding categorical variables
    - Selecting only the features the model expects
    """
    processed = df.copy()
    
    # Convert transdate to numeric features if present
    if "transdate" in processed.columns:
        processed["transdate"] = pd.to_datetime(processed["transdate"], errors="coerce")
        processed["trans_day"] = processed["transdate"].dt.day
        processed["trans_month"] = processed["transdate"].dt.month
        processed["trans_year"] = processed["transdate"].dt.year
        processed["trans_dayofweek"] = processed["transdate"].dt.dayofweek
        processed = processed.drop(columns=["transdate"])
    
    # Convert transdatetime to numeric features if present
    if "transdatetime" in processed.columns:
        processed["transdatetime"] = pd.to_datetime(processed["transdatetime"], errors="coerce")
        processed["trans_hour"] = processed["transdatetime"].dt.hour
        processed["trans_minute"] = processed["transdatetime"].dt.minute
        processed = processed.drop(columns=["transdatetime"])
    
    # Encode direction (assuming it's categorical)
    if "direction" in processed.columns:
        processed["direction"] = processed["direction"].map({"incoming": 0, "outgoing": 1}).fillna(-1).astype(int)
    
    # Convert docno to numeric (hash or drop)
    if "docno" in processed.columns:
        processed = processed.drop(columns=["docno"])
    
    # Ensure numeric types
    for col in processed.columns:
        if processed[col].dtype == "object":
            try:
                processed[col] = pd.to_numeric(processed[col], errors="coerce")
            except Exception:
                processed = processed.drop(columns=[col])
    
    return processed


def predict_fraud(model, df: pd.DataFrame) -> pd.DataFrame:
    """Run fraud detection on the DataFrame using the loaded model."""
    # Preprocess the data for the model
    processed_df = preprocess_for_model(df)
    
    # Make predictions
    predictions = model.predict(processed_df)
    
    # Add predictions to original dataframe
    result_df = df.copy()
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
    
    # Check for default model
    default_model_path = Path("model.pkl")
    model = None
    
    uploaded_model = st.sidebar.file_uploader(
        "Upload ML Model (.pkl)", 
        type=["pkl"],
        help="Upload a pre-trained pickle model file"
    )
    
    if uploaded_model is not None:
        try:
            model = pickle.load(uploaded_model)
            st.sidebar.success("✅ Model loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"❌ Error loading model: {e}")
    elif default_model_path.exists():
        try:
            model = load_model(str(default_model_path))
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
            if model is None:
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
                    
                    # Run prediction
                    result = predict_fraud(model, transaction_df)
                    
                    # Display result
                    st.subheader("Analysis Result")
                    
                    is_fraud = result["fraud_prediction"].iloc[0] == 1
                    
                    if is_fraud:
                        st.error("⚠️ **POTENTIAL FRAUD DETECTED!**")
                        st.markdown("""
                        This transaction has been flagged as potentially fraudulent.
                        Please review the transaction details carefully.
                        """)
                    else:
                        st.success("✅ **Transaction appears legitimate**")
                        st.markdown("""
                        No fraud indicators were detected for this transaction.
                        """)
                    
                    # Show transaction details
                    st.dataframe(result, use_container_width=True)
                    
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
                    if model is None:
                        st.error("❌ Please upload a model file first!")
                    else:
                        try:
                            with st.spinner("Analyzing transactions..."):
                                # Run predictions
                                results = predict_fraud(model, df)
                            
                            # Summary statistics
                            st.subheader("📈 Analysis Summary")
                            
                            total = len(results)
                            fraudulent = results["fraud_prediction"].sum()
                            legitimate = total - fraudulent
                            
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Total Transactions", total)
                            col2.metric("Potentially Fraudulent", int(fraudulent), delta=None)
                            col3.metric("Legitimate", int(legitimate))
                            
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
