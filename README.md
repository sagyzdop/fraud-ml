# fraud-ml

ML модель по выявлению мошеннических переводов в Мобильном интернет-Банкинге для AI-хакатона ForteBank

## Fraud Detection System with Real ML Model

A production-ready fraud detection system trained on real mobile banking transaction data. The system uses a Random Forest model with 33 features including transaction details and customer behavioral patterns to identify potentially fraudulent transactions.

### Key Features

- **Real ML Model**: Trained on 13,140 actual transactions with 1.26% fraud rate
- **High Performance**: 97.3% accuracy, 94.5% ROC-AUC score
- **Rich Feature Set**: 33 features including behavioral patterns, temporal features, and transaction characteristics
- **F_β Metrics**: Calculates F_0.5, F_1.0, F_2.0 scores for flexible precision/recall tuning
- **SHAP Explainability**: Complete interpretability layer with SHAP values and feature importance
- **Configurable Threshold**: Adjustable fraud probability threshold via UI, config file, or API
- **Automated Retraining**: MLOps pipeline for automatic model updates with validation
- **Single Transaction Analysis**: Real-time fraud detection with explanations
- **Bulk CSV Processing**: Batch analysis of multiple transactions
- **Export Results**: Download analysis results as CSV files

### Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | 97.30% |
| Precision | 20.31% |
| Recall | 39.39% |
| F1-Score (F_β, β=1.0) | 26.80% |
| F_0.5-Score (precision-focused) | 22.49% |
| F_2.0-Score (recall-focused) | 33.16% |
| ROC-AUC | 94.48% |

See [MODEL_DOCUMENTATION.md](MODEL_DOCUMENTATION.md) for detailed model information.

### Dataset Information

The model is trained on two merged datasets:

1. **Transaction Data**: 13,113 mobile banking transactions
   - 350 unique customers
   - 165 fraudulent transactions (1.26%)
   - Fields: customer ID, date, time, amount, direction, target

2. **Behavioral Patterns**: 8,587 customer behavior records
   - 19 behavioral features
   - Login patterns, device changes, OS changes
   - Time-based activity metrics

### Installation

1. Clone the repository:
```bash
git clone https://github.com/sagyzdop/fraud-ml.git
cd fraud-ml
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Optional (for full explainability features):
```bash
pip install shap
```

3. Train the model on real data:
```bash
python create_sample_model.py
```

This will:
- Load and merge the transaction and behavioral datasets
- Preprocess and engineer 33 features
- Train a Random Forest classifier with class balancing
- Calculate F_β metrics (F_0.5, F_1.0, F_2.0)
- Evaluate performance with ROC-AUC, precision, recall
- Save the model as `model.pkl` with all metadata

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

### Usage

#### Configurable Fraud Threshold

Adjust the fraud detection sensitivity using the sidebar slider:
- **Lower threshold (e.g., 0.3)**: More sensitive, catches more fraud but increases false positives
- **Default threshold (0.5)**: Balanced precision/recall trade-off
- **Higher threshold (e.g., 0.7)**: More conservative, fewer false positives but may miss some fraud

The threshold can also be configured via `config.json` or API parameter.

#### Single Transaction Analysis

Enter transaction details in the web form:
- Customer ID
- Transaction date and time
- Amount
- Direction (incoming/outgoing)
- Additional metadata

The system will:
- Predict fraud probability (0-100%)
- Apply the configured threshold
- Show risk level (Low, Moderate, High, Critical)
- **Display SHAP explanations** (if enabled): Why was this transaction flagged?
- Provide actionable recommendations

#### Bulk CSV Processing

Upload a CSV file with the following required columns:

```csv
cst_dim_id,transdate,transdatetime,amount,docno,direction,target
2937833270,2025-01-05,2025-01-05 16:32:02,31000.0,5343,outgoing,0
2096229005,2025-03-04,2025-03-04 17:41:57,4000.0,8442,incoming,0
```

Optional columns for enhanced predictions:
- All 19 behavioral pattern features (see MODEL_DOCUMENTATION.md)

The system will:
- Process all transactions
- Generate predictions
- Show summary statistics
- Allow download of results with fraud flags

### Model Features

The model uses 33 engineered features:

**Transaction Features (2)**
- Amount and log-transformed amount

**Temporal Features (10)**
- Day, month, year, day of week, quarter
- Hour, minute
- Weekend, night, and business hours flags

**Behavioral Features (18)**
- Login frequency and patterns
- Device and OS change patterns
- Login interval statistics
- Behavioral data availability flags

**Categorical Encodings (3)**
- Transaction direction
- Phone model frequency
- OS frequency

### Testing

**Test the prediction pipeline:**
```bash
python test_prediction.py
```

**Test explainability features:**
```bash
python model_explainer.py
```

**Test all hackathon compliance features:**
```bash
python test_hackathon_features.py
```

**Test automated retraining pipeline:**
```bash
python retrain_pipeline.py --dry-run
```

### Project Structure

```
fraud-ml/
├── app.py                              # Streamlit web application with threshold control
├── create_sample_model.py              # Model training script with F_β metrics
├── model_explainer.py                  # SHAP explainability module (NEW)
├── retrain_pipeline.py                 # Automated retraining pipeline (NEW)
├── test_prediction.py                  # Prediction testing script
├── test_hackathon_features.py          # Hackathon compliance tests (NEW)
├── model.pkl                          # Trained model (generated)
├── config.json                        # Configuration file (NEW)
├── requirements.txt                   # Python dependencies
├── MODEL_DOCUMENTATION.md             # Detailed model docs
├── HACKATHON_COMPLIANCE.md            # Hackathon criteria compliance (NEW)
├── IMPLEMENTATION_SUMMARY.md          # Implementation overview (NEW)
├── README.md                          # This file
├── sample_transactions.csv            # Sample data
├── model_backups/                     # Model backup directory (generated)
├── поведенческие_паттерны_клиентов_3.csv   # Behavioral data
└── транзакции_в_Мобильном_интернет_Банкинге.csv  # Transaction data
```

### Requirements

```
pandas>=2.2.3
scikit-learn>=1.5.2
streamlit>=1.40.0
numpy>=1.24.0
shap>=0.45.0          # For explainability (optional)
matplotlib>=3.8.0     # For visualizations (optional)
```

### Model Retraining

#### Manual Retraining

To retrain the model with new data:

1. Update the CSV files with new transactions
2. Run the training script:
```bash
python create_sample_model.py
```
3. The new `model.pkl` will be saved and automatically used by the application

#### Automated Retraining Pipeline

Use the automated retraining pipeline for production deployments:

```bash
# Standard retraining with validation
python retrain_pipeline.py

# Dry run (test without deploying)
python retrain_pipeline.py --dry-run

# Force deployment (skip validation)
python retrain_pipeline.py --force

# Create default configuration
python retrain_pipeline.py --create-config
```

**Features:**
- Automatic data loading and preprocessing
- Performance validation against thresholds
- Automatic backup of old models
- Training history logging
- Safe deployment with rollback capability

**Schedule automated retraining:**
```bash
# Weekly retraining via cron
crontab -e
# Add: 0 2 * * 0 cd /path/to/fraud-ml && python retrain_pipeline.py
```

### Documentation

- [MODEL_DOCUMENTATION.md](MODEL_DOCUMENTATION.md) - Comprehensive model documentation including:
  - Dataset description
  - Feature engineering details
  - Model architecture and hyperparameters
  - Performance metrics and interpretation
  - Usage examples
  - Future improvement suggestions

### License

See [LICENSE](LICENSE) file for details.

### Contributing

This project was created for the ForteBank AI Hackathon. For questions or contributions, please open an issue or submit a pull request.

### Quick Links

- **Run Application**: `streamlit run app.py`
- **Train Model**: `python create_sample_model.py`
- **Test Model**: `python test_prediction.py`
- **Explain Predictions**: `python model_explainer.py`
- **Retrain Pipeline**: `python retrain_pipeline.py --dry-run`
