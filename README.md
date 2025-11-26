# fraud-ml

ML модель по выявлению мошеннических переводов в Мобильном интернет-Банкинге для AI-хакатона ForteBank

## Fraud Detection System with Real ML Model

A production-ready fraud detection system trained on real mobile banking transaction data. The system uses a Random Forest model with 33 features including transaction details and customer behavioral patterns to identify potentially fraudulent transactions.

### Key Features

- **Real ML Model**: Trained on 13,140 actual transactions with 1.26% fraud rate
- **High Performance**: 97.3% accuracy, 94.5% ROC-AUC score
- **Rich Feature Set**: 33 features including behavioral patterns, temporal features, and transaction characteristics
- **Single Transaction Analysis**: Real-time fraud detection for individual transactions
- **Bulk CSV Processing**: Batch analysis of multiple transactions
- **Model Metrics Dashboard**: View detailed performance metrics
- **Export Results**: Download analysis results as CSV files

### Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | 97.30% |
| Precision | 20.31% |
| Recall | 39.39% |
| F1-Score | 26.80% |
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

3. Train the model on real data:
```bash
python create_sample_model.py
```

This will:
- Load and merge the transaction and behavioral datasets
- Preprocess and engineer 33 features
- Train a Random Forest classifier
- Evaluate performance and save the model as `model.pkl`

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

### Usage

#### Single Transaction Analysis

Enter transaction details in the web form:
- Customer ID
- Transaction date and time
- Amount
- Direction (incoming/outgoing)
- Additional metadata

The system will predict whether the transaction is fraudulent and provide a risk assessment.

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

Test the prediction pipeline:

```bash
python test_prediction.py
```

This script:
- Loads the trained model
- Tests with sample transactions
- Verifies the preprocessing pipeline
- Shows prediction results

### Project Structure

```
fraud-ml/
├── app.py                              # Streamlit web application
├── create_sample_model.py              # Model training script
├── test_prediction.py                  # Testing script
├── model.pkl                          # Trained model (generated)
├── requirements.txt                   # Python dependencies
├── MODEL_DOCUMENTATION.md             # Detailed model docs
├── README.md                          # This file
├── sample_transactions.csv            # Sample data
├── поведенческие_паттерны_клиентов_3.csv   # Behavioral data
└── транзакции_в_Мобильном_интернет_Банкинге.csv  # Transaction data
```

### Requirements

```
pandas>=2.2.3
scikit-learn>=1.5.2
streamlit>=1.40.0
numpy>=1.24.0
```

### Model Retraining

To retrain the model with new data:

1. Update the CSV files with new transactions
2. Run the training script:
```bash
python create_sample_model.py
```
3. The new `model.pkl` will be saved and automatically used by the application

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

---

**Last Updated**: November 26, 2025  
**Model Version**: 1.0  
**Training Data Period**: January - August 2025
