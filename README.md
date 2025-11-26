# fraud-ml

ML модель по выявлению мошеннических переводов в Мобильном интернет-Банкинге для AI-хакатона ForteBank

## Fraud Detection Streamlit Application

A web application for detecting fraudulent bank transactions using a pre-trained ML model. Supports both single transaction input and bulk CSV file processing.

### Features

- **Single Transaction Analysis**: Enter individual transaction details to check for fraud
- **Bulk CSV Processing**: Upload a CSV file with multiple transactions for batch analysis
- **Model Upload**: Load your own pre-trained `.pkl` model or use the default model
- **Export Results**: Download analysis results as CSV files

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

3. (Optional) Create a sample model for testing:
```bash
python create_sample_model.py
```

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

### Input Fields

The application expects transactions with the following fields:

| Field | Description | Type |
|-------|-------------|------|
| `cst_dim_id` | Customer Dimension ID | Integer |
| `transdate` | Transaction Date | Date (YYYY-MM-DD) |
| `transdatetime` | Transaction Date and Time | DateTime |
| `amount` | Transaction Amount | Float |
| `docno` | Document Number | String |
| `direction` | Transaction Direction | "incoming" or "outgoing" |
| `target` | Target Variable | Integer (0 or 1) |

### CSV File Format

For bulk processing, upload a CSV file with the following columns:

```csv
cst_dim_id,transdate,transdatetime,amount,docno,direction,target
1001,2024-01-15,2024-01-15 10:30:00,150.50,DOC001,outgoing,0
1002,2024-01-15,2024-01-15 14:45:00,2500.00,DOC002,incoming,0
```

A sample CSV file (`sample_transactions.csv`) is included in the repository.

### Model Requirements

The application expects a scikit-learn compatible model saved in pickle (`.pkl`) format. The model should be able to predict binary classification (0 = legitimate, 1 = fraudulent).

### License

See [LICENSE](LICENSE) file for details.
