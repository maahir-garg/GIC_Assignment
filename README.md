# Financial Transaction Analysis Dashboard

A professional-grade interactive dashboard for analyzing financial transaction data, built with Python and Streamlit.

## Overview
This dashboard provides comprehensive analysis and visualization of financial transaction data, designed to meet enterprise-grade requirements for data exploration and analysis.

## Key Features
### Data Analysis
- Interactive date and amount range filters
- Real-time transaction category filtering
- Comprehensive KPI tracking
  - Transaction totals
  - Average amounts
  - Transaction counts
  - Peak activity periods

### Visualizations
- Time series analysis
- Category distribution charts
- Weekday-hour activity heatmaps
- Amount distribution analysis
- Merchant analysis

### Advanced Analytics
- Anomaly detection using IsolationForest
- Configurable detection parameters
- Interactive data filtering
- Chatbot

## Installation

1. Set up Python environment (recommended):
```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Launch the dashboard:
```bash
streamlit run app.py
```

## Data Requirements

### Required CSV Format
The dashboard expects a CSV file with the following structure:
- **Required columns:**
  - Transaction date/timestamp
  - Transaction amount
- **Optional columns:**
  - Category
  - Merchant
  - Description
  - Transaction ID

### Column Name Flexibility
The system automatically maps common column names:
- Date/Time: 'date', 'timestamp', 'transaction_date'
- Amount: 'amount', 'value', 'transaction_amount'
- Categories: 'category', 'type', 'transaction_type'
- Merchant: 'merchant', 'vendor', 'seller'

## Technical Notes
- Minimum of 10 transactions required for anomaly detection
- Supports both positive and negative transaction amounts
- Auto-handles missing values and data type conversion
- Built with Streamlit and Pandas

## Performance Considerations
- Optimized for datasets up to 100,000 rows
- Uses efficient data filtering techniques
- Implements caching for visualization performance

## Support
For technical support or data format questions, please refer to the documentation or raise an issue in the repository.


