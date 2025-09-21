# Financial Transaction Analysis Dashboard

A professional-grade interactive dashboard for analyzing financial transaction data, built with Python and Streamlit.

## Core Features

### 1. Data Analysis & Visualization
- **Time Series Analysis**
  - Transaction patterns and trends
  - Seasonal variations
  - Volume changes visualization

- **Transaction Categories**
  - Category distribution
  - Merchant analysis
  - Payment method breakdown

- **Activity Patterns**
  - Weekday-hour heatmaps
  - Monthly seasonality
  - Peak activity periods

### 2. Machine Learning Features
- **Predictive Analytics**
  - Transaction trend forecasting
  - 30-day amount predictions
  - Historical pattern analysis

- **Advanced Anomaly Detection**
  - Multi-feature ML Detection
    - Transaction timing analysis
    - Amount pattern recognition
    - Seasonal variation detection
    - Combined behavior scoring
    - Anomaly Score Interpretation:
      - Scores range from -1 to 1
      - Negative scores indicate anomalies
      - Lower scores = more unusual
      - Considers multiple patterns simultaneously
  - Statistical Detection (Z-Score)
  - Configurable sensitivity

- **Category Prediction**
  - ML-based category suggestions
  - Description-based classification
  - Training on existing data

### 3. Interactive Features
- **Dynamic Filtering**
  - Date range selection
  - Amount range filters
  - Category/Merchant filters
  
- **Data Export**
  - Filtered data download
  - Custom CSV export
  - Analysis results export

- **Insight Chatbot**
  - Natural language queries
  - Statistical summaries
  - Trend analysis

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

## Technical Details
- Python 3.8+
- Key Dependencies:
  - Streamlit
  - Pandas
  - Scikit-learn
  - Plotly/Altair
  - NumPy

## Usage Guide
1. **Data Loading**
   - Place CSV file in application directory
   - Automatic column mapping
   - Data validation checks

2. **Analysis Flow**
   - Start with overview metrics
   - Use filters to segment data
   - Apply ML analysis as needed
   - Export results

3. **Best Practices**
   - Filter data before ML analysis
   - Review anomalies in context
   - Use chatbot for quick insights
   - Export findings for reporting

## Performance Considerations
- Optimized for datasets up to 100,000 rows
- Uses efficient data filtering techniques
- Implements caching for visualization performance

## Support
For technical support or data format questions, please refer to the documentation or raise an issue in the repository.


