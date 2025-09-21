import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

class TransactionAnalytics:
    def __init__(self):
        self.scaler = StandardScaler()
        self.trend_model = RandomForestRegressor(n_estimators=100)
        self.category_model = RandomForestClassifier(n_estimators=100)
        
    def prepare_features(self, df, timestamp_col):
        """Extract features from timestamp and amounts"""
        features = pd.DataFrame()
        features['hour'] = pd.to_datetime(df[timestamp_col]).dt.hour
        features['day_of_week'] = pd.to_datetime(df[timestamp_col]).dt.dayofweek
        features['day_of_month'] = pd.to_datetime(df[timestamp_col]).dt.day
        features['month'] = pd.to_datetime(df[timestamp_col]).dt.month
        return features
        
    def predict_trend(self, df, cols, forecast_days=30):
        """Predict transaction trends"""
        features = self.prepare_features(df, cols['timestamp'])
        amounts = df[cols['amount']].values
        
        X_train, X_test, y_train, y_test = train_test_split(features, amounts, test_size=0.2)
        self.trend_model.fit(X_train, y_train)
        
        # Generate future dates
        last_date = pd.to_datetime(df[cols['timestamp']]).max()
        future_dates = pd.date_range(last_date, periods=forecast_days + 1)[1:]
        future_features = self.prepare_features(
            pd.DataFrame({cols['timestamp']: future_dates}), 
            cols['timestamp']
        )
        
        predictions = self.trend_model.predict(future_features)
        return pd.DataFrame({
            'date': future_dates,
            'predicted_amount': predictions
        })

    def suggest_categories(self, df, cols, description_col='description'):
        """Suggest transaction categories based on description"""
        if description_col not in df.columns or cols.get('category') not in df.columns:
            return None
            
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Prepare text features
        vectorizer = TfidfVectorizer(max_features=100)
        text_features = vectorizer.fit_transform(df[description_col].fillna(''))
        
        # Train classifier
        y = df[cols['category']]
        self.category_model.fit(text_features, y)
        
        return vectorizer, self.category_model

    def detect_complex_anomalies(self, df, cols, contamination=0.1, random_state=42):
        """Enhanced anomaly detection using multiple features"""
        features = self.prepare_features(df, cols['timestamp'])
        features['amount'] = df[cols['amount']]
        features['amount_scaled'] = self.scaler.fit_transform(df[cols['amount']].values.reshape(-1, 1))
        
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100
        )
        
        # Fit and predict
        anomaly_labels = iso_forest.fit_predict(features)
        anomaly_scores = iso_forest.score_samples(features)
        
        return pd.DataFrame({
            'is_anomaly': anomaly_labels == -1,
            'anomaly_score': anomaly_scores,
            'timestamp': df[cols['timestamp']],
            'amount': df[cols['amount']]
        })

    def get_feature_importance(self, df, cols):
        """Analyze feature importance for amount prediction"""
        features = self.prepare_features(df, cols['timestamp'])
        amounts = df[cols['amount']].values
        
        self.trend_model.fit(features, amounts)
        
        return pd.DataFrame({
            'feature': features.columns,
            'importance': self.trend_model.feature_importances_
        }).sort_values('importance', ascending=False)
