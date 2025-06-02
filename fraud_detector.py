#!/usr/bin/env python3
"""
Fraud Detection System
Advanced machine learning system for real-time fraud detection using multiple algorithms,
anomaly detection, and risk scoring with comprehensive monitoring dashboard.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import joblib
import sqlite3
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

class FraudDetectionSystem:
    def __init__(self):
        """Initialize the fraud detection system."""
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.risk_thresholds = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8
        }
        
    def generate_transaction_data(self, n_transactions=50000):
        """Generate realistic transaction data with fraud patterns."""
        np.random.seed(42)
        
        transactions = []
        
        for i in range(n_transactions):
            # Normal transaction patterns
            if np.random.random() > 0.02:  # 98% normal transactions
                amount = np.random.lognormal(4, 1.5)  # Log-normal distribution
                amount = min(amount, 10000)  # Cap at $10,000
                
                transaction = {
                    'transaction_id': f'TXN_{i:06d}',
                    'amount': round(amount, 2),
                    'merchant_category': np.random.choice([
                        'grocery', 'gas_station', 'restaurant', 'retail', 'online',
                        'pharmacy', 'entertainment', 'travel', 'utilities'
                    ]),
                    'transaction_time': datetime.now() - timedelta(
                        days=np.random.randint(0, 365),
                        hours=np.random.randint(0, 24),
                        minutes=np.random.randint(0, 60)
                    ),
                    'location': np.random.choice(['domestic', 'international'], p=[0.95, 0.05]),
                    'payment_method': np.random.choice(['credit', 'debit', 'mobile'], p=[0.6, 0.3, 0.1]),
                    'is_weekend': np.random.choice([0, 1], p=[0.71, 0.29]),
                    'customer_age': np.random.randint(18, 80),
                    'account_age_days': np.random.randint(30, 3650),
                    'previous_failed_attempts': np.random.poisson(0.1),
                    'is_fraud': 0
                }
            else:  # 2% fraudulent transactions
                # Fraudulent patterns: higher amounts, unusual times, international
                amount = np.random.lognormal(6, 2)  # Higher amounts
                amount = min(amount, 50000)
                
                transaction = {
                    'transaction_id': f'TXN_{i:06d}',
                    'amount': round(amount, 2),
                    'merchant_category': np.random.choice([
                        'online', 'entertainment', 'travel', 'retail'
                    ]),  # More likely categories for fraud
                    'transaction_time': datetime.now() - timedelta(
                        days=np.random.randint(0, 365),
                        hours=np.random.choice([2, 3, 4, 22, 23, 0, 1]),  # Unusual hours
                        minutes=np.random.randint(0, 60)
                    ),
                    'location': np.random.choice(['domestic', 'international'], p=[0.3, 0.7]),
                    'payment_method': np.random.choice(['credit', 'debit', 'mobile'], p=[0.8, 0.15, 0.05]),
                    'is_weekend': np.random.choice([0, 1], p=[0.4, 0.6]),
                    'customer_age': np.random.randint(18, 80),
                    'account_age_days': np.random.randint(1, 365),  # Newer accounts
                    'previous_failed_attempts': np.random.poisson(2),  # More failed attempts
                    'is_fraud': 1
                }
            
            # Add derived features
            transaction['hour'] = transaction['transaction_time'].hour
            transaction['day_of_week'] = transaction['transaction_time'].weekday()
            transaction['amount_log'] = np.log1p(transaction['amount'])
            
            transactions.append(transaction)
        
        return pd.DataFrame(transactions)
    
    def preprocess_data(self, df):
        """Preprocess transaction data for machine learning."""
        # Create copy to avoid modifying original
        processed_df = df.copy()
        
        # Encode categorical variables
        categorical_columns = ['merchant_category', 'location', 'payment_method']
        
        for col in categorical_columns:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                processed_df[f'{col}_encoded'] = self.encoders[col].fit_transform(processed_df[col])
            else:
                processed_df[f'{col}_encoded'] = self.encoders[col].transform(processed_df[col])
        
        # Feature engineering
        processed_df['amount_zscore'] = (processed_df['amount'] - processed_df['amount'].mean()) / processed_df['amount'].std()
        processed_df['is_high_amount'] = (processed_df['amount'] > processed_df['amount'].quantile(0.95)).astype(int)
        processed_df['is_unusual_hour'] = processed_df['hour'].apply(lambda x: 1 if x < 6 or x > 22 else 0)
        processed_df['is_new_account'] = (processed_df['account_age_days'] < 90).astype(int)
        
        # Select features for modeling
        feature_columns = [
            'amount', 'amount_log', 'amount_zscore', 'is_high_amount',
            'merchant_category_encoded', 'location_encoded', 'payment_method_encoded',
            'is_weekend', 'hour', 'day_of_week', 'is_unusual_hour',
            'customer_age', 'account_age_days', 'is_new_account',
            'previous_failed_attempts'
        ]
        
        X = processed_df[feature_columns]
        y = processed_df['is_fraud'] if 'is_fraud' in processed_df.columns else None
        
        return X, y, processed_df
    
    def train_isolation_forest(self, X):
        """Train Isolation Forest for anomaly detection."""
        # Scale features
        if 'isolation_forest' not in self.scalers:
            self.scalers['isolation_forest'] = StandardScaler()
            X_scaled = self.scalers['isolation_forest'].fit_transform(X)
        else:
            X_scaled = self.scalers['isolation_forest'].transform(X)
        
        # Train Isolation Forest
        iso_forest = IsolationForest(
            contamination=0.02,  # Expected fraud rate
            random_state=42,
            n_estimators=100
        )
        iso_forest.fit(X_scaled)
        
        self.models['isolation_forest'] = iso_forest
        return iso_forest
    
    def train_random_forest(self, X, y):
        """Train Random Forest classifier."""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale features
        if 'random_forest' not in self.scalers:
            self.scalers['random_forest'] = StandardScaler()
            X_train_scaled = self.scalers['random_forest'].fit_transform(X_train)
        else:
            X_train_scaled = self.scalers['random_forest'].transform(X_train)
        
        X_test_scaled = self.scalers['random_forest'].transform(X_test)
        
        # Train Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        rf_model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = rf_model.predict(X_test_scaled)
        y_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
        
        evaluation = {
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'roc_auc_score': roc_auc_score(y_test, y_pred_proba),
            'feature_importance': dict(zip(X.columns, rf_model.feature_importances_))
        }
        
        self.models['random_forest'] = rf_model
        return rf_model, evaluation
    
    def train_neural_network(self, X, y):
        """Train deep neural network for fraud detection."""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale features
        if 'neural_network' not in self.scalers:
            self.scalers['neural_network'] = StandardScaler()
            X_train_scaled = self.scalers['neural_network'].fit_transform(X_train)
        else:
            X_train_scaled = self.scalers['neural_network'].transform(X_train)
        
        X_test_scaled = self.scalers['neural_network'].transform(X_test)
        
        # Build neural network
        model = Sequential([
            Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        # Train model
        history = model.fit(
            X_train_scaled, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        # Evaluate model
        y_pred_proba = model.predict(X_test_scaled).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        evaluation = {
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'roc_auc_score': roc_auc_score(y_test, y_pred_proba),
            'training_history': history.history
        }
        
        self.models['neural_network'] = model
        return model, evaluation
    
    def calculate_risk_score(self, transaction_data):
        """Calculate comprehensive risk score for transactions."""
        X, _, _ = self.preprocess_data(transaction_data)
        
        risk_scores = []
        
        for _, row in X.iterrows():
            row_data = row.values.reshape(1, -1)
            
            # Get predictions from all models
            scores = {}
            
            # Isolation Forest (anomaly score)
            if 'isolation_forest' in self.models:
                X_scaled = self.scalers['isolation_forest'].transform(row_data)
                anomaly_score = self.models['isolation_forest'].decision_function(X_scaled)[0]
                scores['anomaly'] = max(0, min(1, (anomaly_score + 0.5) / 1.0))  # Normalize to 0-1
            
            # Random Forest probability
            if 'random_forest' in self.models:
                X_scaled = self.scalers['random_forest'].transform(row_data)
                rf_prob = self.models['random_forest'].predict_proba(X_scaled)[0, 1]
                scores['random_forest'] = rf_prob
            
            # Neural Network probability
            if 'neural_network' in self.models:
                X_scaled = self.scalers['neural_network'].transform(row_data)
                nn_prob = self.models['neural_network'].predict(X_scaled)[0, 0]
                scores['neural_network'] = nn_prob
            
            # Ensemble risk score (weighted average)
            if scores:
                weights = {'anomaly': 0.2, 'random_forest': 0.4, 'neural_network': 0.4}
                risk_score = sum(scores.get(model, 0) * weight for model, weight in weights.items())
                risk_score = max(0, min(1, risk_score))  # Ensure 0-1 range
            else:
                risk_score = 0.5  # Default if no models available
            
            risk_scores.append(risk_score)
        
        return risk_scores
    
    def classify_risk_level(self, risk_score):
        """Classify risk level based on score."""
        if risk_score < self.risk_thresholds['low']:
            return 'Low'
        elif risk_score < self.risk_thresholds['medium']:
            return 'Medium'
        elif risk_score < self.risk_thresholds['high']:
            return 'High'
        else:
            return 'Critical'
    
    def real_time_fraud_check(self, transaction):
        """Perform real-time fraud check on a single transaction."""
        # Convert single transaction to DataFrame
        transaction_df = pd.DataFrame([transaction])
        
        # Calculate risk score
        risk_scores = self.calculate_risk_score(transaction_df)
        risk_score = risk_scores[0]
        
        # Classify risk level
        risk_level = self.classify_risk_level(risk_score)
        
        # Generate recommendations
        recommendations = []
        if risk_score > 0.8:
            recommendations.append("Block transaction immediately")
            recommendations.append("Contact customer for verification")
        elif risk_score > 0.6:
            recommendations.append("Require additional authentication")
            recommendations.append("Monitor account closely")
        elif risk_score > 0.3:
            recommendations.append("Flag for review")
            recommendations.append("Increase monitoring frequency")
        else:
            recommendations.append("Approve transaction")
        
        return {
            'transaction_id': transaction.get('transaction_id', 'N/A'),
            'risk_score': round(risk_score, 4),
            'risk_level': risk_level,
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat()
        }
    
    def generate_fraud_report(self, transactions_df):
        """Generate comprehensive fraud analysis report."""
        # Calculate risk scores for all transactions
        risk_scores = self.calculate_risk_score(transactions_df)
        transactions_df['risk_score'] = risk_scores
        transactions_df['risk_level'] = [self.classify_risk_level(score) for score in risk_scores]
        
        # Generate statistics
        total_transactions = len(transactions_df)
        high_risk_count = sum(1 for score in risk_scores if score > self.risk_thresholds['high'])
        medium_risk_count = sum(1 for score in risk_scores if self.risk_thresholds['medium'] < score <= self.risk_thresholds['high'])
        
        # Fraud patterns analysis
        if 'is_fraud' in transactions_df.columns:
            actual_fraud_count = transactions_df['is_fraud'].sum()
            detected_fraud = sum(1 for i, score in enumerate(risk_scores) 
                               if score > self.risk_thresholds['medium'] and transactions_df.iloc[i]['is_fraud'] == 1)
            detection_rate = detected_fraud / actual_fraud_count if actual_fraud_count > 0 else 0
        else:
            actual_fraud_count = "N/A"
            detection_rate = "N/A"
        
        report = {
            'summary': {
                'total_transactions': total_transactions,
                'high_risk_transactions': high_risk_count,
                'medium_risk_transactions': medium_risk_count,
                'high_risk_percentage': round(high_risk_count / total_transactions * 100, 2),
                'actual_fraud_count': actual_fraud_count,
                'detection_rate': round(detection_rate * 100, 2) if detection_rate != "N/A" else "N/A"
            },
            'risk_distribution': transactions_df['risk_level'].value_counts().to_dict(),
            'top_risk_transactions': transactions_df.nlargest(10, 'risk_score')[
                ['transaction_id', 'amount', 'merchant_category', 'risk_score', 'risk_level']
            ].to_dict('records'),
            'patterns': {
                'high_risk_categories': transactions_df[transactions_df['risk_score'] > self.risk_thresholds['high']]['merchant_category'].value_counts().head().to_dict(),
                'high_risk_amounts': {
                    'mean': transactions_df[transactions_df['risk_score'] > self.risk_thresholds['high']]['amount'].mean(),
                    'median': transactions_df[transactions_df['risk_score'] > self.risk_thresholds['high']]['amount'].median()
                }
            }
        }
        
        return report, transactions_df
    
    def save_models(self, filepath_prefix='fraud_detection'):
        """Save trained models and preprocessors."""
        # Save sklearn models and scalers
        for model_name, model in self.models.items():
            if model_name != 'neural_network':
                joblib.dump(model, f'{filepath_prefix}_{model_name}_model.pkl')
        
        for scaler_name, scaler in self.scalers.items():
            joblib.dump(scaler, f'{filepath_prefix}_{scaler_name}_scaler.pkl')
        
        for encoder_name, encoder in self.encoders.items():
            joblib.dump(encoder, f'{filepath_prefix}_{encoder_name}_encoder.pkl')
        
        # Save neural network separately
        if 'neural_network' in self.models:
            self.models['neural_network'].save(f'{filepath_prefix}_neural_network_model.h5')
        
        # Save configuration
        config = {
            'risk_thresholds': self.risk_thresholds,
            'model_names': list(self.models.keys()),
            'scaler_names': list(self.scalers.keys()),
            'encoder_names': list(self.encoders.keys())
        }
        
        with open(f'{filepath_prefix}_config.json', 'w') as f:
            json.dump(config, f, indent=2)
    
    def train_all_models(self, transactions_df):
        """Train all fraud detection models."""
        print("Training fraud detection models...")
        
        # Preprocess data
        X, y, processed_df = self.preprocess_data(transactions_df)
        
        # Train Isolation Forest
        print("Training Isolation Forest...")
        iso_forest = self.train_isolation_forest(X)
        
        # Train Random Forest
        print("Training Random Forest...")
        rf_model, rf_evaluation = self.train_random_forest(X, y)
        
        # Train Neural Network
        print("Training Neural Network...")
        nn_model, nn_evaluation = self.train_neural_network(X, y)
        
        print("All models trained successfully!")
        
        return {
            'isolation_forest': iso_forest,
            'random_forest': {'model': rf_model, 'evaluation': rf_evaluation},
            'neural_network': {'model': nn_model, 'evaluation': nn_evaluation}
        }

def main():
    """Main function to demonstrate fraud detection system."""
    # Initialize system
    fraud_detector = FraudDetectionSystem()
    
    # Generate sample data
    print("Generating sample transaction data...")
    transactions_df = fraud_detector.generate_transaction_data(50000)
    
    # Train models
    models = fraud_detector.train_all_models(transactions_df)
    
    # Generate fraud report
    print("Generating fraud analysis report...")
    report, analyzed_transactions = fraud_detector.generate_fraud_report(transactions_df)
    
    # Print summary
    print("\n" + "="*60)
    print("FRAUD DETECTION SYSTEM REPORT")
    print("="*60)
    print(f"Total Transactions: {report['summary']['total_transactions']:,}")
    print(f"High Risk Transactions: {report['summary']['high_risk_transactions']:,} ({report['summary']['high_risk_percentage']}%)")
    print(f"Medium Risk Transactions: {report['summary']['medium_risk_transactions']:,}")
    if report['summary']['detection_rate'] != "N/A":
        print(f"Fraud Detection Rate: {report['summary']['detection_rate']}%")
    
    # Test real-time fraud check
    print("\nTesting real-time fraud detection...")
    sample_transaction = {
        'transaction_id': 'TEST_001',
        'amount': 5000,
        'merchant_category': 'online',
        'transaction_time': datetime.now(),
        'location': 'international',
        'payment_method': 'credit',
        'is_weekend': 0,
        'customer_age': 25,
        'account_age_days': 30,
        'previous_failed_attempts': 3
    }
    
    fraud_check_result = fraud_detector.real_time_fraud_check(sample_transaction)
    print(f"Sample Transaction Risk Assessment:")
    print(f"Risk Score: {fraud_check_result['risk_score']}")
    print(f"Risk Level: {fraud_check_result['risk_level']}")
    print(f"Recommendations: {', '.join(fraud_check_result['recommendations'])}")
    
    # Save models
    fraud_detector.save_models()
    print("\nModels saved successfully!")

if __name__ == "__main__":
    main()

