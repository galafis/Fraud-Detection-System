"""
Fraud Detection System

Advanced fraud detection using ensemble methods and imbalanced learning techniques.

Author: Gabriel Demetrios Lafis
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, Tuple
from loguru import logger


class FraudDetector:
    """
    Fraud detection system with imbalanced learning techniques.
    """
    
    def __init__(
        self,
        model_type: str = 'xgboost',
        sampling_strategy: str = 'smote',
        scale_pos_weight: float = None
    ):
        """
        Initialize fraud detector.
        
        Args:
            model_type: Type of model ('xgboost', 'lightgbm')
            sampling_strategy: Sampling strategy ('smote', 'adasyn', 'undersample', 'none')
            scale_pos_weight: Weight for positive class (auto-calculated if None)
        """
        self.model_type = model_type
        self.sampling_strategy = sampling_strategy
        self.scale_pos_weight = scale_pos_weight
        self.model = None
        self.scaler = StandardScaler()
        
        logger.info(f"Initialized FraudDetector with {model_type} and {sampling_strategy}")
    
    def _get_sampler(self):
        """Get sampling strategy."""
        if self.sampling_strategy == 'smote':
            return SMOTE(random_state=42)
        elif self.sampling_strategy == 'adasyn':
            return ADASYN(random_state=42)
        elif self.sampling_strategy == 'undersample':
            return RandomUnderSampler(random_state=42)
        else:
            return None
    
    def _get_model(self, scale_pos_weight: float):
        """Get model instance."""
        if self.model_type == 'xgboost':
            return xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=scale_pos_weight,
                eval_metric='auc',
                random_state=42
            )
        elif self.model_type == 'lightgbm':
            return lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=scale_pos_weight,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None
    ) -> Dict[str, float]:
        """
        Train fraud detection model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Dictionary with training metrics
        """
        logger.info("Training fraud detection model...")
        
        # Calculate scale_pos_weight if not provided
        if self.scale_pos_weight is None:
            neg_count = np.sum(y_train == 0)
            pos_count = np.sum(y_train == 1)
            self.scale_pos_weight = neg_count / pos_count
            logger.info(f"Auto-calculated scale_pos_weight: {self.scale_pos_weight:.2f}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Apply sampling strategy
        sampler = self._get_sampler()
        if sampler is not None:
            X_train_resampled, y_train_resampled = sampler.fit_resample(
                X_train_scaled, y_train
            )
            logger.info(f"After {self.sampling_strategy}: {len(y_train_resampled)} samples")
        else:
            X_train_resampled = X_train_scaled
            y_train_resampled = y_train
        
        # Train model
        self.model = self._get_model(self.scale_pos_weight)
        
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            self.model.fit(
                X_train_resampled,
                y_train_resampled,
                eval_set=[(X_val_scaled, y_val)],
                verbose=False
            )
        else:
            self.model.fit(X_train_resampled, y_train_resampled)
        
        # Evaluate on training set
        y_pred = self.model.predict(X_train_scaled)
        y_pred_proba = self.model.predict_proba(X_train_scaled)[:, 1]
        
        metrics = {
            'train_auc': roc_auc_score(y_train, y_pred_proba),
        }
        
        logger.success(f"Model trained. Train AUC: {metrics['train_auc']:.4f}")
        
        return metrics
    
    def predict(
        self,
        X: np.ndarray,
        threshold: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict fraud probability.
        
        Args:
            X: Features
            threshold: Classification threshold
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        X_scaled = self.scaler.transform(X)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        predictions = (probabilities >= threshold).astype(int)
        
        return predictions, probabilities
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test labels
            threshold: Classification threshold
            
        Returns:
            Dictionary with evaluation metrics
        """
        predictions, probabilities = self.predict(X_test, threshold)
        
        # Calculate metrics
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, confusion_matrix
        )
        
        metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions),
            'recall': recall_score(y_test, predictions),
            'f1_score': f1_score(y_test, predictions),
            'roc_auc': roc_auc_score(y_test, probabilities)
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_test, predictions)
        metrics['true_negatives'] = int(cm[0, 0])
        metrics['false_positives'] = int(cm[0, 1])
        metrics['false_negatives'] = int(cm[1, 0])
        metrics['true_positives'] = int(cm[1, 1])
        
        logger.info(f"Evaluation metrics: {metrics}")
        
        return metrics
    
    def get_feature_importance(self, feature_names: list = None) -> pd.DataFrame:
        """
        Get feature importance.
        
        Args:
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            
            if feature_names is None:
                feature_names = [f'feature_{i}' for i in range(len(importance))]
            
            df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            return df
        else:
            logger.warning("Model does not have feature_importances_ attribute")
            return None


if __name__ == "__main__":
    # Example usage with synthetic data
    from sklearn.datasets import make_classification
    
    # Generate imbalanced dataset
    X, y = make_classification(
        n_samples=10000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_clusters_per_class=2,
        weights=[0.95, 0.05],  # 5% fraud
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Train detector
    detector = FraudDetector(
        model_type='xgboost',
        sampling_strategy='smote'
    )
    
    detector.train(X_train, y_train)
    
    # Evaluate
    metrics = detector.evaluate(X_test, y_test, threshold=0.5)
    
    print("\nFraud Detection Results:")
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")
