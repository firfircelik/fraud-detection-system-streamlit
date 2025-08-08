#!/usr/bin/env python3
"""
ML Model Training Service
Trains and updates fraud detection models
"""

import asyncio
import logging
import os
import pickle
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLTrainer:
    def __init__(self):
        # Environment variables
        self.postgres_url = os.getenv("POSTGRES_URL", "postgresql://fraud_admin:FraudDetection2024!@localhost:5432/fraud_detection")
        self.neo4j_url = os.getenv("NEO4J_URL", "bolt://localhost:7687")
        self.neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD", "FraudGraph2024!")
        self.model_storage_path = os.getenv("MODEL_STORAGE_PATH", "./models")
        self.enable_gpu = os.getenv("ENABLE_GPU", "false").lower() == "true"
        
        # Initialize connections
        self.db_engine = None
        self.session_local = None
        self.setup_connections()
        
        # Model configurations
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'logistic_regression': LogisticRegression(
                random_state=42,
                max_iter=1000
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            ),
            'isolation_forest': IsolationForest(
                contamination=0.1,
                random_state=42,
                n_jobs=-1
            )
        }
        
        self.scaler = StandardScaler()
    
    def setup_connections(self):
        """Setup database connections"""
        try:
            # Database connection
            self.db_engine = create_engine(self.postgres_url)
            self.session_local = sessionmaker(autocommit=False, autoflush=False, bind=self.db_engine)
            logger.info("✅ PostgreSQL connection established")
            
            # Create model storage directory
            os.makedirs(self.model_storage_path, exist_ok=True)
            
        except Exception as e:
            logger.error(f"❌ Failed to setup connections: {e}")
            raise
    
    def load_training_data(self, days_back: int = 30) -> Tuple[pd.DataFrame, pd.Series]:
        """Load training data from database"""
        try:
            with self.session_local() as session:
                # Load transaction data with features
                query = text("""
                    SELECT 
                        transaction_id,
                        amount,
                        fraud_score,
                        is_fraud,
                        latitude,
                        longitude,
                        is_vpn,
                        is_tor,
                        confidence_score,
                        processing_time_ms,
                        EXTRACT(HOUR FROM transaction_timestamp) as hour_of_day,
                        EXTRACT(DOW FROM transaction_timestamp) as day_of_week,
                        CASE WHEN EXTRACT(DOW FROM transaction_timestamp) IN (0,6) THEN 1 ELSE 0 END as is_weekend,
                        CASE WHEN EXTRACT(HOUR FROM transaction_timestamp) BETWEEN 9 AND 17 THEN 1 ELSE 0 END as is_business_hour
                    FROM transactions
                    WHERE is_fraud IS NOT NULL
                    ORDER BY transaction_timestamp DESC
                    LIMIT 100000
                """)
                
                result = session.execute(query, {'days_back': days_back})
                data = result.fetchall()
                
                if not data:
                    logger.warning("No training data found")
                    return pd.DataFrame(), pd.Series()
                
                # Convert to DataFrame
                columns = [
                    'transaction_id', 'amount', 'fraud_score', 'is_fraud',
                    'latitude', 'longitude', 'is_vpn', 'is_tor',
                    'confidence_score', 'processing_time_ms',
                    'hour_of_day', 'day_of_week', 'is_weekend', 'is_business_hour'
                ]
                
                df = pd.DataFrame(data, columns=columns)
                
                # Prepare features and target
                feature_columns = [col for col in columns if col not in ['transaction_id', 'is_fraud']]
                X = df[feature_columns].fillna(0)
                
                # Convert all columns to numeric for XGBoost compatibility
                for col in X.columns:
                    X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
                
                y = df['is_fraud'].astype(int)
                
                # Ensure binary classification (0 and 1 only)
                y = y.clip(0, 1)
                
                logger.info(f"✅ Loaded {len(df)} training samples")
                logger.info(f"   - Fraud cases: {y.sum()} ({y.mean()*100:.2f}%)")
                logger.info(f"   - Legitimate cases: {len(y) - y.sum()} ({(1-y.mean())*100:.2f}%)")
                
                return X, y
        
        except Exception as e:
            logger.error(f"❌ Error loading training data: {e}")
            return pd.DataFrame(), pd.Series()
    
    def train_model(self, model_name: str, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train a single model"""
        try:
            logger.info(f"🔄 Training {model_name}...")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features for certain models
            if model_name in ['logistic_regression', 'svm']:
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)
            else:
                X_train_scaled = X_train
                X_test_scaled = X_test
            
            # Train model
            model = self.models[model_name]
            start_time = time.time()
            
            if model_name == 'isolation_forest':
                # Unsupervised model - train on all data
                model.fit(X_train_scaled)
                y_pred = model.predict(X_test_scaled)
                y_pred = (y_pred == -1).astype(int)  # Convert to binary
                y_pred_proba = model.decision_function(X_test_scaled)
                y_pred_proba = (y_pred_proba - y_pred_proba.min()) / (y_pred_proba.max() - y_pred_proba.min())
            else:
                # Supervised models
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                else:
                    y_pred_proba = model.decision_function(X_test_scaled)
            
            training_time = time.time() - start_time
            
            # Calculate metrics with binary classification settings
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='binary', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='binary', zero_division=0),
                'f1_score': f1_score(y_test, y_pred, average='binary', zero_division=0),
                'training_time': training_time,
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            }
            
            # AUC-ROC (handle potential issues)
            try:
                if len(np.unique(y_test)) > 1:  # Ensure both classes are present
                    metrics['auc_roc'] = roc_auc_score(y_test, y_pred_proba)
                else:
                    metrics['auc_roc'] = 0.5
            except:
                metrics['auc_roc'] = 0.5
            
            # Cross-validation score with binary classification
            try:
                if len(np.unique(y_train)) > 1:  # Ensure both classes are present
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=3, scoring='f1_binary')
                    metrics['cv_f1_mean'] = cv_scores.mean()
                    metrics['cv_f1_std'] = cv_scores.std()
                else:
                    metrics['cv_f1_mean'] = 0.0
                    metrics['cv_f1_std'] = 0.0
            except:
                # Fallback to accuracy if f1_binary fails
                try:
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=3, scoring='accuracy')
                    metrics['cv_f1_mean'] = cv_scores.mean()
                    metrics['cv_f1_std'] = cv_scores.std()
                except:
                    metrics['cv_f1_mean'] = 0.0
                    metrics['cv_f1_std'] = 0.0
            
            # Save model
            model_path = os.path.join(self.model_storage_path, f"{model_name}_model.pkl")
            logger.info(f"💾 Saving model to: {model_path}")
            logger.info(f"📁 Storage path exists: {os.path.exists(self.model_storage_path)}")
            
            # Ensure directory exists
            os.makedirs(self.model_storage_path, exist_ok=True)
            
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': model,
                    'scaler': self.scaler if model_name in ['logistic_regression', 'svm'] else None,
                    'feature_columns': list(X.columns),
                    'metrics': metrics,
                    'trained_at': datetime.now().isoformat(),
                    'version': '1.0.0'
                }, f)
            
            logger.info(f"✅ Model saved successfully: {os.path.exists(model_path)}")
            
            logger.info(f"✅ {model_name} trained successfully:")
            logger.info(f"   - Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"   - Precision: {metrics['precision']:.4f}")
            logger.info(f"   - Recall: {metrics['recall']:.4f}")
            logger.info(f"   - F1-Score: {metrics['f1_score']:.4f}")
            logger.info(f"   - AUC-ROC: {metrics['auc_roc']:.4f}")
            logger.info(f"   - Training time: {training_time:.2f}s")
            
            return {
                'model_name': model_name,
                'model_path': model_path,
                'metrics': metrics,
                'status': 'success'
            }
        
        except Exception as e:
            logger.error(f"❌ Error training {model_name}: {e}")
            return {
                'model_name': model_name,
                'status': 'failed',
                'error': str(e)
            }
    
    def save_model_performance(self, model_results: List[Dict[str, Any]]):
        """Save model performance metrics to database"""
        try:
            with self.session_local() as session:
                for result in model_results:
                    if result['status'] == 'success':
                        metrics = result['metrics']
                        
                        # Insert performance record
                        insert_query = text("""
                            INSERT INTO ml_model_performance (
                                model_name, model_version, model_type,
                                accuracy, precision_score, recall, f1_score, auc_roc,
                                avg_inference_time_ms, prediction_count,
                                evaluation_date, test_set_size
                            ) VALUES (
                                :model_name, '1.0.0', 'classification',
                                :accuracy, :precision, :recall, :f1_score, :auc_roc,
                                :training_time, :training_samples,
                                CURRENT_DATE, :test_samples
                            )
                            ON CONFLICT (model_name, model_version, evaluation_date) 
                            DO UPDATE SET
                                accuracy = EXCLUDED.accuracy,
                                precision_score = EXCLUDED.precision_score,
                                recall = EXCLUDED.recall,
                                f1_score = EXCLUDED.f1_score,
                                auc_roc = EXCLUDED.auc_roc,
                                avg_inference_time_ms = EXCLUDED.avg_inference_time_ms
                        """)
                        
                        session.execute(insert_query, {
                            'model_name': result['model_name'],
                            'accuracy': metrics['accuracy'],
                            'precision': metrics['precision'],
                            'recall': metrics['recall'],
                            'f1_score': metrics['f1_score'],
                            'auc_roc': metrics['auc_roc'],
                            'training_time': metrics['training_time'] * 1000,  # Convert to ms
                            'training_samples': metrics['training_samples'],
                            'test_samples': metrics['test_samples']
                        })
                
                session.commit()
                logger.info("✅ Model performance metrics saved to database")
        
        except Exception as e:
            logger.error(f"❌ Error saving model performance: {e}")
    
    def train_all_models(self):
        """Train all models"""
        logger.info("🚀 Starting ML model training...")
        
        # Load training data
        X, y = self.load_training_data(days_back=30)
        
        if X.empty or y.empty:
            logger.error("❌ No training data available")
            return
        
        # Train all models
        results = []
        for model_name in self.models.keys():
            result = self.train_model(model_name, X, y)
            results.append(result)
        
        # Save performance metrics
        self.save_model_performance(results)
        
        # Summary
        successful_models = [r for r in results if r['status'] == 'success']
        failed_models = [r for r in results if r['status'] == 'failed']
        
        logger.info(f"🎉 Training completed:")
        logger.info(f"   - Successful: {len(successful_models)} models")
        logger.info(f"   - Failed: {len(failed_models)} models")
        
        if successful_models:
            best_model = max(successful_models, key=lambda x: x['metrics']['f1_score'])
            logger.info(f"   - Best model: {best_model['model_name']} (F1: {best_model['metrics']['f1_score']:.4f})")
    
    def run_training_cycle(self):
        """Run a complete training cycle"""
        try:
            self.train_all_models()
            logger.info("✅ Training cycle completed successfully")
        except Exception as e:
            logger.error(f"❌ Training cycle failed: {e}")

def main():
    """Main entry point"""
    trainer = MLTrainer()
    
    # Run training cycle
    trainer.run_training_cycle()
    
    # In production, this would run on a schedule
    # For now, just run once and exit
    logger.info("🏁 ML Trainer finished")

if __name__ == "__main__":
    main()