#!/usr/bin/env python3
"""
ML Model Management System
Handles model storage, versioning, deployment, and performance monitoring
"""

import asyncio
import asyncpg
import json
import pickle
import hashlib
import os
import shutil
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
import logging
import boto3
from botocore.exceptions import ClientError
import torch
import joblib
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class ModelMetadata:
    model_id: str
    model_name: str
    model_version: str
    model_type: str  # 'sklearn', 'pytorch', 'tensorflow', 'xgboost'
    model_architecture: Dict[str, Any]
    training_config: Dict[str, Any]
    performance_metrics: Dict[str, float]
    model_size_bytes: int
    storage_path: str
    deployment_status: str  # 'TRAINING', 'READY', 'DEPLOYED', 'DEPRECATED'
    created_by: str
    created_at: datetime
    deployed_at: Optional[datetime]
    deprecated_at: Optional[datetime]

@dataclass
class ModelPerformance:
    model_id: str
    evaluation_date: datetime
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    avg_inference_time_ms: float
    prediction_count: int
    drift_score: float
    is_healthy: bool

class ModelManager:
    """Manages ML model lifecycle, storage, and deployment"""
    
    def __init__(self):
        self.pg_dsn = os.getenv('POSTGRES_URL', 'postgresql://fraud_admin:FraudDetection2024!@localhost:5432/fraud_detection')
        self.model_storage_path = os.getenv('MODEL_STORAGE_PATH', './models')
        self.use_s3 = os.getenv('USE_S3_STORAGE', 'false').lower() == 'true'
        
        # S3 configuration
        if self.use_s3:
            self.s3_bucket = os.getenv('S3_BUCKET', 'fraud-detection-models')
            self.s3_client = boto3.client('s3')
        
        self.pool = None
        
        # Create local storage directory
        Path(self.model_storage_path).mkdir(parents=True, exist_ok=True)
    
    async def initialize(self):
        """Initialize database connection and model tables"""
        self.pool = await asyncpg.create_pool(
            self.pg_dsn,
            min_size=5,
            max_size=20,
            command_timeout=60
        )
        
        await self.create_model_tables()
        logger.info("Model manager initialized")
    
    async def create_model_tables(self):
        """Create model management tables"""
        async with self.pool.acquire() as conn:
            # AI models metadata table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS ai_models (
                    model_id VARCHAR(100) PRIMARY KEY,
                    model_name VARCHAR(100) NOT NULL,
                    model_version VARCHAR(20) NOT NULL,
                    model_type VARCHAR(50) NOT NULL,
                    model_architecture JSONB NOT NULL,
                    training_config JSONB NOT NULL,
                    performance_metrics JSONB DEFAULT '{}',
                    model_size_bytes BIGINT DEFAULT 0,
                    storage_path TEXT,
                    deployment_status VARCHAR(20) DEFAULT 'TRAINING',
                    created_by VARCHAR(100),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    deployed_at TIMESTAMP WITH TIME ZONE,
                    deprecated_at TIMESTAMP WITH TIME ZONE,
                    UNIQUE(model_name, model_version)
                )
            """)
            
            # Model performance tracking
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS ml_model_performance (
                    id BIGSERIAL PRIMARY KEY,
                    model_id VARCHAR(100) REFERENCES ai_models(model_id),
                    evaluation_date DATE NOT NULL,
                    accuracy DECIMAL(5,4),
                    precision_score DECIMAL(5,4),
                    recall DECIMAL(5,4),
                    f1_score DECIMAL(5,4),
                    auc_roc DECIMAL(5,4),
                    avg_inference_time_ms DECIMAL(8,2),
                    prediction_count INTEGER DEFAULT 0,
                    drift_score DECIMAL(5,4) DEFAULT 0.0000,
                    is_healthy BOOLEAN DEFAULT TRUE,
                    data_window_start TIMESTAMP WITH TIME ZONE,
                    data_window_end TIMESTAMP WITH TIME ZONE,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """)
            
            # Model experiments tracking
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS model_experiments (
                    experiment_id VARCHAR(100) PRIMARY KEY,
                    experiment_name VARCHAR(100) NOT NULL,
                    model_type VARCHAR(50) NOT NULL,
                    model_version VARCHAR(20) NOT NULL,
                    hyperparameters JSONB NOT NULL,
                    training_data_hash VARCHAR(64),
                    feature_set_version VARCHAR(20),
                    validation_metrics JSONB,
                    training_duration_minutes INTEGER,
                    status VARCHAR(20) DEFAULT 'RUNNING',
                    created_by VARCHAR(100),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    completed_at TIMESTAMP WITH TIME ZONE
                )
            """)
            
            # Model A/B testing
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS model_ab_tests (
                    test_id VARCHAR(100) PRIMARY KEY,
                    test_name VARCHAR(100) NOT NULL,
                    model_a_id VARCHAR(100) REFERENCES ai_models(model_id),
                    model_b_id VARCHAR(100) REFERENCES ai_models(model_id),
                    traffic_split DECIMAL(3,2) DEFAULT 0.5,
                    start_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    end_date TIMESTAMP WITH TIME ZONE,
                    status VARCHAR(20) DEFAULT 'ACTIVE',
                    results JSONB DEFAULT '{}',
                    created_by VARCHAR(100),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """)
            
            # Create indexes
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_ai_models_name_version ON ai_models (model_name, model_version)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_ai_models_status ON ai_models (deployment_status)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_model_performance_model_date ON ml_model_performance (model_id, evaluation_date DESC)")
    
    def _generate_model_id(self, model_name: str, model_version: str) -> str:
        """Generate unique model ID"""
        return f"{model_name}_{model_version}_{hashlib.md5(f'{model_name}{model_version}{datetime.now()}'.encode()).hexdigest()[:8]}"
    
    def _calculate_model_hash(self, model_data: bytes) -> str:
        """Calculate hash of model data"""
        return hashlib.sha256(model_data).hexdigest()
    
    async def save_model(self, model: Any, metadata: Dict[str, Any]) -> str:
        """Save model with metadata"""
        try:
            model_id = self._generate_model_id(metadata['model_name'], metadata['model_version'])
            
            # Serialize model based on type
            model_data = None
            model_extension = None
            
            if metadata['model_type'] == 'sklearn':
                model_data = pickle.dumps(model)
                model_extension = '.pkl'
            elif metadata['model_type'] == 'pytorch':
                import io
                buffer = io.BytesIO()
                torch.save(model, buffer)
                model_data = buffer.getvalue()
                model_extension = '.pth'
            elif metadata['model_type'] == 'joblib':
                import io
                buffer = io.BytesIO()
                joblib.dump(model, buffer)
                model_data = buffer.getvalue()
                model_extension = '.joblib'
            else:
                # Default to pickle
                model_data = pickle.dumps(model)
                model_extension = '.pkl'
            
            # Calculate model size and hash
            model_size = len(model_data)
            model_hash = self._calculate_model_hash(model_data)
            
            # Determine storage path
            if self.use_s3:
                storage_path = f"models/{model_id}{model_extension}"
                # Upload to S3
                self.s3_client.put_object(
                    Bucket=self.s3_bucket,
                    Key=storage_path,
                    Body=model_data,
                    Metadata={
                        'model_id': model_id,
                        'model_name': metadata['model_name'],
                        'model_version': metadata['model_version'],
                        'model_hash': model_hash
                    }
                )
            else:
                # Save locally
                storage_path = os.path.join(self.model_storage_path, f"{model_id}{model_extension}")
                with open(storage_path, 'wb') as f:
                    f.write(model_data)
            
            # Save metadata to database
            model_metadata = ModelMetadata(
                model_id=model_id,
                model_name=metadata['model_name'],
                model_version=metadata['model_version'],
                model_type=metadata['model_type'],
                model_architecture=metadata.get('model_architecture', {}),
                training_config=metadata.get('training_config', {}),
                performance_metrics=metadata.get('performance_metrics', {}),
                model_size_bytes=model_size,
                storage_path=storage_path,
                deployment_status='READY',
                created_by=metadata.get('created_by', 'system'),
                created_at=datetime.now(timezone.utc),
                deployed_at=None,
                deprecated_at=None
            )
            
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO ai_models 
                    (model_id, model_name, model_version, model_type, model_architecture,
                     training_config, performance_metrics, model_size_bytes, storage_path,
                     deployment_status, created_by, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                """, (
                    model_metadata.model_id, model_metadata.model_name, model_metadata.model_version,
                    model_metadata.model_type, model_metadata.model_architecture,
                    model_metadata.training_config, model_metadata.performance_metrics,
                    model_metadata.model_size_bytes, model_metadata.storage_path,
                    model_metadata.deployment_status, model_metadata.created_by,
                    model_metadata.created_at
                ))
            
            logger.info(f"Saved model {model_id} ({model_size} bytes)")
            return model_id
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
    
    async def load_model(self, model_id: str) -> Any:
        """Load model by ID"""
        try:
            # Get model metadata
            async with self.pool.acquire() as conn:
                model_row = await conn.fetchrow("""
                    SELECT model_type, storage_path, deployment_status
                    FROM ai_models
                    WHERE model_id = $1
                """, model_id)
            
            if not model_row:
                raise ValueError(f"Model {model_id} not found")
            
            if model_row['deployment_status'] == 'DEPRECATED':
                raise ValueError(f"Model {model_id} is deprecated")
            
            # Load model data
            if self.use_s3:
                # Download from S3
                response = self.s3_client.get_object(
                    Bucket=self.s3_bucket,
                    Key=model_row['storage_path']
                )
                model_data = response['Body'].read()
            else:
                # Load from local storage
                with open(model_row['storage_path'], 'rb') as f:
                    model_data = f.read()
            
            # Deserialize model based on type
            if model_row['model_type'] == 'sklearn':
                model = pickle.loads(model_data)
            elif model_row['model_type'] == 'pytorch':
                import io
                buffer = io.BytesIO(model_data)
                model = torch.load(buffer)
            elif model_row['model_type'] == 'joblib':
                import io
                buffer = io.BytesIO(model_data)
                model = joblib.load(buffer)
            else:
                # Default to pickle
                model = pickle.loads(model_data)
            
            logger.info(f"Loaded model {model_id}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            raise
    
    async def deploy_model(self, model_id: str) -> bool:
        """Deploy model to production"""
        try:
            async with self.pool.acquire() as conn:
                # Update deployment status
                await conn.execute("""
                    UPDATE ai_models
                    SET deployment_status = 'DEPLOYED', deployed_at = NOW()
                    WHERE model_id = $1
                """, model_id)
                
                # Deprecate previous versions of the same model
                model_info = await conn.fetchrow("""
                    SELECT model_name FROM ai_models WHERE model_id = $1
                """, model_id)
                
                if model_info:
                    await conn.execute("""
                        UPDATE ai_models
                        SET deployment_status = 'DEPRECATED', deprecated_at = NOW()
                        WHERE model_name = $1 AND model_id != $2 AND deployment_status = 'DEPLOYED'
                    """, model_info['model_name'], model_id)
            
            logger.info(f"Deployed model {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to deploy model {model_id}: {e}")
            return False
    
    async def get_model_metadata(self, model_id: str) -> Optional[ModelMetadata]:
        """Get model metadata"""
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT * FROM ai_models WHERE model_id = $1
                """, model_id)
            
            if row:
                return ModelMetadata(**dict(row))
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get model metadata: {e}")
            return None
    
    async def list_models(self, model_name: str = None, deployment_status: str = None) -> List[ModelMetadata]:
        """List models with optional filters"""
        try:
            query = "SELECT * FROM ai_models WHERE 1=1"
            params = []
            
            if model_name:
                query += " AND model_name = $" + str(len(params) + 1)
                params.append(model_name)
            
            if deployment_status:
                query += " AND deployment_status = $" + str(len(params) + 1)
                params.append(deployment_status)
            
            query += " ORDER BY created_at DESC"
            
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query, *params)
            
            return [ModelMetadata(**dict(row)) for row in rows]
            
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    async def record_model_performance(self, model_id: str, performance_data: Dict[str, Any]) -> bool:
        """Record model performance metrics"""
        try:
            performance = ModelPerformance(
                model_id=model_id,
                evaluation_date=performance_data.get('evaluation_date', datetime.now(timezone.utc).date()),
                accuracy=performance_data.get('accuracy', 0.0),
                precision=performance_data.get('precision', 0.0),
                recall=performance_data.get('recall', 0.0),
                f1_score=performance_data.get('f1_score', 0.0),
                auc_roc=performance_data.get('auc_roc', 0.0),
                avg_inference_time_ms=performance_data.get('avg_inference_time_ms', 0.0),
                prediction_count=performance_data.get('prediction_count', 0),
                drift_score=performance_data.get('drift_score', 0.0),
                is_healthy=performance_data.get('is_healthy', True)
            )
            
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO ml_model_performance
                    (model_id, evaluation_date, accuracy, precision_score, recall, f1_score,
                     auc_roc, avg_inference_time_ms, prediction_count, drift_score, is_healthy,
                     data_window_start, data_window_end)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                """, (
                    performance.model_id, performance.evaluation_date, performance.accuracy,
                    performance.precision, performance.recall, performance.f1_score,
                    performance.auc_roc, performance.avg_inference_time_ms,
                    performance.prediction_count, performance.drift_score, performance.is_healthy,
                    performance_data.get('data_window_start'),
                    performance_data.get('data_window_end')
                ))
            
            logger.info(f"Recorded performance for model {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to record model performance: {e}")
            return False
    
    async def get_model_performance_history(self, model_id: str, days: int = 30) -> List[ModelPerformance]:
        """Get model performance history"""
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT * FROM ml_model_performance
                    WHERE model_id = $1 AND evaluation_date >= CURRENT_DATE - INTERVAL '%s days'
                    ORDER BY evaluation_date DESC
                """, model_id, days)
            
            return [ModelPerformance(**dict(row)) for row in rows]
            
        except Exception as e:
            logger.error(f"Failed to get model performance history: {e}")
            return []
    
    async def create_ab_test(self, test_name: str, model_a_id: str, model_b_id: str,
                           traffic_split: float = 0.5, duration_days: int = 7) -> str:
        """Create A/B test between two models"""
        try:
            test_id = f"ab_test_{hashlib.md5(f'{test_name}{datetime.now()}'.encode()).hexdigest()[:8]}"
            end_date = datetime.now(timezone.utc) + timedelta(days=duration_days)
            
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO model_ab_tests
                    (test_id, test_name, model_a_id, model_b_id, traffic_split, end_date, created_by)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                """, (
                    test_id, test_name, model_a_id, model_b_id,
                    traffic_split, end_date, 'system'
                ))
            
            logger.info(f"Created A/B test {test_id}")
            return test_id
            
        except Exception as e:
            logger.error(f"Failed to create A/B test: {e}")
            raise
    
    async def delete_model(self, model_id: str) -> bool:
        """Delete model and its files"""
        try:
            # Get model metadata
            metadata = await self.get_model_metadata(model_id)
            if not metadata:
                return False
            
            # Delete model file
            if self.use_s3:
                self.s3_client.delete_object(
                    Bucket=self.s3_bucket,
                    Key=metadata.storage_path
                )
            else:
                if os.path.exists(metadata.storage_path):
                    os.remove(metadata.storage_path)
            
            # Delete from database
            async with self.pool.acquire() as conn:
                await conn.execute("DELETE FROM ai_models WHERE model_id = $1", model_id)
            
            logger.info(f"Deleted model {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete model {model_id}: {e}")
            return False
    
    async def close(self):
        """Close database connection"""
        if self.pool:
            await self.pool.close()

# Example usage
async def main():
    model_manager = ModelManager()
    await model_manager.initialize()
    
    try:
        # Create a simple sklearn model for testing
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        
        # Generate sample data
        X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Save model
        metadata = {
            'model_name': 'fraud_detector',
            'model_version': '1.0.0',
            'model_type': 'sklearn',
            'model_architecture': {
                'algorithm': 'RandomForest',
                'n_estimators': 100,
                'max_depth': None
            },
            'training_config': {
                'training_samples': 1000,
                'features': 20,
                'random_state': 42
            },
            'performance_metrics': {
                'accuracy': 0.95,
                'precision': 0.94,
                'recall': 0.96,
                'f1_score': 0.95
            },
            'created_by': 'test_user'
        }
        
        model_id = await model_manager.save_model(model, metadata)
        print(f"Saved model: {model_id}")
        
        # Load model
        loaded_model = await model_manager.load_model(model_id)
        print(f"Loaded model type: {type(loaded_model)}")
        
        # Deploy model
        await model_manager.deploy_model(model_id)
        print(f"Deployed model: {model_id}")
        
        # Record performance
        await model_manager.record_model_performance(model_id, {
            'accuracy': 0.95,
            'precision': 0.94,
            'recall': 0.96,
            'f1_score': 0.95,
            'auc_roc': 0.98,
            'avg_inference_time_ms': 15.5,
            'prediction_count': 1000,
            'is_healthy': True
        })
        
        # List models
        models = await model_manager.list_models()
        print(f"Total models: {len(models)}")
        
        print("✅ Model manager test completed successfully!")
        
    except Exception as e:
        print(f"❌ Model manager test failed: {e}")
    finally:
        await model_manager.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())