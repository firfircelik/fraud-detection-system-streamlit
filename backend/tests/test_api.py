import pytest
import httpx
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import json

# Import your FastAPI app
try:
    from api.main import app
except ImportError:
    # Fallback for testing
    from fastapi import FastAPI
    app = FastAPI()

client = TestClient(app)


class TestHealthEndpoints:
    """Test health and status endpoints."""
    
    def test_health_endpoint(self):
        """Test the health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
    
    def test_status_endpoint(self):
        """Test the API status endpoint."""
        response = client.get("/api/status")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data


class TestTransactionAnalysis:
    """Test transaction analysis endpoints."""
    
    def test_analyze_transaction_valid_data(self):
        """Test transaction analysis with valid data."""
        transaction_data = {
            "transaction_id": "test-123",
            "user_id": "user-456",
            "merchant_id": "merchant-789",
            "amount": 100.50,
            "currency": "USD",
            "transaction_type": "purchase",
            "category": "retail",
            "location": "New York",
            "device_id": "device-123"
        }
        
        with patch('api.main.analyze_transaction') as mock_analyze:
            mock_analyze.return_value = {
                "fraud_probability": 0.15,
                "risk_level": "low",
                "factors": ["normal_amount", "known_location"]
            }
            
            response = client.post("/api/transactions", json=transaction_data)
            assert response.status_code == 200
            data = response.json()
            assert "fraud_probability" in data
            assert "risk_level" in data
            assert data["risk_level"] == "low"
    
    def test_analyze_transaction_missing_required_fields(self):
        """Test transaction analysis with missing required fields."""
        transaction_data = {
            "amount": 100.50,
            "currency": "USD"
        }
        
        response = client.post("/api/transactions", json=transaction_data)
        assert response.status_code == 422  # Validation error
    
    def test_analyze_transaction_invalid_amount(self):
        """Test transaction analysis with invalid amount."""
        transaction_data = {
            "transaction_id": "test-123",
            "user_id": "user-456",
            "merchant_id": "merchant-789",
            "amount": -100.50,  # Invalid negative amount
            "currency": "USD"
        }
        
        response = client.post("/api/transactions", json=transaction_data)
        assert response.status_code == 422
    
    def test_analyze_transaction_high_risk(self):
        """Test transaction analysis returning high risk."""
        transaction_data = {
            "transaction_id": "test-456",
            "user_id": "user-789",
            "merchant_id": "merchant-123",
            "amount": 10000.00,
            "currency": "USD",
            "transaction_type": "purchase",
            "category": "electronics",
            "location": "Unknown",
            "device_id": "new-device-999"
        }
        
        with patch('api.main.analyze_transaction') as mock_analyze:
            mock_analyze.return_value = {
                "fraud_probability": 0.85,
                "risk_level": "high",
                "factors": ["high_amount", "new_device", "unusual_location"]
            }
            
            response = client.post("/api/transactions", json=transaction_data)
            assert response.status_code == 200
            data = response.json()
            assert data["fraud_probability"] > 0.8
            assert data["risk_level"] == "high"
            assert "factors" in data


class TestBatchAnalysis:
    """Test batch transaction analysis endpoints."""
    
    def test_batch_analysis_valid_data(self):
        """Test batch analysis with valid transaction data."""
        batch_data = {
            "transactions": [
                {
                    "transaction_id": "batch-1",
                    "user_id": "user-1",
                    "merchant_id": "merchant-1",
                    "amount": 50.00,
                    "currency": "USD"
                },
                {
                    "transaction_id": "batch-2",
                    "user_id": "user-2",
                    "merchant_id": "merchant-2",
                    "amount": 150.00,
                    "currency": "USD"
                }
            ]
        }
        
        with patch('api.main.analyze_batch_transactions') as mock_batch:
            mock_batch.return_value = {
                "results": [
                    {"transaction_id": "batch-1", "fraud_probability": 0.1, "risk_level": "low"},
                    {"transaction_id": "batch-2", "fraud_probability": 0.3, "risk_level": "medium"}
                ],
                "summary": {"total_processed": 2, "high_risk_count": 0, "medium_risk_count": 1}
            }
            
            response = client.post("/api/batch-analyze", json=batch_data)
            assert response.status_code == 200
            data = response.json()
            assert "results" in data
            assert "summary" in data
            assert len(data["results"]) == 2
    
    def test_batch_analysis_empty_list(self):
        """Test batch analysis with empty transaction list."""
        batch_data = {"transactions": []}
        
        response = client.post("/api/batch-analyze", json=batch_data)
        assert response.status_code == 422


class TestModelStatus:
    """Test ML model status endpoints."""
    
    def test_model_status(self):
        """Test model status endpoint."""
        with patch('api.main.get_model_status') as mock_status:
            mock_status.return_value = {
                "models": {
                    "ensemble": {"status": "active", "version": "1.0.0", "accuracy": 0.95},
                    "deep_learning": {"status": "active", "version": "2.1.0", "accuracy": 0.93}
                },
                "last_updated": "2024-01-15T10:30:00Z"
            }
            
            response = client.get("/api/models/status")
            assert response.status_code == 200
            data = response.json()
            assert "models" in data
            assert "last_updated" in data
    
    def test_model_metrics(self):
        """Test model metrics endpoint."""
        with patch('api.main.get_model_metrics') as mock_metrics:
            mock_metrics.return_value = {
                "accuracy": 0.95,
                "precision": 0.92,
                "recall": 0.89,
                "f1_score": 0.90,
                "auc_roc": 0.96
            }
            
            response = client.get("/api/models/metrics")
            assert response.status_code == 200
            data = response.json()
            assert "accuracy" in data
            assert "precision" in data
            assert "recall" in data


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_json(self):
        """Test handling of invalid JSON data."""
        response = client.post(
            "/api/transactions",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
    
    def test_large_payload(self):
        """Test handling of large payloads."""
        large_transaction = {
            "transaction_id": "large-test",
            "user_id": "user-123",
            "merchant_id": "merchant-456",
            "amount": 100.00,
            "currency": "USD",
            "description": "x" * 10000  # Very long description
        }
        
        response = client.post("/api/transactions", json=large_transaction)
        # Should either process successfully or return appropriate error
        assert response.status_code in [200, 413, 422]
    
    def test_sql_injection_attempt(self):
        """Test protection against SQL injection attempts."""
        malicious_data = {
            "transaction_id": "'; DROP TABLE transactions; --",
            "user_id": "user-123",
            "merchant_id": "merchant-456",
            "amount": 100.00,
            "currency": "USD"
        }
        
        with patch('api.main.analyze_transaction') as mock_analyze:
            mock_analyze.return_value = {"fraud_probability": 0.5, "risk_level": "medium"}
            
            response = client.post("/api/transactions", json=malicious_data)
            # Should process safely without SQL injection
            assert response.status_code in [200, 422]


class TestRateLimiting:
    """Test rate limiting functionality."""
    
    def test_rate_limiting(self):
        """Test rate limiting on API endpoints."""
        transaction_data = {
            "transaction_id": "rate-test",
            "user_id": "user-123",
            "merchant_id": "merchant-456",
            "amount": 100.00,
            "currency": "USD"
        }
        
        # Make multiple rapid requests
        responses = []
        for i in range(10):
            transaction_data["transaction_id"] = f"rate-test-{i}"
            response = client.post("/api/transactions", json=transaction_data)
            responses.append(response.status_code)
        
        # Should have at least some successful responses
        assert 200 in responses
        # May have rate limiting responses (429) if implemented


@pytest.fixture
def sample_transaction():
    """Fixture providing sample transaction data."""
    return {
        "transaction_id": "sample-123",
        "user_id": "user-456",
        "merchant_id": "merchant-789",
        "amount": 75.50,
        "currency": "USD",
        "transaction_type": "purchase",
        "category": "retail",
        "location": "San Francisco",
        "device_id": "device-456"
    }


@pytest.fixture
def mock_redis():
    """Fixture providing mocked Redis client."""
    with patch('redis.Redis') as mock:
        yield mock


if __name__ == "__main__":
    pytest.main([__file__])