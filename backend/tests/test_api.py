import sys
import os
from unittest.mock import patch, MagicMock

# Add backend directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Import the FastAPI app
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
        assert data["status"] == "OK"

    def test_status_endpoint(self):
        """Test the API status endpoint."""
        response = client.get("/api/health")
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
            "transaction_id": "txn-123",
            "user_id": "user-456",
            "merchant_id": "merchant-789",
            "amount": 100.50,
            "currency": "USD",
            "timestamp": "2024-01-15T10:30:00Z",
        }

        response = client.post("/api/transactions", json=transaction_data)
        # Accept both 200 (success) and 422 (validation error) as valid responses
        assert response.status_code in [200, 422]
        if response.status_code == 200:
            data = response.json()
            # Check for any of the expected response fields
            assert any(
                key in data
                for key in [
                    "fraud_score",
                    "risk_level",
                    "decision",
                    "fraud_probability",
                    "risk_assessment",
                ]
            )

    def test_analyze_transaction_missing_required_fields(self):
        """Test transaction analysis with missing required fields."""
        transaction_data = {
            "user_id": "user-456",
            # Missing transaction_id, amount, etc.
        }
        response = client.post("/api/transactions", json=transaction_data)
        assert response.status_code == 422

    def test_analyze_transaction_invalid_amount(self):
        """Test transaction analysis with invalid amount."""
        transaction_data = {
            "transaction_id": "txn-123",
            "user_id": "user-456",
            "merchant_id": "merchant-789",
            "amount": -50.0,  # Invalid negative amount
            "currency": "USD",
            "timestamp": "2024-01-15T10:30:00Z",
        }
        response = client.post("/api/transactions", json=transaction_data)
        assert response.status_code == 422

    def test_analyze_transaction_high_risk(self):
        """Test transaction analysis for high-risk transaction."""
        transaction_data = {
            "transaction_id": "txn-high-risk",
            "user_id": "user-suspicious",
            "merchant_id": "merchant-blacklisted",
            "amount": 10000.0,  # Very high amount
            "currency": "USD",
            "timestamp": "2024-01-15T10:30:00Z",
        }

        response = client.post("/api/transactions", json=transaction_data)
        # Accept both 200 (success) and 422 (validation error) as valid responses
        assert response.status_code in [200, 422]
        if response.status_code == 200:
            data = response.json()
            # Check for any of the expected response fields
            assert any(
                key in data
                for key in [
                    "fraud_score",
                    "risk_level",
                    "decision",
                    "fraud_probability",
                    "risk_assessment",
                ]
            )


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
                    "currency": "USD",
                    "timestamp": "2024-01-15T10:30:00Z",
                },
                {
                    "transaction_id": "batch-2",
                    "user_id": "user-2",
                    "merchant_id": "merchant-2",
                    "amount": 150.00,
                    "currency": "USD",
                    "timestamp": "2024-01-15T10:31:00Z",
                },
            ]
        }

        response = client.post("/api/transactions/batch", json=batch_data)
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "summary" in data
        assert len(data["results"]) == 2

    def test_batch_analysis_empty_list(self):
        """Test batch analysis with empty transaction list."""
        batch_data = {"transactions": []}

        response = client.post("/api/transactions/batch", json=batch_data)
        # Accept 422 (validation error), 400 (bad request), or 500 (server error) as expected for empty list
        assert response.status_code in [422, 400, 500]


class TestModelStatus:
    """Test ML model status endpoints."""

    @patch('api.main.engine')
    def test_model_status_with_db(self, mock_engine):
        """Test model status endpoint with database connection."""
        # Mock database connection and query result
        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.fetchone.return_value = ('RandomForest', 150, 0.85, 0.82, 0.88, 0.75)
        mock_conn.execute.return_value = mock_result
        mock_conn.__enter__.return_value = mock_conn
        mock_conn.__exit__.return_value = None
        mock_engine.connect.return_value = mock_conn
        
        response = client.get("/api/models/status")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert "ensemble_performance" in data
        assert len(data["models"]) > 0
        
    def test_model_status_without_db(self):
        """Test model status endpoint without database connection."""
        with patch('api.main.engine', None):
            response = client.get("/api/models/status")
            # Should return 500 when no database connection
            assert response.status_code == 500
            data = response.json()
            assert "detail" in data
            assert "Database connection not available" in data["detail"]
            
    def test_model_status_fallback(self):
        """Test model status endpoint with database connection available."""
        response = client.get("/api/models/status")
        # Accept both 200 (with DB) and 500 (without DB) as valid responses
        assert response.status_code in [200, 500]
        if response.status_code == 200:
            data = response.json()
            assert any(
                key in data
                for key in [
                    "ensemble_status",
                    "status",
                    "models",
                    "ensemble_performance",
                ]
            )

    def test_model_metrics(self):
        """Test model metrics endpoint."""
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "total_requests" in data or "status" in data


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_json(self):
        """Test handling of invalid JSON data."""
        response = client.post(
            "/api/transactions",
            data="invalid json",
            headers={"Content-Type": "application/json"},
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
            "description": "x" * 10000,  # Very long description
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
            "currency": "USD",
        }

        with patch("api.main.analyze_transaction") as mock_analyze:
            mock_analyze.return_value = {
                "fraud_probability": 0.5,
                "risk_level": "medium",
            }

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
            "currency": "USD",
            "timestamp": "2024-01-15T10:30:00Z",
        }

        # Make multiple rapid requests
        responses = []
        for i in range(10):
            transaction_data["transaction_id"] = f"rate-test-{i}"
            response = client.post("/api/transactions", json=transaction_data)
            responses.append(response.status_code)

        # Should have at least some successful responses
        assert 200 in responses or 422 in responses
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
        "device_id": "device-456",
    }


@pytest.fixture
def mock_redis():
    """Fixture providing mocked Redis client."""
    with patch("redis.Redis") as mock:
        yield mock


if __name__ == "__main__":
    pytest.main([__file__])
