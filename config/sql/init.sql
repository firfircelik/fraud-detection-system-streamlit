-- Fraud Detection System Database Initialization

-- Create database and user (already handled by docker-compose environment)
-- CREATE DATABASE fraud_detection;
-- CREATE USER fraud_user WITH PASSWORD 'fraud_password';
-- GRANT ALL PRIVILEGES ON DATABASE fraud_detection TO fraud_user;

-- Create tables for Akka Persistence
CREATE TABLE IF NOT EXISTS journal (
  ordering BIGSERIAL,
  persistence_id VARCHAR(255) NOT NULL,
  sequence_number BIGINT NOT NULL,
  deleted BOOLEAN DEFAULT FALSE,
  tags VARCHAR(255) DEFAULT NULL,
  message BYTEA NOT NULL,
  PRIMARY KEY(persistence_id, sequence_number)
);

CREATE UNIQUE INDEX journal_ordering_idx ON journal(ordering);

CREATE TABLE IF NOT EXISTS snapshot (
  persistence_id VARCHAR(255) NOT NULL,
  sequence_number BIGINT NOT NULL,
  created BIGINT NOT NULL,
  snapshot BYTEA NOT NULL,
  PRIMARY KEY(persistence_id, sequence_number)
);

-- Create application-specific tables
CREATE TABLE IF NOT EXISTS transactions (
  id UUID PRIMARY KEY,
  account_id VARCHAR(255) NOT NULL,
  merchant_id VARCHAR(255) NOT NULL,
  amount DECIMAL(15,2) NOT NULL,
  currency VARCHAR(3) NOT NULL,
  status VARCHAR(50) NOT NULL,
  timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
  metadata JSONB,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_transactions_account_id ON transactions(account_id);
CREATE INDEX idx_transactions_timestamp ON transactions(timestamp);
CREATE INDEX idx_transactions_status ON transactions(status);

CREATE TABLE IF NOT EXISTS fraud_scores (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  transaction_id UUID NOT NULL REFERENCES transactions(id),
  score DECIMAL(3,2) NOT NULL CHECK (score >= 0 AND score <= 1),
  risk_level VARCHAR(20) NOT NULL,
  factors JSONB,
  model_version VARCHAR(50) NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_fraud_scores_transaction_id ON fraud_scores(transaction_id);
CREATE INDEX idx_fraud_scores_risk_level ON fraud_scores(risk_level);

-- Insert some sample data
INSERT INTO transactions (id, account_id, merchant_id, amount, currency, status, timestamp, metadata) VALUES
('550e8400-e29b-41d4-a716-446655440000', 'acc_sample001', 'mer_amazon001', 99.99, 'USD', 'APPROVED', NOW() - INTERVAL '1 hour', '{"device_id": "device_123", "ip_address": "192.168.1.100"}'),
('550e8400-e29b-41d4-a716-446655440001', 'acc_sample002', 'mer_walmart001', 1500.00, 'USD', 'PENDING', NOW() - INTERVAL '30 minutes', '{"device_id": "device_456", "ip_address": "10.0.0.50"}'),
('550e8400-e29b-41d4-a716-446655440002', 'acc_sample001', 'mer_target001', 25.50, 'USD', 'APPROVED', NOW() - INTERVAL '15 minutes', '{"device_id": "device_123", "ip_address": "192.168.1.100"}');

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO fraud_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO fraud_user;