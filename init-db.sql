-- 🚨 Fraud Detection Database Schema
-- PostgreSQL initialization script

-- Create transactions table
CREATE TABLE IF NOT EXISTS transactions (
    id SERIAL PRIMARY KEY,
    transaction_id VARCHAR(255) UNIQUE NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    merchant_id VARCHAR(255) NOT NULL,
    amount DECIMAL(15,2) NOT NULL,
    currency VARCHAR(3) NOT NULL DEFAULT 'USD',
    category VARCHAR(100),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    device_id VARCHAR(255),
    ip_address INET,
    lat DECIMAL(10,8),
    lon DECIMAL(11,8),
    user_age INTEGER,
    user_income VARCHAR(50),
    is_fraud BOOLEAN DEFAULT FALSE,
    fraud_score DECIMAL(3,2) DEFAULT 0.0,
    risk_level VARCHAR(20) DEFAULT 'MINIMAL',
    decision VARCHAR(20) DEFAULT 'APPROVED',
    risk_factors JSONB DEFAULT '[]',
    recommendations JSONB DEFAULT '[]',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_transactions_user_id ON transactions(user_id);
CREATE INDEX IF NOT EXISTS idx_transactions_merchant_id ON transactions(merchant_id);
CREATE INDEX IF NOT EXISTS idx_transactions_timestamp ON transactions(timestamp);
CREATE INDEX IF NOT EXISTS idx_transactions_is_fraud ON transactions(is_fraud);
CREATE INDEX IF NOT EXISTS idx_transactions_amount ON transactions(amount);
CREATE INDEX IF NOT EXISTS idx_transactions_risk_level ON transactions(risk_level);

-- Create users table for user profiles
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255),
    age INTEGER,
    income_level VARCHAR(50),
    risk_profile VARCHAR(20) DEFAULT 'LOW',
    total_transactions INTEGER DEFAULT 0,
    fraud_incidents INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create merchants table
CREATE TABLE IF NOT EXISTS merchants (
    id SERIAL PRIMARY KEY,
    merchant_id VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    category VARCHAR(100),
    risk_score DECIMAL(3,2) DEFAULT 0.5,
    total_transactions INTEGER DEFAULT 0,
    fraud_incidents INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create fraud_patterns table for storing detected patterns
CREATE TABLE IF NOT EXISTS fraud_patterns (
    id SERIAL PRIMARY KEY,
    pattern_type VARCHAR(50) NOT NULL,
    description TEXT,
    confidence DECIMAL(3,2),
    affected_transactions INTEGER,
    pattern_data JSONB,
    detected_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Insert sample data for testing
INSERT INTO merchants (merchant_id, name, category, risk_score) VALUES
('merchant_001', 'Amazon', 'E-commerce', 0.1),
('merchant_002', 'Walmart', 'Retail', 0.1),
('merchant_003', 'Target', 'Retail', 0.2),
('merchant_004', 'Best Buy', 'Electronics', 0.3),
('merchant_005', 'Casino Royale', 'Gambling', 0.8),
('merchant_006', 'Crypto Exchange', 'Cryptocurrency', 0.9),
('merchant_007', 'Forex Trading', 'Financial Services', 0.7),
('merchant_008', 'Local Store', 'Retail', 0.4)
ON CONFLICT (merchant_id) DO NOTHING;

INSERT INTO users (user_id, email, age, income_level, risk_profile) VALUES
('user_001', 'john.doe@email.com', 35, '50k-100k', 'LOW'),
('user_002', 'jane.smith@email.com', 28, '30k-50k', 'LOW'),
('user_003', 'bob.wilson@email.com', 45, '100k-200k', 'MEDIUM'),
('user_004', 'alice.brown@email.com', 32, '50k-100k', 'LOW'),
('user_005', 'charlie.davis@email.com', 55, '200k+', 'HIGH')
ON CONFLICT (user_id) DO NOTHING;

-- Insert sample transactions for testing
INSERT INTO transactions (transaction_id, user_id, merchant_id, amount, currency, category, timestamp, device_id, ip_address, lat, lon, is_fraud, fraud_score, risk_level, decision) VALUES
('tx_001', 'user_001', 'merchant_001', 150.00, 'USD', 'Electronics', NOW() - INTERVAL '1 day', 'device_001', '192.168.1.100', 40.7128, -74.0060, false, 0.05, 'MINIMAL', 'APPROVED'),
('tx_002', 'user_002', 'merchant_002', 75.50, 'USD', 'Groceries', NOW() - INTERVAL '2 days', 'device_002', '192.168.1.101', 34.0522, -118.2437, false, 0.10, 'LOW', 'APPROVED'),
('tx_003', 'user_003', 'merchant_005', 5000.00, 'USD', 'Gambling', NOW() - INTERVAL '3 hours', 'device_003', '10.0.0.1', 36.1699, -115.1398, true, 0.85, 'HIGH', 'DECLINED'),
('tx_004', 'user_004', 'merchant_006', 2500.00, 'USD', 'Cryptocurrency', NOW() - INTERVAL '1 hour', 'device_004', '172.16.0.1', 51.5074, -0.1278, true, 0.75, 'HIGH', 'REVIEW');