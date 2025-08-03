-- =====================================================
-- TIMESCALEDB INITIALIZATION SCRIPT
-- Time-series database for fraud detection metrics
-- =====================================================

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- =====================================================
-- SYSTEM METRICS TABLES
-- =====================================================

-- System performance metrics
CREATE TABLE system_metrics (
    time TIMESTAMPTZ NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value DOUBLE PRECISION NOT NULL,
    labels JSONB DEFAULT '{}',
    service_name TEXT,
    instance_id TEXT,
    environment TEXT DEFAULT 'production'
);

-- Convert to hypertable with 1-hour chunks
SELECT create_hypertable('system_metrics', 'time', chunk_time_interval => INTERVAL '1 hour');

-- Create indexes for efficient queries
CREATE INDEX idx_system_metrics_name_time ON system_metrics (metric_name, time DESC);
CREATE INDEX idx_system_metrics_service_time ON system_metrics (service_name, time DESC);
CREATE INDEX idx_system_metrics_labels ON system_metrics USING GIN (labels);

-- =====================================================
-- API USAGE METRICS
-- =====================================================

-- API request tracking
CREATE TABLE api_usage_metrics (
    time TIMESTAMPTZ NOT NULL,
    endpoint TEXT NOT NULL,
    method TEXT NOT NULL,
    status_code INTEGER,
    response_time_ms INTEGER,
    request_size_bytes INTEGER,
    response_size_bytes INTEGER,
    user_id TEXT,
    ip_address INET,
    user_agent TEXT,
    api_key_id TEXT,
    rate_limit_remaining INTEGER,
    error_message TEXT
);

-- Convert to hypertable with 1-day chunks
SELECT create_hypertable('api_usage_metrics', 'time', chunk_time_interval => INTERVAL '1 day');

-- Create indexes
CREATE INDEX idx_api_usage_endpoint_time ON api_usage_metrics (endpoint, time DESC);
CREATE INDEX idx_api_usage_user_time ON api_usage_metrics (user_id, time DESC);
CREATE INDEX idx_api_usage_status_time ON api_usage_metrics (status_code, time DESC);
CREATE INDEX idx_api_usage_ip_time ON api_usage_metrics (ip_address, time DESC);

-- =====================================================
-- FRAUD DETECTION METRICS
-- =====================================================

-- Real-time fraud detection metrics
CREATE TABLE fraud_detection_metrics (
    time TIMESTAMPTZ NOT NULL,
    transaction_id TEXT,
    user_id TEXT,
    merchant_id TEXT,
    fraud_score DOUBLE PRECISION NOT NULL,
    risk_level TEXT NOT NULL,
    decision TEXT NOT NULL,
    model_name TEXT,
    model_version TEXT,
    processing_time_ms INTEGER,
    feature_count INTEGER,
    confidence_score DOUBLE PRECISION,
    alert_triggered BOOLEAN DEFAULT FALSE,
    alert_severity TEXT
);

-- Convert to hypertable with 1-hour chunks
SELECT create_hypertable('fraud_detection_metrics', 'time', chunk_time_interval => INTERVAL '1 hour');

-- Create indexes
CREATE INDEX idx_fraud_detection_score_time ON fraud_detection_metrics (fraud_score DESC, time DESC);
CREATE INDEX idx_fraud_detection_user_time ON fraud_detection_metrics (user_id, time DESC);
CREATE INDEX idx_fraud_detection_merchant_time ON fraud_detection_metrics (merchant_id, time DESC);
CREATE INDEX idx_fraud_detection_decision_time ON fraud_detection_metrics (decision, time DESC);
CREATE INDEX idx_fraud_detection_model_time ON fraud_detection_metrics (model_name, time DESC);

-- =====================================================
-- ML MODEL PERFORMANCE METRICS
-- =====================================================

-- Model performance tracking
CREATE TABLE ml_model_metrics (
    time TIMESTAMPTZ NOT NULL,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    metric_type TEXT NOT NULL, -- 'accuracy', 'precision', 'recall', 'f1_score', 'auc_roc'
    metric_value DOUBLE PRECISION NOT NULL,
    dataset_size INTEGER,
    training_time_minutes INTEGER,
    inference_time_ms DOUBLE PRECISION,
    memory_usage_mb INTEGER,
    cpu_usage_percent DOUBLE PRECISION,
    gpu_usage_percent DOUBLE PRECISION,
    drift_score DOUBLE PRECISION,
    is_production BOOLEAN DEFAULT FALSE
);

-- Convert to hypertable with 1-day chunks
SELECT create_hypertable('ml_model_metrics', 'time', chunk_time_interval => INTERVAL '1 day');

-- Create indexes
CREATE INDEX idx_ml_model_name_time ON ml_model_metrics (model_name, time DESC);
CREATE INDEX idx_ml_model_type_time ON ml_model_metrics (metric_type, time DESC);
CREATE INDEX idx_ml_model_production_time ON ml_model_metrics (is_production, time DESC);

-- =====================================================
-- TRANSACTION VOLUME METRICS
-- =====================================================

-- Transaction volume and patterns
CREATE TABLE transaction_volume_metrics (
    time TIMESTAMPTZ NOT NULL,
    total_transactions INTEGER DEFAULT 0,
    total_amount DOUBLE PRECISION DEFAULT 0,
    avg_amount DOUBLE PRECISION DEFAULT 0,
    fraud_transactions INTEGER DEFAULT 0,
    fraud_amount DOUBLE PRECISION DEFAULT 0,
    fraud_rate DOUBLE PRECISION DEFAULT 0,
    unique_users INTEGER DEFAULT 0,
    unique_merchants INTEGER DEFAULT 0,
    peak_tps INTEGER DEFAULT 0, -- transactions per second
    avg_processing_time_ms DOUBLE PRECISION DEFAULT 0,
    error_count INTEGER DEFAULT 0,
    timeout_count INTEGER DEFAULT 0
);

-- Convert to hypertable with 5-minute chunks for high-frequency data
SELECT create_hypertable('transaction_volume_metrics', 'time', chunk_time_interval => INTERVAL '5 minutes');

-- Create indexes
CREATE INDEX idx_transaction_volume_time ON transaction_volume_metrics (time DESC);
CREATE INDEX idx_transaction_volume_fraud_rate ON transaction_volume_metrics (fraud_rate DESC, time DESC);

-- =====================================================
-- ALERT METRICS
-- =====================================================

-- Alert and notification metrics
CREATE TABLE alert_metrics (
    time TIMESTAMPTZ NOT NULL,
    alert_id TEXT,
    alert_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    source_system TEXT,
    entity_type TEXT, -- 'user', 'merchant', 'transaction', 'system'
    entity_id TEXT,
    alert_data JSONB,
    is_resolved BOOLEAN DEFAULT FALSE,
    resolution_time_minutes INTEGER,
    false_positive BOOLEAN DEFAULT FALSE,
    escalated BOOLEAN DEFAULT FALSE,
    notification_sent BOOLEAN DEFAULT FALSE,
    notification_channels TEXT[] -- ['email', 'slack', 'webhook']
);

-- Convert to hypertable with 1-hour chunks
SELECT create_hypertable('alert_metrics', 'time', chunk_time_interval => INTERVAL '1 hour');

-- Create indexes
CREATE INDEX idx_alert_type_time ON alert_metrics (alert_type, time DESC);
CREATE INDEX idx_alert_severity_time ON alert_metrics (severity, time DESC);
CREATE INDEX idx_alert_entity_time ON alert_metrics (entity_type, entity_id, time DESC);
CREATE INDEX idx_alert_resolved_time ON alert_metrics (is_resolved, time DESC);

-- =====================================================
-- CONTINUOUS AGGREGATES (Pre-computed Views)
-- =====================================================

-- Hourly fraud detection summary
CREATE MATERIALIZED VIEW hourly_fraud_summary
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 hour', time) AS hour,
    COUNT(*) as total_detections,
    AVG(fraud_score) as avg_fraud_score,
    COUNT(*) FILTER (WHERE decision = 'DECLINED') as declined_count,
    COUNT(*) FILTER (WHERE decision = 'APPROVED') as approved_count,
    COUNT(*) FILTER (WHERE decision = 'REVIEW') as review_count,
    COUNT(DISTINCT user_id) as unique_users,
    COUNT(DISTINCT merchant_id) as unique_merchants,
    AVG(processing_time_ms) as avg_processing_time,
    MAX(processing_time_ms) as max_processing_time
FROM fraud_detection_metrics
GROUP BY hour;

-- Daily API usage summary
CREATE MATERIALIZED VIEW daily_api_summary
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 day', time) AS day,
    endpoint,
    COUNT(*) as request_count,
    AVG(response_time_ms) as avg_response_time,
    MAX(response_time_ms) as max_response_time,
    COUNT(*) FILTER (WHERE status_code >= 400) as error_count,
    COUNT(*) FILTER (WHERE status_code = 200) as success_count,
    COUNT(DISTINCT user_id) as unique_users,
    SUM(request_size_bytes) as total_request_bytes,
    SUM(response_size_bytes) as total_response_bytes
FROM api_usage_metrics
GROUP BY day, endpoint;

-- Weekly model performance summary
CREATE MATERIALIZED VIEW weekly_model_performance
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 week', time) AS week,
    model_name,
    model_version,
    AVG(CASE WHEN metric_type = 'accuracy' THEN metric_value END) as avg_accuracy,
    AVG(CASE WHEN metric_type = 'precision' THEN metric_value END) as avg_precision,
    AVG(CASE WHEN metric_type = 'recall' THEN metric_value END) as avg_recall,
    AVG(CASE WHEN metric_type = 'f1_score' THEN metric_value END) as avg_f1_score,
    AVG(CASE WHEN metric_type = 'auc_roc' THEN metric_value END) as avg_auc_roc,
    AVG(inference_time_ms) as avg_inference_time,
    AVG(drift_score) as avg_drift_score,
    COUNT(*) as evaluation_count
FROM ml_model_metrics
GROUP BY week, model_name, model_version;

-- 5-minute transaction volume rollup
CREATE MATERIALIZED VIEW transaction_volume_5min
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('5 minutes', time) AS bucket,
    SUM(total_transactions) as total_transactions,
    SUM(total_amount) as total_amount,
    AVG(avg_amount) as avg_amount,
    SUM(fraud_transactions) as fraud_transactions,
    SUM(fraud_amount) as fraud_amount,
    AVG(fraud_rate) as avg_fraud_rate,
    MAX(peak_tps) as max_tps,
    AVG(avg_processing_time_ms) as avg_processing_time,
    SUM(error_count) as total_errors
FROM transaction_volume_metrics
GROUP BY bucket;

-- =====================================================
-- REFRESH POLICIES FOR CONTINUOUS AGGREGATES
-- =====================================================

-- Refresh policies (automatic updates)
SELECT add_continuous_aggregate_policy('hourly_fraud_summary',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour');

SELECT add_continuous_aggregate_policy('daily_api_summary',
    start_offset => INTERVAL '2 days',
    end_offset => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 day');

SELECT add_continuous_aggregate_policy('weekly_model_performance',
    start_offset => INTERVAL '2 weeks',
    end_offset => INTERVAL '1 week',
    schedule_interval => INTERVAL '1 day');

SELECT add_continuous_aggregate_policy('transaction_volume_5min',
    start_offset => INTERVAL '1 hour',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '5 minutes');

-- =====================================================
-- DATA RETENTION POLICIES
-- =====================================================

-- Retention policies (automatic cleanup)
SELECT add_retention_policy('system_metrics', INTERVAL '90 days');
SELECT add_retention_policy('api_usage_metrics', INTERVAL '180 days');
SELECT add_retention_policy('fraud_detection_metrics', INTERVAL '365 days');
SELECT add_retention_policy('ml_model_metrics', INTERVAL '730 days'); -- 2 years
SELECT add_retention_policy('transaction_volume_metrics', INTERVAL '30 days');
SELECT add_retention_policy('alert_metrics', INTERVAL '365 days');

-- =====================================================
-- COMPRESSION POLICIES
-- =====================================================

-- Enable compression for older data
SELECT add_compression_policy('system_metrics', INTERVAL '7 days');
SELECT add_compression_policy('api_usage_metrics', INTERVAL '30 days');
SELECT add_compression_policy('fraud_detection_metrics', INTERVAL '30 days');
SELECT add_compression_policy('ml_model_metrics', INTERVAL '90 days');
SELECT add_compression_policy('transaction_volume_metrics', INTERVAL '7 days');
SELECT add_compression_policy('alert_metrics', INTERVAL '30 days');

-- =====================================================
-- UTILITY FUNCTIONS
-- =====================================================

-- Function to get current fraud detection stats
CREATE OR REPLACE FUNCTION get_fraud_stats(
    start_time TIMESTAMPTZ DEFAULT NOW() - INTERVAL '1 hour',
    end_time TIMESTAMPTZ DEFAULT NOW()
)
RETURNS TABLE(
    total_transactions BIGINT,
    fraud_transactions BIGINT,
    fraud_rate NUMERIC,
    avg_fraud_score NUMERIC,
    avg_processing_time NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(*)::BIGINT as total_transactions,
        COUNT(*) FILTER (WHERE decision = 'DECLINED')::BIGINT as fraud_transactions,
        ROUND(
            COUNT(*) FILTER (WHERE decision = 'DECLINED')::NUMERIC * 100.0 / 
            NULLIF(COUNT(*), 0), 2
        ) as fraud_rate,
        ROUND(AVG(fraud_score)::NUMERIC, 4) as avg_fraud_score,
        ROUND(AVG(processing_time_ms)::NUMERIC, 2) as avg_processing_time
    FROM fraud_detection_metrics
    WHERE time BETWEEN start_time AND end_time;
END;
$$ LANGUAGE plpgsql;

-- Function to get API performance stats
CREATE OR REPLACE FUNCTION get_api_stats(
    start_time TIMESTAMPTZ DEFAULT NOW() - INTERVAL '1 hour',
    end_time TIMESTAMPTZ DEFAULT NOW()
)
RETURNS TABLE(
    total_requests BIGINT,
    avg_response_time NUMERIC,
    error_rate NUMERIC,
    requests_per_minute NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(*)::BIGINT as total_requests,
        ROUND(AVG(response_time_ms)::NUMERIC, 2) as avg_response_time,
        ROUND(
            COUNT(*) FILTER (WHERE status_code >= 400)::NUMERIC * 100.0 / 
            NULLIF(COUNT(*), 0), 2
        ) as error_rate,
        ROUND(
            COUNT(*)::NUMERIC / 
            EXTRACT(EPOCH FROM (end_time - start_time)) * 60, 2
        ) as requests_per_minute
    FROM api_usage_metrics
    WHERE time BETWEEN start_time AND end_time;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- SAMPLE DATA FOR TESTING
-- =====================================================

-- Insert sample system metrics
INSERT INTO system_metrics (time, metric_name, metric_value, service_name, instance_id) VALUES
(NOW() - INTERVAL '1 hour', 'cpu_usage_percent', 45.2, 'fraud-api', 'api-001'),
(NOW() - INTERVAL '1 hour', 'memory_usage_percent', 67.8, 'fraud-api', 'api-001'),
(NOW() - INTERVAL '1 hour', 'disk_usage_percent', 23.4, 'fraud-api', 'api-001'),
(NOW() - INTERVAL '30 minutes', 'cpu_usage_percent', 52.1, 'fraud-api', 'api-001'),
(NOW() - INTERVAL '30 minutes', 'memory_usage_percent', 71.2, 'fraud-api', 'api-001'),
(NOW() - INTERVAL '30 minutes', 'disk_usage_percent', 23.5, 'fraud-api', 'api-001');

-- Insert sample API usage metrics
INSERT INTO api_usage_metrics (time, endpoint, method, status_code, response_time_ms, user_id, ip_address) VALUES
(NOW() - INTERVAL '1 hour', '/api/transactions', 'POST', 200, 45, 'user_001', '192.168.1.100'),
(NOW() - INTERVAL '1 hour', '/api/transactions', 'POST', 200, 67, 'user_002', '10.0.0.1'),
(NOW() - INTERVAL '30 minutes', '/api/transactions', 'POST', 200, 34, 'user_001', '192.168.1.100'),
(NOW() - INTERVAL '30 minutes', '/api/health', 'GET', 200, 12, NULL, '172.16.0.1'),
(NOW() - INTERVAL '15 minutes', '/api/transactions', 'POST', 500, 1234, 'user_003', '172.16.0.1');

-- Insert sample fraud detection metrics
INSERT INTO fraud_detection_metrics (time, transaction_id, user_id, merchant_id, fraud_score, risk_level, decision, model_name, processing_time_ms) VALUES
(NOW() - INTERVAL '1 hour', 'tx_001', 'user_001', 'merchant_001', 0.15, 'LOW', 'APPROVED', 'ensemble_v1', 45),
(NOW() - INTERVAL '1 hour', 'tx_002', 'user_002', 'merchant_002', 0.25, 'MEDIUM', 'APPROVED', 'ensemble_v1', 67),
(NOW() - INTERVAL '30 minutes', 'tx_003', 'user_003', 'merchant_003', 0.85, 'HIGH', 'DECLINED', 'ensemble_v1', 123),
(NOW() - INTERVAL '15 minutes', 'tx_004', 'user_001', 'merchant_001', 0.12, 'LOW', 'APPROVED', 'ensemble_v1', 34);

-- Insert sample ML model metrics
INSERT INTO ml_model_metrics (time, model_name, model_version, metric_type, metric_value, inference_time_ms) VALUES
(NOW() - INTERVAL '1 day', 'ensemble_v1', '1.0.0', 'accuracy', 0.945, 45.2),
(NOW() - INTERVAL '1 day', 'ensemble_v1', '1.0.0', 'precision', 0.923, 45.2),
(NOW() - INTERVAL '1 day', 'ensemble_v1', '1.0.0', 'recall', 0.967, 45.2),
(NOW() - INTERVAL '1 day', 'ensemble_v1', '1.0.0', 'f1_score', 0.944, 45.2),
(NOW() - INTERVAL '1 day', 'ensemble_v1', '1.0.0', 'auc_roc', 0.987, 45.2);

-- =====================================================
-- MONITORING VIEWS
-- =====================================================

-- Real-time fraud detection dashboard view
CREATE VIEW v_fraud_dashboard_realtime AS
SELECT 
    DATE_TRUNC('hour', time) as hour,
    COUNT(*) as total_transactions,
    COUNT(*) FILTER (WHERE decision = 'DECLINED') as fraud_count,
    ROUND(AVG(fraud_score), 4) as avg_fraud_score,
    COUNT(DISTINCT user_id) as unique_users,
    COUNT(DISTINCT merchant_id) as unique_merchants,
    ROUND(AVG(processing_time_ms), 2) as avg_processing_time
FROM fraud_detection_metrics
WHERE time > NOW() - INTERVAL '24 hours'
GROUP BY DATE_TRUNC('hour', time)
ORDER BY hour DESC;

-- System health overview
CREATE VIEW v_system_health AS
SELECT 
    service_name,
    instance_id,
    MAX(CASE WHEN metric_name = 'cpu_usage_percent' THEN metric_value END) as cpu_usage,
    MAX(CASE WHEN metric_name = 'memory_usage_percent' THEN metric_value END) as memory_usage,
    MAX(CASE WHEN metric_name = 'disk_usage_percent' THEN metric_value END) as disk_usage,
    MAX(time) as last_update
FROM system_metrics
WHERE time > NOW() - INTERVAL '5 minutes'
GROUP BY service_name, instance_id
ORDER BY service_name, instance_id;

-- Success message
SELECT 'TimescaleDB fraud detection metrics database initialized successfully! 🎉' as status;