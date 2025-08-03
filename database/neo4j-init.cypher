// =====================================================
// NEO4J FRAUD DETECTION GRAPH SCHEMA
// Initialization script for fraud ring detection
// =====================================================

// Create constraints for unique identifiers
CREATE CONSTRAINT user_id_unique IF NOT EXISTS FOR (u:User) REQUIRE u.user_id IS UNIQUE;
CREATE CONSTRAINT merchant_id_unique IF NOT EXISTS FOR (m:Merchant) REQUIRE m.merchant_id IS UNIQUE;
CREATE CONSTRAINT transaction_id_unique IF NOT EXISTS FOR (t:Transaction) REQUIRE t.transaction_id IS UNIQUE;
CREATE CONSTRAINT device_id_unique IF NOT EXISTS FOR (d:Device) REQUIRE d.device_id IS UNIQUE;
CREATE CONSTRAINT location_id_unique IF NOT EXISTS FOR (l:Location) REQUIRE l.location_id IS UNIQUE;
CREATE CONSTRAINT ip_address_unique IF NOT EXISTS FOR (i:IPAddress) REQUIRE i.ip_address IS UNIQUE;

// Create indexes for performance
CREATE INDEX user_risk_score_idx IF NOT EXISTS FOR (u:User) ON (u.risk_score);
CREATE INDEX merchant_risk_score_idx IF NOT EXISTS FOR (m:Merchant) ON (m.risk_score);
CREATE INDEX transaction_timestamp_idx IF NOT EXISTS FOR (t:Transaction) ON (t.timestamp);
CREATE INDEX transaction_amount_idx IF NOT EXISTS FOR (t:Transaction) ON (t.amount);
CREATE INDEX transaction_fraud_score_idx IF NOT EXISTS FOR (t:Transaction) ON (t.fraud_score);

// =====================================================
// SAMPLE DATA FOR TESTING
// =====================================================

// Create sample users
CREATE (u1:User {
    user_id: 'user_001',
    email: 'john.doe@example.com',
    first_name: 'John',
    last_name: 'Doe',
    age: 35,
    country: 'USA',
    risk_score: 0.2,
    account_created: datetime('2023-01-15T10:30:00Z'),
    total_transactions: 150,
    total_amount: 25000.00,
    is_verified: true
});

CREATE (u2:User {
    user_id: 'user_002',
    email: 'jane.smith@example.com',
    first_name: 'Jane',
    last_name: 'Smith',
    age: 28,
    country: 'GBR',
    risk_score: 0.1,
    account_created: datetime('2023-03-20T14:15:00Z'),
    total_transactions: 89,
    total_amount: 12500.00,
    is_verified: true
});

CREATE (u3:User {
    user_id: 'user_003',
    email: 'suspicious.user@temp.com',
    first_name: 'Bob',
    last_name: 'Suspicious',
    age: 45,
    country: 'UNK',
    risk_score: 0.85,
    account_created: datetime('2024-01-01T00:00:00Z'),
    total_transactions: 25,
    total_amount: 50000.00,
    is_verified: false
});

// Create sample merchants
CREATE (m1:Merchant {
    merchant_id: 'merchant_001',
    business_name: 'TechStore Inc',
    category: 'Electronics',
    mcc: '5732',
    country: 'USA',
    risk_score: 0.1,
    total_transactions: 5000,
    total_amount: 2500000.00,
    fraud_rate: 0.02,
    is_verified: true
});

CREATE (m2:Merchant {
    merchant_id: 'merchant_002',
    business_name: 'Fashion Boutique',
    category: 'Clothing',
    mcc: '5651',
    country: 'GBR',
    risk_score: 0.15,
    total_transactions: 3200,
    total_amount: 800000.00,
    fraud_rate: 0.03,
    is_verified: true
});

CREATE (m3:Merchant {
    merchant_id: 'merchant_003',
    business_name: 'Crypto Exchange',
    category: 'Cryptocurrency',
    mcc: '6051',
    country: 'MLT',
    risk_score: 0.75,
    total_transactions: 1500,
    total_amount: 5000000.00,
    fraud_rate: 0.15,
    is_verified: false
});

// Create sample devices
CREATE (d1:Device {
    device_id: 'device_001',
    device_type: 'mobile',
    os: 'iOS',
    browser: 'Safari',
    fingerprint: 'fp_abc123',
    risk_score: 0.1,
    first_seen: datetime('2023-01-15T10:30:00Z'),
    last_seen: datetime('2024-01-15T16:45:00Z'),
    usage_count: 150
});

CREATE (d2:Device {
    device_id: 'device_002',
    device_type: 'desktop',
    os: 'Windows',
    browser: 'Chrome',
    fingerprint: 'fp_def456',
    risk_score: 0.05,
    first_seen: datetime('2023-03-20T14:15:00Z'),
    last_seen: datetime('2024-01-14T12:30:00Z'),
    usage_count: 89
});

CREATE (d3:Device {
    device_id: 'device_003',
    device_type: 'mobile',
    os: 'Android',
    browser: 'Chrome',
    fingerprint: 'fp_ghi789',
    risk_score: 0.9,
    first_seen: datetime('2024-01-01T00:00:00Z'),
    last_seen: datetime('2024-01-15T23:59:00Z'),
    usage_count: 25
});

// Create sample locations
CREATE (l1:Location {
    location_id: 'loc_001',
    latitude: 40.7128,
    longitude: -74.0060,
    city: 'New York',
    country: 'USA',
    risk_score: 0.1
});

CREATE (l2:Location {
    location_id: 'loc_002',
    latitude: 51.5074,
    longitude: -0.1278,
    city: 'London',
    country: 'GBR',
    risk_score: 0.05
});

CREATE (l3:Location {
    location_id: 'loc_003',
    latitude: 35.6762,
    longitude: 139.6503,
    city: 'Tokyo',
    country: 'JPN',
    risk_score: 0.8
});

// Create sample IP addresses
CREATE (ip1:IPAddress {
    ip_address: '192.168.1.100',
    country: 'USA',
    isp: 'Comcast',
    is_proxy: false,
    is_tor: false,
    risk_score: 0.1
});

CREATE (ip2:IPAddress {
    ip_address: '10.0.0.1',
    country: 'GBR',
    isp: 'BT Group',
    is_proxy: false,
    is_tor: false,
    risk_score: 0.05
});

CREATE (ip3:IPAddress {
    ip_address: '172.16.0.1',
    country: 'UNK',
    isp: 'Unknown',
    is_proxy: true,
    is_tor: true,
    risk_score: 0.95
});

// Create sample transactions
CREATE (t1:Transaction {
    transaction_id: 'tx_001',
    amount: 150.50,
    currency: 'USD',
    timestamp: datetime('2024-01-15T14:30:00Z'),
    fraud_score: 0.15,
    risk_level: 'LOW',
    decision: 'APPROVED',
    processing_time_ms: 45
});

CREATE (t2:Transaction {
    transaction_id: 'tx_002',
    amount: 2500.00,
    currency: 'GBP',
    timestamp: datetime('2024-01-15T16:45:00Z'),
    fraud_score: 0.25,
    risk_level: 'MEDIUM',
    decision: 'APPROVED',
    processing_time_ms: 67
});

CREATE (t3:Transaction {
    transaction_id: 'tx_003',
    amount: 10000.00,
    currency: 'USD',
    timestamp: datetime('2024-01-15T23:15:00Z'),
    fraud_score: 0.85,
    risk_level: 'HIGH',
    decision: 'DECLINED',
    processing_time_ms: 123
});

// =====================================================
// CREATE RELATIONSHIPS
// =====================================================

// User-Transaction relationships
MATCH (u:User {user_id: 'user_001'}), (t:Transaction {transaction_id: 'tx_001'})
CREATE (u)-[:MADE_TRANSACTION {timestamp: t.timestamp, amount: t.amount}]->(t);

MATCH (u:User {user_id: 'user_002'}), (t:Transaction {transaction_id: 'tx_002'})
CREATE (u)-[:MADE_TRANSACTION {timestamp: t.timestamp, amount: t.amount}]->(t);

MATCH (u:User {user_id: 'user_003'}), (t:Transaction {transaction_id: 'tx_003'})
CREATE (u)-[:MADE_TRANSACTION {timestamp: t.timestamp, amount: t.amount}]->(t);

// Transaction-Merchant relationships
MATCH (t:Transaction {transaction_id: 'tx_001'}), (m:Merchant {merchant_id: 'merchant_001'})
CREATE (t)-[:PAID_TO {amount: t.amount, timestamp: t.timestamp}]->(m);

MATCH (t:Transaction {transaction_id: 'tx_002'}), (m:Merchant {merchant_id: 'merchant_002'})
CREATE (t)-[:PAID_TO {amount: t.amount, timestamp: t.timestamp}]->(m);

MATCH (t:Transaction {transaction_id: 'tx_003'}), (m:Merchant {merchant_id: 'merchant_003'})
CREATE (t)-[:PAID_TO {amount: t.amount, timestamp: t.timestamp}]->(m);

// User-Device relationships
MATCH (u:User {user_id: 'user_001'}), (d:Device {device_id: 'device_001'})
CREATE (u)-[:USED_DEVICE {first_used: datetime('2023-01-15T10:30:00Z'), last_used: datetime('2024-01-15T16:45:00Z'), frequency: 150}]->(d);

MATCH (u:User {user_id: 'user_002'}), (d:Device {device_id: 'device_002'})
CREATE (u)-[:USED_DEVICE {first_used: datetime('2023-03-20T14:15:00Z'), last_used: datetime('2024-01-14T12:30:00Z'), frequency: 89}]->(d);

MATCH (u:User {user_id: 'user_003'}), (d:Device {device_id: 'device_003'})
CREATE (u)-[:USED_DEVICE {first_used: datetime('2024-01-01T00:00:00Z'), last_used: datetime('2024-01-15T23:59:00Z'), frequency: 25}]->(d);

// Transaction-Device relationships
MATCH (t:Transaction {transaction_id: 'tx_001'}), (d:Device {device_id: 'device_001'})
CREATE (t)-[:USED_DEVICE {timestamp: t.timestamp}]->(d);

MATCH (t:Transaction {transaction_id: 'tx_002'}), (d:Device {device_id: 'device_002'})
CREATE (t)-[:USED_DEVICE {timestamp: t.timestamp}]->(d);

MATCH (t:Transaction {transaction_id: 'tx_003'}), (d:Device {device_id: 'device_003'})
CREATE (t)-[:USED_DEVICE {timestamp: t.timestamp}]->(d);

// Transaction-Location relationships
MATCH (t:Transaction {transaction_id: 'tx_001'}), (l:Location {location_id: 'loc_001'})
CREATE (t)-[:OCCURRED_AT {timestamp: t.timestamp, confidence: 0.95}]->(l);

MATCH (t:Transaction {transaction_id: 'tx_002'}), (l:Location {location_id: 'loc_002'})
CREATE (t)-[:OCCURRED_AT {timestamp: t.timestamp, confidence: 0.90}]->(l);

MATCH (t:Transaction {transaction_id: 'tx_003'}), (l:Location {location_id: 'loc_003'})
CREATE (t)-[:OCCURRED_AT {timestamp: t.timestamp, confidence: 0.60}]->(l);

// Transaction-IP relationships
MATCH (t:Transaction {transaction_id: 'tx_001'}), (ip:IPAddress {ip_address: '192.168.1.100'})
CREATE (t)-[:ORIGINATED_FROM {timestamp: t.timestamp}]->(ip);

MATCH (t:Transaction {transaction_id: 'tx_002'}), (ip:IPAddress {ip_address: '10.0.0.1'})
CREATE (t)-[:ORIGINATED_FROM {timestamp: t.timestamp}]->(ip);

MATCH (t:Transaction {transaction_id: 'tx_003'}), (ip:IPAddress {ip_address: '172.16.0.1'})
CREATE (t)-[:ORIGINATED_FROM {timestamp: t.timestamp}]->(ip);

// User-Merchant relationships (for repeat customers)
MATCH (u:User {user_id: 'user_001'}), (m:Merchant {merchant_id: 'merchant_001'})
CREATE (u)-[:CUSTOMER_OF {first_transaction: datetime('2023-01-15T10:30:00Z'), total_transactions: 15, total_amount: 2250.00, avg_amount: 150.00}]->(m);

// Suspicious relationships for fraud ring detection
MATCH (u1:User {user_id: 'user_003'}), (u2:User {user_id: 'user_001'})
CREATE (u1)-[:SIMILAR_PATTERN {similarity_score: 0.75, pattern_type: 'device_sharing', detected_at: datetime('2024-01-15T12:00:00Z')}]->(u2);

// =====================================================
// FRAUD DETECTION QUERIES (Examples)
// =====================================================

// Query 1: Find potential fraud rings (users sharing devices)
// MATCH (u1:User)-[:USED_DEVICE]->(d:Device)<-[:USED_DEVICE]-(u2:User)
// WHERE u1.user_id <> u2.user_id AND u1.risk_score > 0.5 AND u2.risk_score > 0.5
// RETURN u1, u2, d, u1.risk_score + u2.risk_score as combined_risk
// ORDER BY combined_risk DESC;

// Query 2: Find users with suspicious transaction patterns
// MATCH (u:User)-[:MADE_TRANSACTION]->(t:Transaction)-[:PAID_TO]->(m:Merchant)
// WHERE t.fraud_score > 0.7 AND m.risk_score > 0.5
// RETURN u.user_id, COUNT(t) as suspicious_transactions, AVG(t.fraud_score) as avg_fraud_score, COLLECT(m.business_name) as merchants
// ORDER BY suspicious_transactions DESC, avg_fraud_score DESC;

// Query 3: Find merchants with high fraud rates
// MATCH (m:Merchant)<-[:PAID_TO]-(t:Transaction)
// WHERE t.fraud_score > 0.6
// WITH m, COUNT(t) as fraud_transactions, AVG(t.fraud_score) as avg_fraud_score
// MATCH (m)<-[:PAID_TO]-(all_t:Transaction)
// WITH m, fraud_transactions, avg_fraud_score, COUNT(all_t) as total_transactions
// RETURN m.business_name, m.category, fraud_transactions, total_transactions, 
//        ROUND(fraud_transactions * 100.0 / total_transactions, 2) as fraud_rate_pct,
//        ROUND(avg_fraud_score, 3) as avg_fraud_score
// ORDER BY fraud_rate_pct DESC;

// Query 4: Community detection for fraud rings
// CALL gds.louvain.stream('fraud-network')
// YIELD nodeId, communityId
// RETURN gds.util.asNode(nodeId).user_id as user_id, communityId
// ORDER BY communityId, user_id;

// Success message
RETURN "Neo4j fraud detection graph schema initialized successfully! 🎉" as status;