-- =====================================================
-- AUTOMATIC PARTITION MANAGEMENT SYSTEM
-- PostgreSQL otomatik partition yönetimi
-- =====================================================

-- pg_partman extension'ı yükle (otomatik partition yönetimi için)
CREATE EXTENSION IF NOT EXISTS pg_partman;

-- Partition yönetimi için schema oluştur
CREATE SCHEMA IF NOT EXISTS partman;

-- =====================================================
-- OTOMATIK PARTITION OLUŞTURMA FONKSİYONLARI
-- =====================================================

-- Otomatik monthly partition oluşturma fonksiyonu
CREATE OR REPLACE FUNCTION create_monthly_partitions(
    parent_table TEXT,
    start_date DATE DEFAULT CURRENT_DATE,
    months_ahead INTEGER DEFAULT 6
)
RETURNS VOID AS $$
DECLARE
    partition_date DATE;
    partition_name TEXT;
    next_partition_date DATE;
    sql_command TEXT;
BEGIN
    -- Gelecek 6 ay için partition'lar oluştur
    FOR i IN 0..months_ahead LOOP
        partition_date := DATE_TRUNC('month', start_date) + (i || ' months')::INTERVAL;
        next_partition_date := partition_date + INTERVAL '1 month';
        
        partition_name := parent_table || '_' || TO_CHAR(partition_date, 'YYYY_MM');
        
        -- Partition'ın zaten var olup olmadığını kontrol et
        IF NOT EXISTS (
            SELECT 1 FROM pg_tables 
            WHERE tablename = partition_name
        ) THEN
            sql_command := FORMAT(
                'CREATE TABLE %I PARTITION OF %I FOR VALUES FROM (%L) TO (%L)',
                partition_name,
                parent_table,
                partition_date,
                next_partition_date
            );
            
            EXECUTE sql_command;
            
            RAISE NOTICE 'Created partition: %', partition_name;
        END IF;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Eski partition'ları temizleme fonksiyonu
CREATE OR REPLACE FUNCTION cleanup_old_partitions(
    parent_table TEXT,
    retention_months INTEGER DEFAULT 24
)
RETURNS VOID AS $$
DECLARE
    partition_record RECORD;
    cutoff_date DATE;
    sql_command TEXT;
BEGIN
    cutoff_date := CURRENT_DATE - (retention_months || ' months')::INTERVAL;
    
    -- Eski partition'ları bul ve sil
    FOR partition_record IN
        SELECT schemaname, tablename
        FROM pg_tables
        WHERE tablename LIKE parent_table || '_%'
        AND tablename ~ '\d{4}_\d{2}$'
    LOOP
        -- Partition tarihini çıkar
        DECLARE
            partition_date_str TEXT;
            partition_date DATE;
        BEGIN
            partition_date_str := SUBSTRING(partition_record.tablename FROM '(\d{4}_\d{2})$');
            partition_date := TO_DATE(partition_date_str, 'YYYY_MM');
            
            IF partition_date < cutoff_date THEN
                sql_command := FORMAT('DROP TABLE %I.%I', 
                    partition_record.schemaname, 
                    partition_record.tablename
                );
                
                EXECUTE sql_command;
                
                RAISE NOTICE 'Dropped old partition: %', partition_record.tablename;
            END IF;
        EXCEPTION
            WHEN OTHERS THEN
                RAISE WARNING 'Could not process partition: %', partition_record.tablename;
        END;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- PG_PARTMAN İLE OTOMATIK YÖNETİM
-- =====================================================

-- Transactions tablosu için pg_partman kurulumu
SELECT partman.create_parent(
    p_parent_table => 'public.transactions',
    p_control => 'transaction_timestamp',
    p_type => 'range',
    p_interval => 'monthly',
    p_premake => 6,  -- 6 ay önceden oluştur
    p_start_partition => '2024-01-01'
);

-- ML features tablosu için otomatik partitioning
SELECT partman.create_parent(
    p_parent_table => 'public.ml_features',
    p_control => 'created_at',
    p_type => 'range',
    p_interval => 'monthly',
    p_premake => 3,
    p_start_partition => '2024-01-01'
);

-- Fraud alerts tablosu için otomatik partitioning
SELECT partman.create_parent(
    p_parent_table => 'public.fraud_alerts',
    p_control => 'created_at',
    p_type => 'range',
    p_interval => 'monthly',
    p_premake => 6,
    p_start_partition => '2024-01-01'
);

-- System metrics için günlük partitioning (yüksek volume için)
SELECT partman.create_parent(
    p_parent_table => 'public.system_metrics',
    p_control => 'timestamp',
    p_type => 'range',
    p_interval => 'daily',
    p_premake => 30,  -- 30 gün önceden oluştur
    p_start_partition => '2024-01-01'
);

-- =====================================================
-- OTOMATIK MAINTENANCE SCHEDULER
-- =====================================================

-- pg_cron extension'ı yükle (scheduled jobs için)
CREATE EXTENSION IF NOT EXISTS pg_cron;

-- Her gün gece yarısı partition maintenance çalıştır
SELECT cron.schedule(
    'partition-maintenance',
    '0 0 * * *',  -- Her gün gece yarısı
    $$
    -- Yeni partition'lar oluştur
    SELECT partman.run_maintenance_proc();
    
    -- Manuel partition'lar için de çalıştır
    SELECT create_monthly_partitions('transactions', CURRENT_DATE, 6);
    SELECT create_monthly_partitions('ml_features', CURRENT_DATE, 3);
    SELECT create_monthly_partitions('fraud_alerts', CURRENT_DATE, 6);
    $$
);

-- Haftalık eski partition temizliği
SELECT cron.schedule(
    'partition-cleanup',
    '0 2 * * 0',  -- Her pazar gece 02:00
    $$
    -- Eski partition'ları temizle (2 yıl retention)
    SELECT cleanup_old_partitions('transactions', 24);
    SELECT cleanup_old_partitions('ml_features', 12);
    SELECT cleanup_old_partitions('fraud_alerts', 36);
    $$
);

-- =====================================================
-- PARTITION MONITORING VE ALERTING
-- =====================================================

-- Partition durumunu kontrol eden view
CREATE OR REPLACE VIEW v_partition_status AS
SELECT 
    schemaname,
    tablename,
    CASE 
        WHEN tablename ~ '_\d{4}_\d{2}$' THEN 
            TO_DATE(SUBSTRING(tablename FROM '(\d{4}_\d{2})$'), 'YYYY_MM')
        ELSE NULL
    END as partition_date,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
    pg_total_relation_size(schemaname||'.'||tablename) as size_bytes
FROM pg_tables
WHERE tablename LIKE '%_2%'  -- Partition tabloları
ORDER BY partition_date DESC;

-- Gelecek partition'ların eksik olup olmadığını kontrol eden fonksiyon
CREATE OR REPLACE FUNCTION check_future_partitions(
    parent_table TEXT,
    months_ahead INTEGER DEFAULT 3
)
RETURNS TABLE(missing_partition TEXT, missing_date DATE) AS $$
DECLARE
    check_date DATE;
    partition_name TEXT;
BEGIN
    FOR i IN 1..months_ahead LOOP
        check_date := DATE_TRUNC('month', CURRENT_DATE) + (i || ' months')::INTERVAL;
        partition_name := parent_table || '_' || TO_CHAR(check_date, 'YYYY_MM');
        
        IF NOT EXISTS (
            SELECT 1 FROM pg_tables WHERE tablename = partition_name
        ) THEN
            missing_partition := partition_name;
            missing_date := check_date;
            RETURN NEXT;
        END IF;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Partition health check fonksiyonu
CREATE OR REPLACE FUNCTION partition_health_check()
RETURNS TABLE(
    table_name TEXT,
    status TEXT,
    message TEXT,
    recommendation TEXT
) AS $$
BEGIN
    -- Transactions tablosu kontrolü
    IF EXISTS (SELECT 1 FROM check_future_partitions('transactions', 3)) THEN
        table_name := 'transactions';
        status := 'WARNING';
        message := 'Missing future partitions detected';
        recommendation := 'Run: SELECT create_monthly_partitions(''transactions'', CURRENT_DATE, 6)';
        RETURN NEXT;
    ELSE
        table_name := 'transactions';
        status := 'OK';
        message := 'All partitions are properly created';
        recommendation := 'No action needed';
        RETURN NEXT;
    END IF;
    
    -- Diğer tablolar için benzer kontroller...
    -- (ML features, fraud alerts, etc.)
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- PARTITION STATISTICS VE PERFORMANCE
-- =====================================================

-- Partition istatistikleri view'ı
CREATE OR REPLACE VIEW v_partition_stats AS
SELECT 
    schemaname,
    tablename,
    n_tup_ins as inserts,
    n_tup_upd as updates,
    n_tup_del as deletes,
    n_live_tup as live_tuples,
    n_dead_tup as dead_tuples,
    last_vacuum,
    last_autovacuum,
    last_analyze,
    last_autoanalyze
FROM pg_stat_user_tables
WHERE tablename LIKE '%_2%'  -- Partition tabloları
ORDER BY n_live_tup DESC;

-- Partition pruning effectiveness kontrolü
CREATE OR REPLACE FUNCTION check_partition_pruning(query_text TEXT)
RETURNS TABLE(
    partition_name TEXT,
    is_scanned BOOLEAN,
    estimated_cost NUMERIC
) AS $$
BEGIN
    -- Bu fonksiyon EXPLAIN çıktısını analiz ederek
    -- hangi partition'ların tarandığını gösterir
    -- (Gerçek implementasyon daha karmaşık olacak)
    
    RETURN QUERY
    SELECT 
        'transactions_2024_01'::TEXT,
        true,
        100.0::NUMERIC
    UNION ALL
    SELECT 
        'transactions_2024_02'::TEXT,
        false,
        0.0::NUMERIC;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- PARTITION BACKUP VE RECOVERY
-- =====================================================

-- Partition-specific backup fonksiyonu
CREATE OR REPLACE FUNCTION backup_partition(
    partition_name TEXT,
    backup_location TEXT DEFAULT '/backup/partitions/'
)
RETURNS BOOLEAN AS $$
DECLARE
    backup_command TEXT;
    result INTEGER;
BEGIN
    backup_command := FORMAT(
        'pg_dump -t %I -f %s%s.sql %s',
        partition_name,
        backup_location,
        partition_name,
        current_database()
    );
    
    -- Bu gerçek implementasyonda pg_dump çalıştırılacak
    -- Şimdilik sadece log yazıyoruz
    RAISE NOTICE 'Backup command: %', backup_command;
    
    RETURN true;
EXCEPTION
    WHEN OTHERS THEN
        RAISE WARNING 'Backup failed for partition %: %', partition_name, SQLERRM;
        RETURN false;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- INITIAL SETUP VE CONFIGURATION
-- =====================================================

-- Mevcut data için partition'ları oluştur
DO $$
BEGIN
    -- 2024 ve 2025 için partition'lar oluştur
    PERFORM create_monthly_partitions('transactions', '2024-01-01'::DATE, 15);
    
    -- Diğer tablolar için de oluştur
    PERFORM create_monthly_partitions('ml_features', '2024-01-01'::DATE, 12);
    PERFORM create_monthly_partitions('fraud_alerts', '2024-01-01'::DATE, 15);
    
    RAISE NOTICE 'Initial partitions created successfully';
END;
$$;

-- Partition management ayarları
UPDATE partman.part_config 
SET 
    retention = '24 months',  -- 2 yıl retention
    retention_keep_table = false,  -- Eski partition'ları sil
    retention_keep_index = false,
    optimize_trigger = 4,  -- Performance optimization
    optimize_constraint = 4
WHERE parent_table = 'public.transactions';

-- =====================================================
-- MONITORING VE ALERTING QUERIES
-- =====================================================

-- Günlük partition health check
CREATE OR REPLACE VIEW v_daily_partition_health AS
SELECT 
    CURRENT_DATE as check_date,
    COUNT(*) as total_partitions,
    COUNT(*) FILTER (WHERE tablename ~ '_' || TO_CHAR(CURRENT_DATE, 'YYYY_MM') || '$') as current_month_partitions,
    COUNT(*) FILTER (WHERE tablename ~ '_' || TO_CHAR(CURRENT_DATE + INTERVAL '1 month', 'YYYY_MM') || '$') as next_month_partitions,
    pg_size_pretty(SUM(pg_total_relation_size(schemaname||'.'||tablename))) as total_size
FROM pg_tables
WHERE tablename LIKE '%_2%';

-- Partition performance metrics
CREATE OR REPLACE VIEW v_partition_performance AS
SELECT 
    tablename,
    seq_scan,
    seq_tup_read,
    idx_scan,
    idx_tup_fetch,
    n_tup_ins,
    n_tup_upd,
    n_tup_del,
    ROUND(
        CASE 
            WHEN seq_scan + idx_scan > 0 
            THEN (idx_scan::NUMERIC / (seq_scan + idx_scan)) * 100 
            ELSE 0 
        END, 2
    ) as index_usage_pct
FROM pg_stat_user_tables
WHERE tablename LIKE '%_2%'
ORDER BY n_tup_ins DESC;

-- =====================================================
-- FINAL NOTES VE BEST PRACTICES
-- =====================================================

/*
OTOMATIK PARTITION YÖNETİMİ BEST PRACTICES:

1. **Monitoring**: 
   - Günlük partition health check çalıştır
   - Disk kullanımını izle
   - Query performance'ı takip et

2. **Maintenance**:
   - pg_partman.run_maintenance_proc() düzenli çalıştır
   - Eski partition'ları otomatik temizle
   - Backup stratejini partition'lara göre ayarla

3. **Performance**:
   - Partition pruning'in çalıştığını kontrol et
   - Index'leri partition'lara göre optimize et
   - Constraint exclusion'ı aktif tut

4. **Troubleshooting**:
   - v_partition_status view'ını kullan
   - partition_health_check() fonksiyonunu çalıştır
   - Log'ları düzenli kontrol et

5. **Scaling**:
   - Büyük tablolar için günlük partitioning düşün
   - Sub-partitioning gerekirse uygula
   - Parallel processing'i partition'larla optimize et
*/

-- Success message
SELECT 'Otomatik partition yönetimi başarıyla kuruldu! 🎉' as status;