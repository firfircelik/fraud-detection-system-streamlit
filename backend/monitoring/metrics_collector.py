#!/usr/bin/env python3
"""
Real-time Metrics Collection System
Collects and stores metrics in TimescaleDB for monitoring and analytics
"""

import asyncio
import asyncpg
import psutil
import time
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
import os
from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram, generate_latest
import aiohttp
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

@dataclass
class SystemMetric:
    time: datetime
    metric_name: str
    metric_value: float
    labels: Dict[str, str]
    service_name: str
    instance_id: str
    environment: str = 'production'

@dataclass
class APIMetric:
    time: datetime
    endpoint: str
    method: str
    status_code: int
    response_time_ms: int
    request_size_bytes: int
    response_size_bytes: int
    user_id: Optional[str]
    ip_address: Optional[str]
    user_agent: Optional[str]
    api_key_id: Optional[str]
    rate_limit_remaining: Optional[int]
    error_message: Optional[str]

@dataclass
class FraudMetric:
    time: datetime
    transaction_id: str
    user_id: str
    merchant_id: str
    fraud_score: float
    risk_level: str
    decision: str
    model_name: str
    model_version: str
    processing_time_ms: int
    feature_count: int
    confidence_score: float
    alert_triggered: bool
    alert_severity: Optional[str]

class MetricsCollector:
    """Collects and stores real-time metrics"""
    
    def __init__(self):
        self.timescale_dsn = os.getenv('TIMESCALEDB_URL', 'postgresql://timescale_admin:TimeScale2024!@localhost:5433/fraud_metrics')
        self.service_name = os.getenv('SERVICE_NAME', 'fraud-detection')
        self.instance_id = os.getenv('INSTANCE_ID', f'instance-{os.getpid()}')
        self.environment = os.getenv('ENVIRONMENT', 'production')
        
        self.pool = None
        self.registry = CollectorRegistry()
        self.setup_prometheus_metrics()
    
    def setup_prometheus_metrics(self):
        """Set up Prometheus metrics"""
        self.cpu_gauge = Gauge('system_cpu_percent', 'CPU usage percentage', registry=self.registry)
        self.memory_gauge = Gauge('system_memory_percent', 'Memory usage percentage', registry=self.registry)
        self.disk_gauge = Gauge('system_disk_percent', 'Disk usage percentage', registry=self.registry)
        
        self.api_requests_total = Counter('api_requests_total', 'Total API requests', 
                                        ['endpoint', 'method', 'status'], registry=self.registry)
        self.api_request_duration = Histogram('api_request_duration_seconds', 'API request duration',
                                            ['endpoint', 'method'], registry=self.registry)
        
        self.fraud_detections_total = Counter('fraud_detections_total', 'Total fraud detections',
                                            ['decision', 'risk_level'], registry=self.registry)
        self.fraud_score_gauge = Gauge('fraud_score_current', 'Current fraud score', registry=self.registry)
    
    async def initialize(self):
        """Initialize database connection"""
        self.pool = await asyncpg.create_pool(
            self.timescale_dsn,
            min_size=5,
            max_size=20,
            command_timeout=60
        )
        logger.info("Metrics collector initialized")
    
    async def collect_system_metrics(self) -> List[SystemMetric]:
        """Collect system performance metrics"""
        now = datetime.now(timezone.utc)
        metrics = []
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_per_core = psutil.cpu_percent(interval=1, percpu=True)
        
        metrics.append(SystemMetric(
            time=now,
            metric_name='cpu_usage_percent',
            metric_value=cpu_percent,
            labels={'type': 'total'},
            service_name=self.service_name,
            instance_id=self.instance_id,
            environment=self.environment
        ))
        
        for i, core_usage in enumerate(cpu_per_core):
            metrics.append(SystemMetric(
                time=now,
                metric_name='cpu_usage_percent',
                metric_value=core_usage,
                labels={'type': 'core', 'core_id': str(i)},
                service_name=self.service_name,
                instance_id=self.instance_id,
                environment=self.environment
            ))
        
        # Memory metrics
        memory = psutil.virtual_memory()
        metrics.extend([
            SystemMetric(
                time=now,
                metric_name='memory_usage_percent',
                metric_value=memory.percent,
                labels={'type': 'virtual'},
                service_name=self.service_name,
                instance_id=self.instance_id,
                environment=self.environment
            ),
            SystemMetric(
                time=now,
                metric_name='memory_usage_bytes',
                metric_value=memory.used,
                labels={'type': 'used'},
                service_name=self.service_name,
                instance_id=self.instance_id,
                environment=self.environment
            ),
            SystemMetric(
                time=now,
                metric_name='memory_usage_bytes',
                metric_value=memory.available,
                labels={'type': 'available'},
                service_name=self.service_name,
                instance_id=self.instance_id,
                environment=self.environment
            )
        ])
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        metrics.extend([
            SystemMetric(
                time=now,
                metric_name='disk_usage_percent',
                metric_value=(disk.used / disk.total) * 100,
                labels={'mount': '/'},
                service_name=self.service_name,
                instance_id=self.instance_id,
                environment=self.environment
            ),
            SystemMetric(
                time=now,
                metric_name='disk_usage_bytes',
                metric_value=disk.used,
                labels={'mount': '/', 'type': 'used'},
                service_name=self.service_name,
                instance_id=self.instance_id,
                environment=self.environment
            )
        ])
        
        # Network metrics
        network = psutil.net_io_counters()
        metrics.extend([
            SystemMetric(
                time=now,
                metric_name='network_bytes_sent',
                metric_value=network.bytes_sent,
                labels={'direction': 'sent'},
                service_name=self.service_name,
                instance_id=self.instance_id,
                environment=self.environment
            ),
            SystemMetric(
                time=now,
                metric_name='network_bytes_recv',
                metric_value=network.bytes_recv,
                labels={'direction': 'received'},
                service_name=self.service_name,
                instance_id=self.instance_id,
                environment=self.environment
            )
        ])
        
        # Update Prometheus metrics
        self.cpu_gauge.set(cpu_percent)
        self.memory_gauge.set(memory.percent)
        self.disk_gauge.set((disk.used / disk.total) * 100)
        
        return metrics
    
    async def store_system_metrics(self, metrics: List[SystemMetric]):
        """Store system metrics in TimescaleDB"""
        async with self.pool.acquire() as conn:
            values = []
            for metric in metrics:
                values.append((
                    metric.time,
                    metric.metric_name,
                    metric.metric_value,
                    json.dumps(metric.labels),
                    metric.service_name,
                    metric.instance_id,
                    metric.environment
                ))
            
            await conn.executemany("""
                INSERT INTO system_metrics 
                (time, metric_name, metric_value, labels, service_name, instance_id, environment)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
            """, values)
    
    async def store_api_metric(self, metric: APIMetric):
        """Store API usage metric"""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO api_usage_metrics 
                (time, endpoint, method, status_code, response_time_ms, request_size_bytes,
                 response_size_bytes, user_id, ip_address, user_agent, api_key_id,
                 rate_limit_remaining, error_message)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            """, (
                metric.time, metric.endpoint, metric.method, metric.status_code,
                metric.response_time_ms, metric.request_size_bytes, metric.response_size_bytes,
                metric.user_id, metric.ip_address, metric.user_agent, metric.api_key_id,
                metric.rate_limit_remaining, metric.error_message
            ))
        
        # Update Prometheus metrics
        self.api_requests_total.labels(
            endpoint=metric.endpoint,
            method=metric.method,
            status=str(metric.status_code)
        ).inc()
        
        self.api_request_duration.labels(
            endpoint=metric.endpoint,
            method=metric.method
        ).observe(metric.response_time_ms / 1000.0)
    
    async def store_fraud_metric(self, metric: FraudMetric):
        """Store fraud detection metric"""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO fraud_detection_metrics 
                (time, transaction_id, user_id, merchant_id, fraud_score, risk_level,
                 decision, model_name, model_version, processing_time_ms, feature_count,
                 confidence_score, alert_triggered, alert_severity)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
            """, (
                metric.time, metric.transaction_id, metric.user_id, metric.merchant_id,
                metric.fraud_score, metric.risk_level, metric.decision, metric.model_name,
                metric.model_version, metric.processing_time_ms, metric.feature_count,
                metric.confidence_score, metric.alert_triggered, metric.alert_severity
            ))
        
        # Update Prometheus metrics
        self.fraud_detections_total.labels(
            decision=metric.decision,
            risk_level=metric.risk_level
        ).inc()
        
        self.fraud_score_gauge.set(metric.fraud_score)
    
    async def collect_database_metrics(self) -> List[SystemMetric]:
        """Collect database performance metrics"""
        now = datetime.now(timezone.utc)
        metrics = []
        
        try:
            async with self.pool.acquire() as conn:
                # Connection count
                result = await conn.fetchrow("SELECT COUNT(*) as active_connections FROM pg_stat_activity WHERE state = 'active'")
                metrics.append(SystemMetric(
                    time=now,
                    metric_name='database_active_connections',
                    metric_value=result['active_connections'],
                    labels={'database': 'timescaledb'},
                    service_name='timescaledb',
                    instance_id=self.instance_id,
                    environment=self.environment
                ))
                
                # Cache hit ratio
                result = await conn.fetchrow("""
                    SELECT ROUND(100.0 * sum(blks_hit) / (sum(blks_hit) + sum(blks_read)), 2) as cache_hit_ratio
                    FROM pg_stat_database WHERE datname = current_database()
                """)
                if result['cache_hit_ratio']:
                    metrics.append(SystemMetric(
                        time=now,
                        metric_name='database_cache_hit_ratio',
                        metric_value=result['cache_hit_ratio'],
                        labels={'database': 'timescaledb'},
                        service_name='timescaledb',
                        instance_id=self.instance_id,
                        environment=self.environment
                    ))
                
                # Table sizes
                tables = ['system_metrics', 'api_usage_metrics', 'fraud_detection_metrics']
                for table in tables:
                    result = await conn.fetchrow(f"SELECT pg_total_relation_size('{table}') as size_bytes")
                    metrics.append(SystemMetric(
                        time=now,
                        metric_name='database_table_size_bytes',
                        metric_value=result['size_bytes'],
                        labels={'table': table, 'database': 'timescaledb'},
                        service_name='timescaledb',
                        instance_id=self.instance_id,
                        environment=self.environment
                    ))
        
        except Exception as e:
            logger.error(f"Failed to collect database metrics: {e}")
        
        return metrics
    
    async def get_prometheus_metrics(self) -> str:
        """Get Prometheus formatted metrics"""
        return generate_latest(self.registry).decode('utf-8')
    
    async def run_collection_loop(self):
        """Run continuous metrics collection"""
        while True:
            try:
                # Collect system metrics
                system_metrics = await self.collect_system_metrics()
                await self.store_system_metrics(system_metrics)
                
                # Collect database metrics
                db_metrics = await self.collect_database_metrics()
                await self.store_system_metrics(db_metrics)
                
                logger.info(f"Collected {len(system_metrics + db_metrics)} metrics")
                
                # Wait before next collection
                await asyncio.sleep(30)  # Collect every 30 seconds
                
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def close(self):
        """Close database connection"""
        if self.pool:
            await self.pool.close()

class APIMetricsMiddleware:
    """Middleware to collect API metrics"""
    
    def __init__(self, collector: MetricsCollector):
        self.collector = collector
    
    async def __call__(self, request, handler):
        start_time = time.time()
        
        try:
            response = await handler(request)
            
            # Calculate metrics
            processing_time = int((time.time() - start_time) * 1000)
            
            # Create metric
            metric = APIMetric(
                time=datetime.now(timezone.utc),
                endpoint=request.path,
                method=request.method,
                status_code=response.status,
                response_time_ms=processing_time,
                request_size_bytes=len(await request.read()) if hasattr(request, 'read') else 0,
                response_size_bytes=len(response.body) if hasattr(response, 'body') else 0,
                user_id=request.headers.get('X-User-ID'),
                ip_address=request.remote,
                user_agent=request.headers.get('User-Agent'),
                api_key_id=request.headers.get('X-API-Key-ID'),
                rate_limit_remaining=response.headers.get('X-RateLimit-Remaining'),
                error_message=None
            )
            
            # Store metric
            await self.collector.store_api_metric(metric)
            
            return response
            
        except Exception as e:
            # Calculate metrics for error case
            processing_time = int((time.time() - start_time) * 1000)
            
            metric = APIMetric(
                time=datetime.now(timezone.utc),
                endpoint=request.path,
                method=request.method,
                status_code=500,
                response_time_ms=processing_time,
                request_size_bytes=0,
                response_size_bytes=0,
                user_id=request.headers.get('X-User-ID'),
                ip_address=request.remote,
                user_agent=request.headers.get('User-Agent'),
                api_key_id=request.headers.get('X-API-Key-ID'),
                rate_limit_remaining=None,
                error_message=str(e)
            )
            
            await self.collector.store_api_metric(metric)
            raise

# Standalone metrics collection service
async def main():
    collector = MetricsCollector()
    await collector.initialize()
    
    try:
        await collector.run_collection_loop()
    except KeyboardInterrupt:
        logger.info("Shutting down metrics collector")
    finally:
        await collector.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())