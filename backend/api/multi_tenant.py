#!/usr/bin/env python3
"""
Multi-Tenant API Architecture for Fraud Detection System
Tenant isolation, resource management, data segregation, configuration management
"""

import asyncio
import json
import logging
import hashlib
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
import uuid
import threading
from collections import defaultdict
from enum import Enum
import sqlite3
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class TenantTier(Enum):
    """Tenant service tiers"""
    FREE = "free"
    BASIC = "basic"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"

class IsolationLevel(Enum):
    """Data isolation levels"""
    SHARED_DATABASE = "shared_db"
    SHARED_SCHEMA = "shared_schema"
    DEDICATED_SCHEMA = "dedicated_schema"
    DEDICATED_DATABASE = "dedicated_db"

@dataclass
class TenantConfig:
    """Tenant configuration and limits"""
    tenant_id: str
    name: str
    tier: TenantTier
    isolation_level: IsolationLevel
    domain: str
    subdomain: Optional[str]
    
    # Resource limits
    max_api_calls_per_minute: int
    max_api_calls_per_day: int
    max_storage_gb: float
    max_concurrent_connections: int
    max_users: int
    
    # Feature flags
    features_enabled: Set[str]
    
    # Database configuration
    database_config: Dict[str, Any]
    
    # Cache configuration
    cache_config: Dict[str, Any]
    
    # Monitoring configuration
    monitoring_config: Dict[str, Any]
    
    # Custom settings
    custom_settings: Dict[str, Any]
    
    # Metadata
    created_at: datetime
    updated_at: datetime
    created_by: str
    status: str  # active, suspended, deleted
    subscription_expires_at: Optional[datetime]

@dataclass
class TenantUser:
    """User within a tenant"""
    user_id: str
    tenant_id: str
    email: str
    username: str
    roles: Set[str]
    permissions: Set[str]
    is_active: bool
    last_login: Optional[datetime]
    created_at: datetime
    metadata: Dict[str, Any]

@dataclass
class TenantResource:
    """Tenant resource usage tracking"""
    tenant_id: str
    resource_type: str  # api_calls, storage, connections
    current_usage: float
    limit: float
    period: str  # minute, hour, day, month
    last_reset: datetime
    usage_history: List[Dict[str, Any]]

@dataclass
class TenantMetrics:
    """Tenant performance metrics"""
    tenant_id: str
    timestamp: datetime
    api_calls_count: int
    avg_response_time_ms: float
    error_rate: float
    storage_used_gb: float
    active_connections: int
    fraud_detections: int
    false_positives: int
    custom_metrics: Dict[str, Any]

class DatabaseManager:
    """Multi-tenant database management"""
    
    def __init__(self, base_config: Dict[str, Any]):
        self.base_config = base_config
        self.tenant_databases: Dict[str, Dict[str, Any]] = {}
        self.connection_pools: Dict[str, Any] = {}
        
    def setup_tenant_database(self, tenant_config: TenantConfig) -> Dict[str, Any]:
        """Setup database for tenant based on isolation level"""
        
        if tenant_config.isolation_level == IsolationLevel.SHARED_DATABASE:
            return self._setup_shared_database(tenant_config)
        elif tenant_config.isolation_level == IsolationLevel.SHARED_SCHEMA:
            return self._setup_shared_schema(tenant_config)
        elif tenant_config.isolation_level == IsolationLevel.DEDICATED_SCHEMA:
            return self._setup_dedicated_schema(tenant_config)
        elif tenant_config.isolation_level == IsolationLevel.DEDICATED_DATABASE:
            return self._setup_dedicated_database(tenant_config)
        
        raise ValueError(f"Unknown isolation level: {tenant_config.isolation_level}")
    
    def _setup_shared_database(self, tenant_config: TenantConfig) -> Dict[str, Any]:
        """Setup shared database with tenant_id prefix"""
        
        db_config = self.base_config.copy()
        db_config['tenant_id'] = tenant_config.tenant_id
        db_config['table_prefix'] = f"t_{tenant_config.tenant_id}_"
        
        # Use shared connection pool
        db_config['connection_pool'] = 'shared'
        
        return db_config
    
    def _setup_shared_schema(self, tenant_config: TenantConfig) -> Dict[str, Any]:
        """Setup shared database with tenant-specific schema"""
        
        db_config = self.base_config.copy()
        db_config['schema'] = f"tenant_{tenant_config.tenant_id}"
        db_config['connection_pool'] = 'shared'
        
        return db_config
    
    def _setup_dedicated_schema(self, tenant_config: TenantConfig) -> Dict[str, Any]:
        """Setup dedicated schema for tenant"""
        
        db_config = self.base_config.copy()
        db_config['schema'] = f"tenant_{tenant_config.tenant_id}"
        db_config['connection_pool'] = f"tenant_{tenant_config.tenant_id}"
        
        # Create dedicated connection pool
        self._create_connection_pool(tenant_config.tenant_id, db_config)
        
        return db_config
    
    def _setup_dedicated_database(self, tenant_config: TenantConfig) -> Dict[str, Any]:
        """Setup dedicated database for tenant"""
        
        db_config = self.base_config.copy()
        db_config['database'] = f"fraud_detection_tenant_{tenant_config.tenant_id}"
        db_config['connection_pool'] = f"tenant_{tenant_config.tenant_id}"
        
        # Create dedicated connection pool
        self._create_connection_pool(tenant_config.tenant_id, db_config)
        
        return db_config
    
    def _create_connection_pool(self, tenant_id: str, db_config: Dict[str, Any]):
        """Create dedicated connection pool for tenant"""
        
        # Implementation would depend on the database library used
        # This is a placeholder for the actual connection pool creation
        self.connection_pools[tenant_id] = {
            'config': db_config,
            'max_connections': 20,
            'created_at': datetime.now(timezone.utc)
        }
        
        logger.info(f"Created connection pool for tenant {tenant_id}")
    
    def get_tenant_connection(self, tenant_id: str):
        """Get database connection for tenant"""
        
        if tenant_id in self.connection_pools:
            # Return connection from tenant-specific pool
            return self.connection_pools[tenant_id]
        else:
            # Return shared connection
            return self.connection_pools.get('shared')
    
    def execute_tenant_query(self, tenant_id: str, query: str, params: Tuple = None):
        """Execute query with tenant context"""
        
        # Add tenant isolation to query based on configuration
        tenant_config = self.tenant_databases.get(tenant_id)
        if not tenant_config:
            raise ValueError(f"Tenant {tenant_id} not configured")
        
        # Modify query based on isolation level
        if 'table_prefix' in tenant_config:
            # Replace table names with prefixed versions
            # This is a simplified example
            query = self._add_table_prefix(query, tenant_config['table_prefix'])
        
        # Execute query with appropriate connection
        connection = self.get_tenant_connection(tenant_id)
        # Implementation would execute the actual query
        
        logger.debug(f"Executed query for tenant {tenant_id}: {query[:100]}...")
    
    def _add_table_prefix(self, query: str, prefix: str) -> str:
        """Add table prefix to query"""
        # Simplified implementation - would need proper SQL parsing
        tables = ['transactions', 'users', 'merchants', 'devices', 'alerts']
        for table in tables:
            query = query.replace(f" {table} ", f" {prefix}{table} ")
            query = query.replace(f" {table}.", f" {prefix}{table}.")
        return query

class CacheManager:
    """Multi-tenant cache management"""
    
    def __init__(self, base_config: Dict[str, Any]):
        self.base_config = base_config
        self.tenant_caches: Dict[str, Dict[str, Any]] = {}
        
    def setup_tenant_cache(self, tenant_config: TenantConfig) -> Dict[str, Any]:
        """Setup cache for tenant"""
        
        cache_config = tenant_config.cache_config.copy()
        
        # Add tenant isolation
        cache_config['key_prefix'] = f"tenant:{tenant_config.tenant_id}:"
        cache_config['namespace'] = tenant_config.tenant_id
        
        # Set resource limits
        if tenant_config.tier == TenantTier.FREE:
            cache_config['max_memory'] = "100MB"
            cache_config['max_keys'] = 10000
        elif tenant_config.tier == TenantTier.BASIC:
            cache_config['max_memory'] = "500MB"
            cache_config['max_keys'] = 50000
        elif tenant_config.tier == TenantTier.PREMIUM:
            cache_config['max_memory'] = "2GB"
            cache_config['max_keys'] = 200000
        else:  # ENTERPRISE
            cache_config['max_memory'] = "10GB"
            cache_config['max_keys'] = 1000000
        
        self.tenant_caches[tenant_config.tenant_id] = cache_config
        
        logger.info(f"Setup cache for tenant {tenant_config.tenant_id}")
        return cache_config
    
    def get_tenant_cache_key(self, tenant_id: str, key: str) -> str:
        """Get cache key with tenant isolation"""
        config = self.tenant_caches.get(tenant_id, {})
        prefix = config.get('key_prefix', f"tenant:{tenant_id}:")
        return f"{prefix}{key}"
    
    def set_tenant_cache(self, tenant_id: str, key: str, value: Any, ttl: int = 3600):
        """Set cache value for tenant"""
        cache_key = self.get_tenant_cache_key(tenant_id, key)
        # Implementation would set value in cache with TTL
        logger.debug(f"Set cache for tenant {tenant_id}: {key}")
    
    def get_tenant_cache(self, tenant_id: str, key: str) -> Optional[Any]:
        """Get cache value for tenant"""
        cache_key = self.get_tenant_cache_key(tenant_id, key)
        # Implementation would get value from cache
        logger.debug(f"Get cache for tenant {tenant_id}: {key}")
        return None
    
    def delete_tenant_cache(self, tenant_id: str, key: str = None):
        """Delete cache entries for tenant"""
        if key:
            cache_key = self.get_tenant_cache_key(tenant_id, key)
            # Delete specific key
        else:
            # Delete all keys for tenant
            prefix = f"tenant:{tenant_id}:*"
            # Implementation would delete all keys with prefix
        
        logger.info(f"Deleted cache for tenant {tenant_id}")

class ResourceManager:
    """Tenant resource usage management"""
    
    def __init__(self):
        self.tenant_resources: Dict[str, Dict[str, TenantResource]] = defaultdict(dict)
        self.lock = threading.RLock()
    
    def initialize_tenant_resources(self, tenant_config: TenantConfig):
        """Initialize resource tracking for tenant"""
        
        tenant_id = tenant_config.tenant_id
        
        with self.lock:
            # API calls per minute
            self.tenant_resources[tenant_id]['api_calls_minute'] = TenantResource(
                tenant_id=tenant_id,
                resource_type='api_calls',
                current_usage=0,
                limit=tenant_config.max_api_calls_per_minute,
                period='minute',
                last_reset=datetime.now(timezone.utc),
                usage_history=[]
            )
            
            # API calls per day
            self.tenant_resources[tenant_id]['api_calls_day'] = TenantResource(
                tenant_id=tenant_id,
                resource_type='api_calls',
                current_usage=0,
                limit=tenant_config.max_api_calls_per_day,
                period='day',
                last_reset=datetime.now(timezone.utc),
                usage_history=[]
            )
            
            # Storage
            self.tenant_resources[tenant_id]['storage'] = TenantResource(
                tenant_id=tenant_id,
                resource_type='storage',
                current_usage=0,
                limit=tenant_config.max_storage_gb,
                period='month',
                last_reset=datetime.now(timezone.utc),
                usage_history=[]
            )
            
            # Concurrent connections
            self.tenant_resources[tenant_id]['connections'] = TenantResource(
                tenant_id=tenant_id,
                resource_type='connections',
                current_usage=0,
                limit=tenant_config.max_concurrent_connections,
                period='instant',
                last_reset=datetime.now(timezone.utc),
                usage_history=[]
            )
        
        logger.info(f"Initialized resources for tenant {tenant_id}")
    
    def check_resource_limit(self, tenant_id: str, resource_type: str, 
                           amount: float = 1.0) -> Tuple[bool, Dict[str, Any]]:
        """Check if resource usage would exceed limit"""
        
        with self.lock:
            if tenant_id not in self.tenant_resources:
                return False, {'error': 'Tenant not found'}
            
            resources = self.tenant_resources[tenant_id]
            resource_key = f"{resource_type}_minute" if resource_type == 'api_calls' else resource_type
            
            if resource_key not in resources:
                return False, {'error': f'Resource {resource_type} not configured'}
            
            resource = resources[resource_key]
            
            # Reset usage if period has elapsed
            self._reset_resource_if_needed(resource)
            
            # Check if adding amount would exceed limit
            new_usage = resource.current_usage + amount
            
            if new_usage > resource.limit:
                return False, {
                    'error': 'Resource limit exceeded',
                    'current_usage': resource.current_usage,
                    'limit': resource.limit,
                    'period': resource.period
                }
            
            return True, {
                'current_usage': resource.current_usage,
                'limit': resource.limit,
                'remaining': resource.limit - new_usage
            }
    
    def consume_resource(self, tenant_id: str, resource_type: str, amount: float = 1.0):
        """Consume resource amount"""
        
        with self.lock:
            if tenant_id not in self.tenant_resources:
                return
            
            resources = self.tenant_resources[tenant_id]
            
            # Update minute and day counters for API calls
            if resource_type == 'api_calls':
                for period in ['minute', 'day']:
                    resource_key = f"{resource_type}_{period}"
                    if resource_key in resources:
                        resource = resources[resource_key]
                        self._reset_resource_if_needed(resource)
                        resource.current_usage += amount
                        
                        # Record usage history
                        resource.usage_history.append({
                            'timestamp': datetime.now(timezone.utc).isoformat(),
                            'amount': amount,
                            'total_usage': resource.current_usage
                        })
                        
                        # Keep only last 100 entries
                        if len(resource.usage_history) > 100:
                            resource.usage_history = resource.usage_history[-100:]
            
            # Update other resources
            elif resource_type in resources:
                resource = resources[resource_type]
                resource.current_usage += amount
                
                resource.usage_history.append({
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'amount': amount,
                    'total_usage': resource.current_usage
                })
                
                if len(resource.usage_history) > 100:
                    resource.usage_history = resource.usage_history[-100:]
    
    def _reset_resource_if_needed(self, resource: TenantResource):
        """Reset resource usage if period has elapsed"""
        
        now = datetime.now(timezone.utc)
        
        if resource.period == 'minute':
            if (now - resource.last_reset).total_seconds() >= 60:
                resource.current_usage = 0
                resource.last_reset = now
        
        elif resource.period == 'hour':
            if (now - resource.last_reset).total_seconds() >= 3600:
                resource.current_usage = 0
                resource.last_reset = now
        
        elif resource.period == 'day':
            if (now - resource.last_reset).total_seconds() >= 86400:
                resource.current_usage = 0
                resource.last_reset = now
        
        elif resource.period == 'month':
            # Reset on first day of month
            if resource.last_reset.month != now.month or resource.last_reset.year != now.year:
                resource.current_usage = 0
                resource.last_reset = now
    
    def get_tenant_resource_usage(self, tenant_id: str) -> Dict[str, Any]:
        """Get current resource usage for tenant"""
        
        with self.lock:
            if tenant_id not in self.tenant_resources:
                return {}
            
            usage = {}
            for resource_key, resource in self.tenant_resources[tenant_id].items():
                self._reset_resource_if_needed(resource)
                
                usage[resource_key] = {
                    'current_usage': resource.current_usage,
                    'limit': resource.limit,
                    'period': resource.period,
                    'utilization_percent': (resource.current_usage / resource.limit) * 100 if resource.limit > 0 else 0,
                    'last_reset': resource.last_reset.isoformat()
                }
            
            return usage

class FeatureManager:
    """Tenant feature flag management"""
    
    def __init__(self):
        self.tenant_features: Dict[str, Set[str]] = {}
        self.feature_definitions: Dict[str, Dict[str, Any]] = {}
        
        # Define available features
        self._define_features()
    
    def _define_features(self):
        """Define available features and their configurations"""
        
        self.feature_definitions = {
            'basic_fraud_detection': {
                'name': 'Basic Fraud Detection',
                'description': 'Rule-based fraud detection',
                'tiers': [TenantTier.FREE, TenantTier.BASIC, TenantTier.PREMIUM, TenantTier.ENTERPRISE],
                'dependencies': []
            },
            'ml_fraud_detection': {
                'name': 'ML Fraud Detection',
                'description': 'Machine learning-based fraud detection',
                'tiers': [TenantTier.BASIC, TenantTier.PREMIUM, TenantTier.ENTERPRISE],
                'dependencies': ['basic_fraud_detection']
            },
            'real_time_scoring': {
                'name': 'Real-time Scoring',
                'description': 'Real-time transaction scoring',
                'tiers': [TenantTier.PREMIUM, TenantTier.ENTERPRISE],
                'dependencies': ['ml_fraud_detection']
            },
            'advanced_analytics': {
                'name': 'Advanced Analytics',
                'description': 'Advanced fraud analytics and reporting',
                'tiers': [TenantTier.PREMIUM, TenantTier.ENTERPRISE],
                'dependencies': []
            },
            'custom_rules': {
                'name': 'Custom Rules',
                'description': 'Custom fraud detection rules',
                'tiers': [TenantTier.PREMIUM, TenantTier.ENTERPRISE],
                'dependencies': []
            },
            'api_webhooks': {
                'name': 'API Webhooks',
                'description': 'Webhook notifications for events',
                'tiers': [TenantTier.BASIC, TenantTier.PREMIUM, TenantTier.ENTERPRISE],
                'dependencies': []
            },
            'white_label': {
                'name': 'White Label',
                'description': 'White label interface customization',
                'tiers': [TenantTier.ENTERPRISE],
                'dependencies': []
            },
            'sso_integration': {
                'name': 'SSO Integration',
                'description': 'Single sign-on integration',
                'tiers': [TenantTier.ENTERPRISE],
                'dependencies': []
            },
            'dedicated_support': {
                'name': 'Dedicated Support',
                'description': 'Dedicated customer support',
                'tiers': [TenantTier.ENTERPRISE],
                'dependencies': []
            }
        }
    
    def initialize_tenant_features(self, tenant_config: TenantConfig):
        """Initialize features for tenant based on tier"""
        
        tenant_id = tenant_config.tenant_id
        tenant_tier = tenant_config.tier
        
        # Get features available for tier
        available_features = set()
        for feature_name, feature_def in self.feature_definitions.items():
            if tenant_tier in feature_def['tiers']:
                available_features.add(feature_name)
        
        # Add any explicitly enabled features
        available_features.update(tenant_config.features_enabled)
        
        # Remove features that don't meet dependencies
        final_features = set()
        for feature in available_features:
            if self._check_feature_dependencies(feature, available_features):
                final_features.add(feature)
        
        self.tenant_features[tenant_id] = final_features
        
        logger.info(f"Initialized features for tenant {tenant_id}: {final_features}")
    
    def _check_feature_dependencies(self, feature: str, available_features: Set[str]) -> bool:
        """Check if feature dependencies are met"""
        
        if feature not in self.feature_definitions:
            return False
        
        dependencies = self.feature_definitions[feature]['dependencies']
        return all(dep in available_features for dep in dependencies)
    
    def is_feature_enabled(self, tenant_id: str, feature: str) -> bool:
        """Check if feature is enabled for tenant"""
        
        if tenant_id not in self.tenant_features:
            return False
        
        return feature in self.tenant_features[tenant_id]
    
    def enable_feature(self, tenant_id: str, feature: str) -> bool:
        """Enable feature for tenant"""
        
        if tenant_id not in self.tenant_features:
            return False
        
        if feature not in self.feature_definitions:
            return False
        
        # Check dependencies
        if not self._check_feature_dependencies(feature, self.tenant_features[tenant_id]):
            return False
        
        self.tenant_features[tenant_id].add(feature)
        logger.info(f"Enabled feature {feature} for tenant {tenant_id}")
        return True
    
    def disable_feature(self, tenant_id: str, feature: str) -> bool:
        """Disable feature for tenant"""
        
        if tenant_id not in self.tenant_features:
            return False
        
        if feature in self.tenant_features[tenant_id]:
            self.tenant_features[tenant_id].remove(feature)
            logger.info(f"Disabled feature {feature} for tenant {tenant_id}")
            return True
        
        return False
    
    def get_tenant_features(self, tenant_id: str) -> Set[str]:
        """Get all enabled features for tenant"""
        return self.tenant_features.get(tenant_id, set())

class TenantManager:
    """Main multi-tenant management system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Core components
        self.tenants: Dict[str, TenantConfig] = {}
        self.tenant_users: Dict[str, Dict[str, TenantUser]] = defaultdict(dict)
        
        # Managers
        self.database_manager = DatabaseManager(config.get('database', {}))
        self.cache_manager = CacheManager(config.get('cache', {}))
        self.resource_manager = ResourceManager()
        self.feature_manager = FeatureManager()
        
        # Metrics and monitoring
        self.tenant_metrics: Dict[str, List[TenantMetrics]] = defaultdict(list)
        
        # Storage
        self.storage_path = Path(config.get('storage_path', './tenant_data'))
        self.storage_path.mkdir(exist_ok=True)
        
        # Initialize storage
        self._initialize_storage()
        
        # Load existing tenants
        self._load_tenants()
    
    def _initialize_storage(self):
        """Initialize tenant data storage"""
        
        db_path = self.storage_path / 'tenants.db'
        self.db_conn = sqlite3.connect(str(db_path), check_same_thread=False)
        
        # Create tables
        self.db_conn.execute("""
            CREATE TABLE IF NOT EXISTS tenants (
                tenant_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                tier TEXT NOT NULL,
                isolation_level TEXT NOT NULL,
                domain TEXT,
                subdomain TEXT,
                config TEXT NOT NULL,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                status TEXT
            )
        """)
        
        self.db_conn.execute("""
            CREATE TABLE IF NOT EXISTS tenant_users (
                user_id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                email TEXT NOT NULL,
                username TEXT NOT NULL,
                roles TEXT,
                permissions TEXT,
                is_active INTEGER,
                created_at TIMESTAMP,
                metadata TEXT,
                FOREIGN KEY (tenant_id) REFERENCES tenants (tenant_id)
            )
        """)
        
        self.db_conn.commit()
        logger.info("Initialized tenant storage")
    
    def _load_tenants(self):
        """Load existing tenants from storage"""
        
        cursor = self.db_conn.execute("SELECT * FROM tenants WHERE status = 'active'")
        
        for row in cursor.fetchall():
            tenant_data = dict(zip([col[0] for col in cursor.description], row))
            config_data = json.loads(tenant_data['config'])
            
            tenant_config = TenantConfig(
                tenant_id=tenant_data['tenant_id'],
                name=tenant_data['name'],
                tier=TenantTier(tenant_data['tier']),
                isolation_level=IsolationLevel(tenant_data['isolation_level']),
                domain=tenant_data['domain'],
                subdomain=tenant_data['subdomain'],
                created_at=datetime.fromisoformat(tenant_data['created_at']),
                updated_at=datetime.fromisoformat(tenant_data['updated_at']),
                status=tenant_data['status'],
                **config_data
            )
            
            self.tenants[tenant_config.tenant_id] = tenant_config
            
            # Initialize tenant components
            self._initialize_tenant_components(tenant_config)
        
        logger.info(f"Loaded {len(self.tenants)} tenants")
    
    def create_tenant(self, tenant_data: Dict[str, Any]) -> str:
        """Create new tenant"""
        
        tenant_id = str(uuid.uuid4())
        
        # Set defaults based on tier
        tier = TenantTier(tenant_data.get('tier', TenantTier.FREE.value))
        isolation_level = IsolationLevel(tenant_data.get('isolation_level', 
                                                       self._get_default_isolation_level(tier).value))
        
        # Resource limits based on tier
        resource_limits = self._get_tier_resource_limits(tier)
        
        tenant_config = TenantConfig(
            tenant_id=tenant_id,
            name=tenant_data['name'],
            tier=tier,
            isolation_level=isolation_level,
            domain=tenant_data.get('domain'),
            subdomain=tenant_data.get('subdomain'),
            max_api_calls_per_minute=resource_limits['api_calls_per_minute'],
            max_api_calls_per_day=resource_limits['api_calls_per_day'],
            max_storage_gb=resource_limits['storage_gb'],
            max_concurrent_connections=resource_limits['concurrent_connections'],
            max_users=resource_limits['max_users'],
            features_enabled=set(tenant_data.get('features_enabled', [])),
            database_config=tenant_data.get('database_config', {}),
            cache_config=tenant_data.get('cache_config', {}),
            monitoring_config=tenant_data.get('monitoring_config', {}),
            custom_settings=tenant_data.get('custom_settings', {}),
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            created_by=tenant_data.get('created_by', 'system'),
            status='active',
            subscription_expires_at=datetime.fromisoformat(tenant_data['subscription_expires_at']) 
                                   if tenant_data.get('subscription_expires_at') else None
        )
        
        # Store tenant
        self.tenants[tenant_id] = tenant_config
        
        # Save to database
        self._save_tenant(tenant_config)
        
        # Initialize tenant components
        self._initialize_tenant_components(tenant_config)
        
        logger.info(f"Created tenant {tenant_id}: {tenant_config.name}")
        return tenant_id
    
    def _get_default_isolation_level(self, tier: TenantTier) -> IsolationLevel:
        """Get default isolation level for tier"""
        
        if tier == TenantTier.FREE:
            return IsolationLevel.SHARED_DATABASE
        elif tier == TenantTier.BASIC:
            return IsolationLevel.SHARED_SCHEMA
        elif tier == TenantTier.PREMIUM:
            return IsolationLevel.DEDICATED_SCHEMA
        else:  # ENTERPRISE
            return IsolationLevel.DEDICATED_DATABASE
    
    def _get_tier_resource_limits(self, tier: TenantTier) -> Dict[str, int]:
        """Get resource limits for tier"""
        
        limits = {
            TenantTier.FREE: {
                'api_calls_per_minute': 10,
                'api_calls_per_day': 1000,
                'storage_gb': 0.1,
                'concurrent_connections': 5,
                'max_users': 3
            },
            TenantTier.BASIC: {
                'api_calls_per_minute': 100,
                'api_calls_per_day': 10000,
                'storage_gb': 1.0,
                'concurrent_connections': 20,
                'max_users': 10
            },
            TenantTier.PREMIUM: {
                'api_calls_per_minute': 1000,
                'api_calls_per_day': 100000,
                'storage_gb': 10.0,
                'concurrent_connections': 100,
                'max_users': 50
            },
            TenantTier.ENTERPRISE: {
                'api_calls_per_minute': 10000,
                'api_calls_per_day': 1000000,
                'storage_gb': 100.0,
                'concurrent_connections': 500,
                'max_users': 1000
            }
        }
        
        return limits[tier]
    
    def _initialize_tenant_components(self, tenant_config: TenantConfig):
        """Initialize all tenant components"""
        
        # Setup database
        db_config = self.database_manager.setup_tenant_database(tenant_config)
        tenant_config.database_config.update(db_config)
        
        # Setup cache
        cache_config = self.cache_manager.setup_tenant_cache(tenant_config)
        tenant_config.cache_config.update(cache_config)
        
        # Initialize resources
        self.resource_manager.initialize_tenant_resources(tenant_config)
        
        # Initialize features
        self.feature_manager.initialize_tenant_features(tenant_config)
        
        logger.info(f"Initialized components for tenant {tenant_config.tenant_id}")
    
    def _save_tenant(self, tenant_config: TenantConfig):
        """Save tenant to database"""
        
        config_data = {
            'max_api_calls_per_minute': tenant_config.max_api_calls_per_minute,
            'max_api_calls_per_day': tenant_config.max_api_calls_per_day,
            'max_storage_gb': tenant_config.max_storage_gb,
            'max_concurrent_connections': tenant_config.max_concurrent_connections,
            'max_users': tenant_config.max_users,
            'features_enabled': list(tenant_config.features_enabled),
            'database_config': tenant_config.database_config,
            'cache_config': tenant_config.cache_config,
            'monitoring_config': tenant_config.monitoring_config,
            'custom_settings': tenant_config.custom_settings,
            'created_by': tenant_config.created_by,
            'subscription_expires_at': tenant_config.subscription_expires_at.isoformat() 
                                      if tenant_config.subscription_expires_at else None
        }
        
        self.db_conn.execute("""
            INSERT OR REPLACE INTO tenants 
            (tenant_id, name, tier, isolation_level, domain, subdomain, config, 
             created_at, updated_at, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            tenant_config.tenant_id,
            tenant_config.name,
            tenant_config.tier.value,
            tenant_config.isolation_level.value,
            tenant_config.domain,
            tenant_config.subdomain,
            json.dumps(config_data),
            tenant_config.created_at.isoformat(),
            tenant_config.updated_at.isoformat(),
            tenant_config.status
        ))
        
        self.db_conn.commit()
    
    def get_tenant(self, tenant_id: str) -> Optional[TenantConfig]:
        """Get tenant configuration"""
        return self.tenants.get(tenant_id)
    
    def get_tenant_by_domain(self, domain: str) -> Optional[TenantConfig]:
        """Get tenant by domain"""
        for tenant in self.tenants.values():
            if tenant.domain == domain or tenant.subdomain == domain:
                return tenant
        return None
    
    def update_tenant(self, tenant_id: str, updates: Dict[str, Any]) -> bool:
        """Update tenant configuration"""
        
        if tenant_id not in self.tenants:
            return False
        
        tenant_config = self.tenants[tenant_id]
        
        # Update allowed fields
        if 'name' in updates:
            tenant_config.name = updates['name']
        if 'tier' in updates:
            tenant_config.tier = TenantTier(updates['tier'])
        if 'features_enabled' in updates:
            tenant_config.features_enabled = set(updates['features_enabled'])
        if 'custom_settings' in updates:
            tenant_config.custom_settings.update(updates['custom_settings'])
        
        tenant_config.updated_at = datetime.now(timezone.utc)
        
        # Re-initialize components if needed
        if 'tier' in updates or 'features_enabled' in updates:
            self._initialize_tenant_components(tenant_config)
        
        # Save changes
        self._save_tenant(tenant_config)
        
        logger.info(f"Updated tenant {tenant_id}")
        return True
    
    def delete_tenant(self, tenant_id: str) -> bool:
        """Soft delete tenant"""
        
        if tenant_id not in self.tenants:
            return False
        
        # Mark as deleted
        self.tenants[tenant_id].status = 'deleted'
        self.tenants[tenant_id].updated_at = datetime.now(timezone.utc)
        
        # Save changes
        self._save_tenant(self.tenants[tenant_id])
        
        # Remove from memory
        del self.tenants[tenant_id]
        
        logger.info(f"Deleted tenant {tenant_id}")
        return True
    
    def add_tenant_user(self, tenant_id: str, user_data: Dict[str, Any]) -> str:
        """Add user to tenant"""
        
        if tenant_id not in self.tenants:
            raise ValueError(f"Tenant {tenant_id} not found")
        
        user_id = str(uuid.uuid4())
        
        tenant_user = TenantUser(
            user_id=user_id,
            tenant_id=tenant_id,
            email=user_data['email'],
            username=user_data['username'],
            roles=set(user_data.get('roles', ['user'])),
            permissions=set(user_data.get('permissions', [])),
            is_active=user_data.get('is_active', True),
            last_login=None,
            created_at=datetime.now(timezone.utc),
            metadata=user_data.get('metadata', {})
        )
        
        # Check user limit
        tenant_config = self.tenants[tenant_id]
        current_users = len(self.tenant_users[tenant_id])
        
        if current_users >= tenant_config.max_users:
            raise ValueError(f"User limit exceeded for tenant {tenant_id}")
        
        self.tenant_users[tenant_id][user_id] = tenant_user
        
        # Save to database
        self.db_conn.execute("""
            INSERT INTO tenant_users 
            (user_id, tenant_id, email, username, roles, permissions, is_active, created_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            tenant_id,
            tenant_user.email,
            tenant_user.username,
            json.dumps(list(tenant_user.roles)),
            json.dumps(list(tenant_user.permissions)),
            1 if tenant_user.is_active else 0,
            tenant_user.created_at.isoformat(),
            json.dumps(tenant_user.metadata)
        ))
        
        self.db_conn.commit()
        
        logger.info(f"Added user {user_id} to tenant {tenant_id}")
        return user_id
    
    def check_tenant_access(self, tenant_id: str, user_id: str, 
                          required_permission: str = None) -> bool:
        """Check if user has access to tenant"""
        
        if tenant_id not in self.tenant_users:
            return False
        
        if user_id not in self.tenant_users[tenant_id]:
            return False
        
        user = self.tenant_users[tenant_id][user_id]
        
        if not user.is_active:
            return False
        
        if required_permission and required_permission not in user.permissions:
            return False
        
        return True
    
    def record_tenant_metrics(self, tenant_id: str, metrics_data: Dict[str, Any]):
        """Record tenant metrics"""
        
        metrics = TenantMetrics(
            tenant_id=tenant_id,
            timestamp=datetime.now(timezone.utc),
            api_calls_count=metrics_data.get('api_calls_count', 0),
            avg_response_time_ms=metrics_data.get('avg_response_time_ms', 0.0),
            error_rate=metrics_data.get('error_rate', 0.0),
            storage_used_gb=metrics_data.get('storage_used_gb', 0.0),
            active_connections=metrics_data.get('active_connections', 0),
            fraud_detections=metrics_data.get('fraud_detections', 0),
            false_positives=metrics_data.get('false_positives', 0),
            custom_metrics=metrics_data.get('custom_metrics', {})
        )
        
        self.tenant_metrics[tenant_id].append(metrics)
        
        # Keep only last 1000 metrics per tenant
        if len(self.tenant_metrics[tenant_id]) > 1000:
            self.tenant_metrics[tenant_id] = self.tenant_metrics[tenant_id][-1000:]
    
    def get_tenant_metrics(self, tenant_id: str, 
                          start_time: datetime = None,
                          end_time: datetime = None) -> List[TenantMetrics]:
        """Get tenant metrics"""
        
        metrics = self.tenant_metrics.get(tenant_id, [])
        
        if start_time or end_time:
            filtered_metrics = []
            for metric in metrics:
                if start_time and metric.timestamp < start_time:
                    continue
                if end_time and metric.timestamp > end_time:
                    continue
                filtered_metrics.append(metric)
            metrics = filtered_metrics
        
        return metrics
    
    def get_tenant_summary(self, tenant_id: str) -> Dict[str, Any]:
        """Get comprehensive tenant summary"""
        
        if tenant_id not in self.tenants:
            return {}
        
        tenant_config = self.tenants[tenant_id]
        
        # Resource usage
        resource_usage = self.resource_manager.get_tenant_resource_usage(tenant_id)
        
        # Features
        enabled_features = list(self.feature_manager.get_tenant_features(tenant_id))
        
        # Users
        user_count = len(self.tenant_users.get(tenant_id, {}))
        
        # Recent metrics
        recent_metrics = self.tenant_metrics.get(tenant_id, [])[-1:] if tenant_id in self.tenant_metrics else []
        
        return {
            'tenant_id': tenant_id,
            'name': tenant_config.name,
            'tier': tenant_config.tier.value,
            'isolation_level': tenant_config.isolation_level.value,
            'status': tenant_config.status,
            'created_at': tenant_config.created_at.isoformat(),
            'resource_usage': resource_usage,
            'enabled_features': enabled_features,
            'user_count': user_count,
            'max_users': tenant_config.max_users,
            'recent_metrics': [asdict(m) for m in recent_metrics],
            'subscription_expires_at': tenant_config.subscription_expires_at.isoformat() 
                                      if tenant_config.subscription_expires_at else None
        }
    
    def list_tenants(self) -> List[Dict[str, Any]]:
        """List all active tenants"""
        
        tenants = []
        for tenant_id, tenant_config in self.tenants.items():
            if tenant_config.status == 'active':
                tenants.append({
                    'tenant_id': tenant_id,
                    'name': tenant_config.name,
                    'tier': tenant_config.tier.value,
                    'domain': tenant_config.domain,
                    'subdomain': tenant_config.subdomain,
                    'created_at': tenant_config.created_at.isoformat(),
                    'user_count': len(self.tenant_users.get(tenant_id, {}))
                })
        
        return tenants

# =====================================================
# USAGE EXAMPLE
# =====================================================

def main():
    """Example usage of multi-tenant system"""
    
    # Configuration
    config = {
        'database': {
            'host': 'localhost',
            'port': 5432,
            'username': 'postgres',
            'password': 'password'
        },
        'cache': {
            'host': 'localhost',
            'port': 6379
        },
        'storage_path': './tenant_data'
    }
    
    # Initialize tenant manager
    tenant_manager = TenantManager(config)
    
    # Create sample tenant
    tenant_data = {
        'name': 'Acme Corp',
        'tier': 'premium',
        'domain': 'acme.example.com',
        'created_by': 'admin@system.com',
        'subscription_expires_at': '2024-12-31T23:59:59Z'
    }
    
    tenant_id = tenant_manager.create_tenant(tenant_data)
    print(f"Created tenant: {tenant_id}")
    
    # Add user to tenant
    user_data = {
        'email': 'john@acme.com',
        'username': 'john_doe',
        'roles': ['admin', 'user'],
        'permissions': ['read', 'write', 'manage']
    }
    
    user_id = tenant_manager.add_tenant_user(tenant_id, user_data)
    print(f"Added user: {user_id}")
    
    # Check resource limit
    allowed, info = tenant_manager.resource_manager.check_resource_limit(
        tenant_id, 'api_calls', 1.0
    )
    print(f"API call allowed: {allowed}, Info: {info}")
    
    # Consume resource
    if allowed:
        tenant_manager.resource_manager.consume_resource(tenant_id, 'api_calls', 1.0)
    
    # Check feature
    has_ml = tenant_manager.feature_manager.is_feature_enabled(tenant_id, 'ml_fraud_detection')
    print(f"ML fraud detection enabled: {has_ml}")
    
    # Record metrics
    metrics_data = {
        'api_calls_count': 1,
        'avg_response_time_ms': 150.0,
        'error_rate': 0.01,
        'fraud_detections': 2
    }
    tenant_manager.record_tenant_metrics(tenant_id, metrics_data)
    
    # Get tenant summary
    summary = tenant_manager.get_tenant_summary(tenant_id)
    print(f"Tenant summary: {json.dumps(summary, indent=2)}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
