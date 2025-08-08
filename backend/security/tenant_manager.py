#!/usr/bin/env python3
"""
Multi-tenant Security Manager
Handles tenant isolation, security policies, and access control
"""

import asyncio
import hashlib
import logging
import os
import secrets
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set

import asyncpg
import bcrypt
import jwt
from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)


class Permission(Enum):
    READ_TRANSACTIONS = "read_transactions"
    WRITE_TRANSACTIONS = "write_transactions"
    READ_USERS = "read_users"
    WRITE_USERS = "write_users"
    READ_MERCHANTS = "read_merchants"
    WRITE_MERCHANTS = "write_merchants"
    ADMIN_ACCESS = "admin_access"
    ML_MODEL_ACCESS = "ml_model_access"
    ANALYTICS_ACCESS = "analytics_access"


@dataclass
class Tenant:
    tenant_id: str
    tenant_name: str
    subscription_tier: str
    resource_limits: Dict[str, Any]
    configuration: Dict[str, Any]
    is_active: bool
    created_at: datetime
    updated_at: datetime


@dataclass
class User:
    user_id: str
    tenant_id: str
    username: str
    email: str
    password_hash: str
    permissions: Set[Permission]
    is_active: bool
    last_login: Optional[datetime]
    created_at: datetime


@dataclass
class Session:
    session_id: str
    user_id: str
    tenant_id: str
    ip_address: str
    user_agent: str
    created_at: datetime
    expires_at: datetime
    is_active: bool


class TenantManager:
    """Manages multi-tenant security and access control"""

    def __init__(self):
        self.pg_dsn = os.getenv(
            "POSTGRES_URL",
            "postgresql://fraud_admin:FraudDetection2024!@localhost:5432/fraud_detection",
        )
        self.jwt_secret = os.getenv("JWT_SECRET", "YourJWTSecretKey2024!")
        self.encryption_key = os.getenv("ENCRYPTION_KEY", Fernet.generate_key())

        if isinstance(self.encryption_key, str):
            self.encryption_key = self.encryption_key.encode()

        self.cipher = Fernet(self.encryption_key)
        self.pool = None

    async def initialize(self):
        """Initialize database connection and security tables"""
        self.pool = await asyncpg.create_pool(
            self.pg_dsn, min_size=5, max_size=20, command_timeout=60
        )

        await self.create_security_tables()
        await self.setup_row_level_security()
        logger.info("Tenant manager initialized")

    async def create_security_tables(self):
        """Create security and tenant management tables"""
        async with self.pool.acquire() as conn:
            # Tenants table
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tenants (
                    tenant_id VARCHAR(100) PRIMARY KEY,
                    tenant_name VARCHAR(255) NOT NULL,
                    subscription_tier VARCHAR(50) DEFAULT 'BASIC',
                    resource_limits JSONB DEFAULT '{}',
                    configuration JSONB DEFAULT '{}',
                    is_active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """
            )

            # Users table for authentication
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS auth_users (
                    user_id VARCHAR(100) PRIMARY KEY,
                    tenant_id VARCHAR(100) REFERENCES tenants(tenant_id),
                    username VARCHAR(100) UNIQUE NOT NULL,
                    email VARCHAR(255) UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    permissions TEXT[] DEFAULT '{}',
                    is_active BOOLEAN DEFAULT TRUE,
                    last_login TIMESTAMP WITH TIME ZONE,
                    failed_login_attempts INTEGER DEFAULT 0,
                    locked_until TIMESTAMP WITH TIME ZONE,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """
            )

            # Sessions table
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_sessions (
                    session_id VARCHAR(100) PRIMARY KEY,
                    user_id VARCHAR(100) REFERENCES auth_users(user_id),
                    tenant_id VARCHAR(100) REFERENCES tenants(tenant_id),
                    ip_address INET,
                    user_agent TEXT,
                    device_fingerprint VARCHAR(255),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    last_activity TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    expires_at TIMESTAMP WITH TIME ZONE,
                    is_active BOOLEAN DEFAULT TRUE
                )
            """
            )

            # Audit trail table
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_trail (
                    id BIGSERIAL PRIMARY KEY,
                    tenant_id VARCHAR(100) REFERENCES tenants(tenant_id),
                    user_id VARCHAR(100),
                    session_id VARCHAR(100),
                    action VARCHAR(100) NOT NULL,
                    resource_type VARCHAR(50) NOT NULL,
                    resource_id VARCHAR(100),
                    old_values JSONB,
                    new_values JSONB,
                    ip_address INET,
                    user_agent TEXT,
                    request_id VARCHAR(100),
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    metadata JSONB DEFAULT '{}'
                )
            """
            )

            # Security events table
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS security_events (
                    id BIGSERIAL PRIMARY KEY,
                    tenant_id VARCHAR(100) REFERENCES tenants(tenant_id),
                    event_type VARCHAR(50) NOT NULL,
                    severity VARCHAR(20) NOT NULL,
                    title VARCHAR(255) NOT NULL,
                    description TEXT NOT NULL,
                    user_id VARCHAR(100),
                    ip_address INET,
                    user_agent TEXT,
                    event_data JSONB DEFAULT '{}',
                    resolved BOOLEAN DEFAULT FALSE,
                    resolved_by VARCHAR(100),
                    resolved_at TIMESTAMP WITH TIME ZONE,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """
            )

            # Create indexes
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_trail_tenant_time ON audit_trail (tenant_id, timestamp DESC)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_security_events_tenant_time ON security_events (tenant_id, created_at DESC)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_user_sessions_active ON user_sessions (user_id, is_active, last_activity DESC)"
            )

    async def setup_row_level_security(self):
        """Set up Row Level Security policies"""
        async with self.pool.acquire() as conn:
            # Enable RLS on main tables
            tables_with_rls = [
                "transactions",
                "users",
                "merchants",
                "fraud_alerts",
                "ml_features",
            ]

            for table in tables_with_rls:
                try:
                    # Add tenant_id column if it doesn't exist
                    await conn.execute(
                        f"""
                        ALTER TABLE {table} 
                        ADD COLUMN IF NOT EXISTS tenant_id VARCHAR(100) 
                        REFERENCES tenants(tenant_id)
                    """
                    )

                    # Enable RLS
                    await conn.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")

                    # Create policy for tenant isolation
                    await conn.execute(
                        f"""
                        DROP POLICY IF EXISTS tenant_isolation ON {table};
                        CREATE POLICY tenant_isolation ON {table}
                        FOR ALL TO application_role
                        USING (tenant_id = current_setting('app.current_tenant_id', true))
                    """
                    )

                except Exception as e:
                    logger.warning(f"Could not set up RLS for {table}: {e}")

    async def create_tenant(self, tenant_data: Dict[str, Any]) -> Tenant:
        """Create a new tenant"""
        tenant = Tenant(
            tenant_id=tenant_data["tenant_id"],
            tenant_name=tenant_data["tenant_name"],
            subscription_tier=tenant_data.get("subscription_tier", "BASIC"),
            resource_limits=tenant_data.get("resource_limits", {}),
            configuration=tenant_data.get("configuration", {}),
            is_active=True,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO tenants (tenant_id, tenant_name, subscription_tier, resource_limits, configuration, is_active, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
                (
                    tenant.tenant_id,
                    tenant.tenant_name,
                    tenant.subscription_tier,
                    tenant.resource_limits,
                    tenant.configuration,
                    tenant.is_active,
                    tenant.created_at,
                    tenant.updated_at,
                ),
            )

        await self.log_audit_event(
            tenant_id=tenant.tenant_id,
            action="CREATE_TENANT",
            resource_type="tenant",
            resource_id=tenant.tenant_id,
            new_values=asdict(tenant),
        )

        logger.info(f"Created tenant: {tenant.tenant_id}")
        return tenant

    async def create_user(self, user_data: Dict[str, Any]) -> User:
        """Create a new user"""
        # Hash password
        password_hash = bcrypt.hashpw(
            user_data["password"].encode("utf-8"), bcrypt.gensalt()
        ).decode("utf-8")

        user = User(
            user_id=user_data["user_id"],
            tenant_id=user_data["tenant_id"],
            username=user_data["username"],
            email=user_data["email"],
            password_hash=password_hash,
            permissions=set(Permission(p) for p in user_data.get("permissions", [])),
            is_active=True,
            last_login=None,
            created_at=datetime.now(timezone.utc),
        )

        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO auth_users (user_id, tenant_id, username, email, password_hash, permissions, is_active, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
                (
                    user.user_id,
                    user.tenant_id,
                    user.username,
                    user.email,
                    user.password_hash,
                    [p.value for p in user.permissions],
                    user.is_active,
                    user.created_at,
                ),
            )

        await self.log_audit_event(
            tenant_id=user.tenant_id,
            action="CREATE_USER",
            resource_type="user",
            resource_id=user.user_id,
            new_values={
                "user_id": user.user_id,
                "username": user.username,
                "email": user.email,
            },
        )

        logger.info(f"Created user: {user.username} for tenant: {user.tenant_id}")
        return user

    async def authenticate_user(
        self, username: str, password: str, ip_address: str, user_agent: str
    ) -> Optional[Dict[str, Any]]:
        """Authenticate user and create session"""
        async with self.pool.acquire() as conn:
            # Get user
            user_row = await conn.fetchrow(
                """
                SELECT user_id, tenant_id, username, email, password_hash, permissions, is_active, failed_login_attempts, locked_until
                FROM auth_users
                WHERE username = $1 OR email = $1
            """,
                username,
            )

            if not user_row:
                await self.log_security_event(
                    event_type="LOGIN_FAILED",
                    severity="MEDIUM",
                    title="Failed login attempt",
                    description=f"Login attempt with unknown username: {username}",
                    ip_address=ip_address,
                    user_agent=user_agent,
                )
                return None

            # Check if account is locked
            if user_row["locked_until"] and user_row["locked_until"] > datetime.now(
                timezone.utc
            ):
                await self.log_security_event(
                    tenant_id=user_row["tenant_id"],
                    event_type="LOGIN_BLOCKED",
                    severity="HIGH",
                    title="Login attempt on locked account",
                    description=f"Login attempt on locked account: {username}",
                    user_id=user_row["user_id"],
                    ip_address=ip_address,
                    user_agent=user_agent,
                )
                return None

            # Verify password
            if not bcrypt.checkpw(
                password.encode("utf-8"), user_row["password_hash"].encode("utf-8")
            ):
                # Increment failed attempts
                failed_attempts = user_row["failed_login_attempts"] + 1
                locked_until = None

                if failed_attempts >= 5:
                    locked_until = datetime.now(timezone.utc) + timedelta(minutes=30)

                await conn.execute(
                    """
                    UPDATE auth_users 
                    SET failed_login_attempts = $1, locked_until = $2
                    WHERE user_id = $3
                """,
                    failed_attempts,
                    locked_until,
                    user_row["user_id"],
                )

                await self.log_security_event(
                    tenant_id=user_row["tenant_id"],
                    event_type="LOGIN_FAILED",
                    severity="MEDIUM",
                    title="Failed login attempt",
                    description=f"Invalid password for user: {username}",
                    user_id=user_row["user_id"],
                    ip_address=ip_address,
                    user_agent=user_agent,
                )
                return None

            # Reset failed attempts on successful login
            await conn.execute(
                """
                UPDATE auth_users 
                SET failed_login_attempts = 0, locked_until = NULL, last_login = NOW()
                WHERE user_id = $1
            """,
                user_row["user_id"],
            )

            # Create session
            session_id = secrets.token_urlsafe(32)
            expires_at = datetime.now(timezone.utc) + timedelta(hours=24)

            await conn.execute(
                """
                INSERT INTO user_sessions (session_id, user_id, tenant_id, ip_address, user_agent, expires_at)
                VALUES ($1, $2, $3, $4, $5, $6)
            """,
                session_id,
                user_row["user_id"],
                user_row["tenant_id"],
                ip_address,
                user_agent,
                expires_at,
            )

            # Generate JWT token
            token_payload = {
                "user_id": user_row["user_id"],
                "tenant_id": user_row["tenant_id"],
                "username": user_row["username"],
                "permissions": user_row["permissions"],
                "session_id": session_id,
                "exp": expires_at.timestamp(),
                "iat": datetime.now(timezone.utc).timestamp(),
            }

            token = jwt.encode(token_payload, self.jwt_secret, algorithm="HS256")

            await self.log_audit_event(
                tenant_id=user_row["tenant_id"],
                user_id=user_row["user_id"],
                session_id=session_id,
                action="LOGIN",
                resource_type="session",
                resource_id=session_id,
                ip_address=ip_address,
                user_agent=user_agent,
            )

            return {
                "token": token,
                "user_id": user_row["user_id"],
                "tenant_id": user_row["tenant_id"],
                "username": user_row["username"],
                "permissions": user_row["permissions"],
                "session_id": session_id,
                "expires_at": expires_at.isoformat(),
            }

    async def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate JWT token and return user info"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])

            # Check if session is still active
            async with self.pool.acquire() as conn:
                session = await conn.fetchrow(
                    """
                    SELECT session_id, user_id, tenant_id, expires_at, is_active
                    FROM user_sessions
                    WHERE session_id = $1 AND is_active = TRUE
                """,
                    payload["session_id"],
                )

                if not session or session["expires_at"] < datetime.now(timezone.utc):
                    return None

                # Update last activity
                await conn.execute(
                    """
                    UPDATE user_sessions 
                    SET last_activity = NOW()
                    WHERE session_id = $1
                """,
                    payload["session_id"],
                )

            return payload

        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

    async def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        return self.cipher.encrypt(data.encode()).decode()

    async def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        return self.cipher.decrypt(encrypted_data.encode()).decode()

    async def log_audit_event(
        self,
        tenant_id: str = None,
        user_id: str = None,
        session_id: str = None,
        action: str = None,
        resource_type: str = None,
        resource_id: str = None,
        old_values: Dict = None,
        new_values: Dict = None,
        ip_address: str = None,
        user_agent: str = None,
        metadata: Dict = None,
    ):
        """Log audit event"""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO audit_trail 
                (tenant_id, user_id, session_id, action, resource_type, resource_id, 
                 old_values, new_values, ip_address, user_agent, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            """,
                (
                    tenant_id,
                    user_id,
                    session_id,
                    action,
                    resource_type,
                    resource_id,
                    old_values,
                    new_values,
                    ip_address,
                    user_agent,
                    metadata or {},
                ),
            )

    async def log_security_event(
        self,
        event_type: str,
        severity: str,
        title: str,
        description: str,
        tenant_id: str = None,
        user_id: str = None,
        ip_address: str = None,
        user_agent: str = None,
        event_data: Dict = None,
    ):
        """Log security event"""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO security_events 
                (tenant_id, event_type, severity, title, description, user_id, ip_address, user_agent, event_data)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """,
                (
                    tenant_id,
                    event_type,
                    severity,
                    title,
                    description,
                    user_id,
                    ip_address,
                    user_agent,
                    event_data or {},
                ),
            )

    async def check_resource_limits(
        self, tenant_id: str, resource_type: str, current_usage: int
    ) -> bool:
        """Check if tenant is within resource limits"""
        async with self.pool.acquire() as conn:
            tenant = await conn.fetchrow(
                """
                SELECT resource_limits FROM tenants WHERE tenant_id = $1
            """,
                tenant_id,
            )

            if not tenant:
                return False

            limits = tenant["resource_limits"]
            limit = limits.get(resource_type)

            if limit is None:
                return True  # No limit set

            return current_usage <= limit

    async def close(self):
        """Close database connection"""
        if self.pool:
            await self.pool.close()


# Example usage
async def main():
    tenant_manager = TenantManager()
    await tenant_manager.initialize()

    try:
        # Create sample tenant
        tenant = await tenant_manager.create_tenant(
            {
                "tenant_id": "tenant_001",
                "tenant_name": "Acme Corp",
                "subscription_tier": "ENTERPRISE",
                "resource_limits": {
                    "max_transactions_per_day": 100000,
                    "max_users": 50,
                    "max_api_calls_per_hour": 10000,
                },
            }
        )

        # Create sample user
        user = await tenant_manager.create_user(
            {
                "user_id": "user_001",
                "tenant_id": "tenant_001",
                "username": "admin",
                "email": "admin@acme.com",
                "password": "SecurePassword123!",
                "permissions": [
                    "admin_access",
                    "read_transactions",
                    "write_transactions",
                ],
            }
        )

        # Test authentication
        auth_result = await tenant_manager.authenticate_user(
            "admin", "SecurePassword123!", "192.168.1.100", "Test Client"
        )

        if auth_result:
            print(f"Authentication successful: {auth_result['username']}")

            # Test token validation
            token_data = await tenant_manager.validate_token(auth_result["token"])
            if token_data:
                print(f"Token valid for user: {token_data['username']}")

        print("✅ Tenant manager test completed successfully!")

    except Exception as e:
        print(f"❌ Tenant manager test failed: {e}")
    finally:
        await tenant_manager.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
