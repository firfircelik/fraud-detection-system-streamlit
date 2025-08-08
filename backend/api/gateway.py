#!/usr/bin/env python3
"""
Advanced API Gateway for Fraud Detection System
Rate limiting, authentication, request/response transformation, API versioning
"""

import asyncio
import hashlib
import hmac
import json
import logging
import os
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import aioredis
import jwt
import redis
import uvicorn
import yaml
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from prometheus_client import Counter, Gauge, Histogram, generate_latest
from starlette.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)


@dataclass
class RateLimitRule:
    """Rate limiting rule configuration"""

    rule_id: str
    name: str
    requests_per_minute: int
    requests_per_hour: int
    requests_per_day: int
    burst_limit: int
    scope: str  # global, user, api_key, ip
    endpoints: List[str]  # specific endpoints or ['*'] for all
    priority: int
    active: bool
    created_at: datetime


@dataclass
class APIKey:
    """API key configuration"""

    key_id: str
    api_key: str
    name: str
    organization: str
    scopes: List[str]
    rate_limits: Dict[str, int]
    allowed_ips: List[str]
    webhook_url: Optional[str]
    active: bool
    expires_at: Optional[datetime]
    created_at: datetime
    last_used: Optional[datetime]


@dataclass
class RequestLog:
    """API request log entry"""

    request_id: str
    timestamp: datetime
    method: str
    endpoint: str
    client_ip: str
    user_agent: str
    api_key_id: Optional[str]
    user_id: Optional[str]
    request_size: int
    response_size: int
    response_time_ms: float
    status_code: int
    error_message: Optional[str]
    rate_limited: bool


@dataclass
class TransformationRule:
    """Request/response transformation rule"""

    rule_id: str
    name: str
    endpoint_pattern: str
    transform_type: str  # request, response, both
    transformations: Dict[str, Any]
    conditions: Dict[str, Any]
    active: bool
    created_at: datetime


class RateLimiter:
    """Advanced rate limiting with multiple algorithms"""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.rules: Dict[str, RateLimitRule] = {}

        # Rate limiting algorithms
        self.algorithms = {
            "token_bucket": self._token_bucket_check,
            "sliding_window": self._sliding_window_check,
            "fixed_window": self._fixed_window_check,
            "leaky_bucket": self._leaky_bucket_check,
        }

    def add_rule(self, rule: RateLimitRule):
        """Add rate limiting rule"""
        self.rules[rule.rule_id] = rule
        logger.info(f"Added rate limit rule: {rule.name}")

    async def check_rate_limit(
        self, key: str, rule_id: str, algorithm: str = "sliding_window"
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check rate limit for key"""

        if rule_id not in self.rules:
            return True, {}

        rule = self.rules[rule_id]
        if not rule.active:
            return True, {}

        # Use specified algorithm
        if algorithm in self.algorithms:
            return await self.algorithms[algorithm](key, rule)

        # Default to sliding window
        return await self._sliding_window_check(key, rule)

    async def _sliding_window_check(
        self, key: str, rule: RateLimitRule
    ) -> Tuple[bool, Dict[str, Any]]:
        """Sliding window rate limiting"""

        now = time.time()
        window_start = now - 60  # 1 minute window

        pipe = self.redis.pipeline()

        # Remove old entries
        pipe.zremrangebyscore(f"rate_limit:{key}", 0, window_start)

        # Count current requests
        pipe.zcard(f"rate_limit:{key}")

        # Add current request
        pipe.zadd(f"rate_limit:{key}", {str(uuid.uuid4()): now})

        # Set expiration
        pipe.expire(f"rate_limit:{key}", 3600)  # 1 hour TTL

        results = pipe.execute()
        current_count = results[1]

        # Check limit
        allowed = current_count < rule.requests_per_minute

        # Calculate reset time
        reset_time = int(now + 60)

        return allowed, {
            "requests_remaining": max(0, rule.requests_per_minute - current_count - 1),
            "requests_limit": rule.requests_per_minute,
            "reset_time": reset_time,
        }

    async def _token_bucket_check(
        self, key: str, rule: RateLimitRule
    ) -> Tuple[bool, Dict[str, Any]]:
        """Token bucket rate limiting"""

        bucket_key = f"token_bucket:{key}"
        now = time.time()

        # Get current bucket state
        bucket_data = self.redis.hmget(bucket_key, ["tokens", "last_refill"])

        if bucket_data[0] is None:
            # Initialize bucket
            tokens = rule.burst_limit
            last_refill = now
        else:
            tokens = float(bucket_data[0])
            last_refill = float(bucket_data[1])

        # Calculate tokens to add
        time_passed = now - last_refill
        tokens_to_add = time_passed * (rule.requests_per_minute / 60.0)
        tokens = min(rule.burst_limit, tokens + tokens_to_add)

        # Check if request is allowed
        if tokens >= 1.0:
            tokens -= 1.0
            allowed = True
        else:
            allowed = False

        # Update bucket
        pipe = self.redis.pipeline()
        pipe.hmset(bucket_key, {"tokens": tokens, "last_refill": now})
        pipe.expire(bucket_key, 3600)
        pipe.execute()

        return allowed, {
            "tokens_remaining": int(tokens),
            "burst_limit": rule.burst_limit,
            "refill_rate": rule.requests_per_minute,
        }

    async def _fixed_window_check(
        self, key: str, rule: RateLimitRule
    ) -> Tuple[bool, Dict[str, Any]]:
        """Fixed window rate limiting"""

        now = time.time()
        window = int(now // 60)  # 1-minute windows
        window_key = f"fixed_window:{key}:{window}"

        # Increment counter
        current_count = self.redis.incr(window_key)

        if current_count == 1:
            self.redis.expire(window_key, 60)

        allowed = current_count <= rule.requests_per_minute

        return allowed, {
            "requests_remaining": max(0, rule.requests_per_minute - current_count),
            "requests_limit": rule.requests_per_minute,
            "window_reset": (window + 1) * 60,
        }

    async def _leaky_bucket_check(
        self, key: str, rule: RateLimitRule
    ) -> Tuple[bool, Dict[str, Any]]:
        """Leaky bucket rate limiting"""

        bucket_key = f"leaky_bucket:{key}"
        now = time.time()

        # Get current bucket state
        bucket_data = self.redis.hmget(bucket_key, ["level", "last_leak"])

        if bucket_data[0] is None:
            level = 0.0
            last_leak = now
        else:
            level = float(bucket_data[0])
            last_leak = float(bucket_data[1])

        # Calculate leakage
        time_passed = now - last_leak
        leak_amount = time_passed * (rule.requests_per_minute / 60.0)
        level = max(0.0, level - leak_amount)

        # Check if request fits in bucket
        if level < rule.burst_limit:
            level += 1.0
            allowed = True
        else:
            allowed = False

        # Update bucket
        pipe = self.redis.pipeline()
        pipe.hmset(bucket_key, {"level": level, "last_leak": now})
        pipe.expire(bucket_key, 3600)
        pipe.execute()

        return allowed, {
            "bucket_level": int(level),
            "bucket_capacity": rule.burst_limit,
            "leak_rate": rule.requests_per_minute,
        }


class AuthenticationManager:
    """Advanced authentication and authorization"""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.api_keys: Dict[str, APIKey] = {}
        self.jwt_secret = os.getenv("JWT_SECRET", "fallback-secret-key-change-in-production")
        self.jwt_algorithm = "HS256"

        # Load API keys from storage
        self._load_api_keys()

    def create_api_key(self, key_data: Dict[str, Any]) -> str:
        """Create new API key"""

        key_id = str(uuid.uuid4())
        api_key = self._generate_api_key()

        api_key_obj = APIKey(
            key_id=key_id,
            api_key=api_key,
            name=key_data["name"],
            organization=key_data["organization"],
            scopes=key_data.get("scopes", ["read"]),
            rate_limits=key_data.get("rate_limits", {}),
            allowed_ips=key_data.get("allowed_ips", []),
            webhook_url=key_data.get("webhook_url"),
            active=True,
            expires_at=(
                datetime.fromisoformat(key_data["expires_at"])
                if key_data.get("expires_at")
                else None
            ),
            created_at=datetime.now(timezone.utc),
            last_used=None,
        )

        self.api_keys[api_key] = api_key_obj

        # Store in Redis
        self.redis.hset(
            "api_keys", api_key, json.dumps(asdict(api_key_obj), default=str)
        )

        logger.info(f"Created API key for {key_data['organization']}: {key_id}")
        return api_key

    def _generate_api_key(self) -> str:
        """Generate secure API key"""
        # Generate random bytes and create hex string
        random_bytes = uuid.uuid4().bytes + uuid.uuid4().bytes
        return hashlib.sha256(random_bytes).hexdigest()

    def _load_api_keys(self):
        """Load API keys from Redis"""
        try:
            api_keys_data = self.redis.hgetall("api_keys")
            for api_key, data in api_keys_data.items():
                key_data = json.loads(data)
                # Convert string timestamps back to datetime
                key_data["created_at"] = datetime.fromisoformat(key_data["created_at"])
                if key_data["expires_at"]:
                    key_data["expires_at"] = datetime.fromisoformat(
                        key_data["expires_at"]
                    )
                if key_data["last_used"]:
                    key_data["last_used"] = datetime.fromisoformat(
                        key_data["last_used"]
                    )

                api_key_obj = APIKey(**key_data)
                self.api_keys[
                    api_key.decode() if isinstance(api_key, bytes) else api_key
                ] = api_key_obj

        except Exception as e:
            logger.warning(f"Failed to load API keys: {e}")

    async def authenticate_api_key(
        self, api_key: str, client_ip: str
    ) -> Optional[APIKey]:
        """Authenticate API key"""

        if api_key not in self.api_keys:
            return None

        key_obj = self.api_keys[api_key]

        # Check if key is active
        if not key_obj.active:
            return None

        # Check expiration
        if key_obj.expires_at and datetime.now(timezone.utc) > key_obj.expires_at:
            return None

        # Check IP restrictions
        if key_obj.allowed_ips and client_ip not in key_obj.allowed_ips:
            return None

        # Update last used timestamp
        key_obj.last_used = datetime.now(timezone.utc)

        return key_obj

    def create_jwt_token(
        self, user_id: str, scopes: List[str], expires_in: int = 3600
    ) -> str:
        """Create JWT token"""

        payload = {
            "user_id": user_id,
            "scopes": scopes,
            "iat": datetime.now(timezone.utc),
            "exp": datetime.now(timezone.utc) + timedelta(seconds=expires_in),
            "iss": "fraud-detection-api",
        }

        return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)

    def verify_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token"""

        try:
            payload = jwt.decode(
                token, self.jwt_secret, algorithms=[self.jwt_algorithm]
            )
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid JWT token")
            return None

    def check_scope_permission(
        self, required_scope: str, user_scopes: List[str]
    ) -> bool:
        """Check if user has required scope"""

        # Admin scope allows everything
        if "admin" in user_scopes:
            return True

        # Check specific scope
        return required_scope in user_scopes


class RequestTransformer:
    """Request/response transformation engine"""

    def __init__(self):
        self.rules: Dict[str, TransformationRule] = {}

    def add_transformation_rule(self, rule: TransformationRule):
        """Add transformation rule"""
        self.rules[rule.rule_id] = rule
        logger.info(f"Added transformation rule: {rule.name}")

    def transform_request(
        self, endpoint: str, request_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Transform incoming request"""

        transformed_data = request_data.copy()

        for rule in self.rules.values():
            if (
                rule.active
                and rule.transform_type in ["request", "both"]
                and self._matches_endpoint(endpoint, rule.endpoint_pattern)
            ):

                # Check conditions
                if self._check_conditions(request_data, rule.conditions):
                    transformed_data = self._apply_transformations(
                        transformed_data, rule.transformations
                    )

        return transformed_data

    def transform_response(
        self, endpoint: str, response_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Transform outgoing response"""

        transformed_data = response_data.copy()

        for rule in self.rules.values():
            if (
                rule.active
                and rule.transform_type in ["response", "both"]
                and self._matches_endpoint(endpoint, rule.endpoint_pattern)
            ):

                transformed_data = self._apply_transformations(
                    transformed_data, rule.transformations
                )

        return transformed_data

    def _matches_endpoint(self, endpoint: str, pattern: str) -> bool:
        """Check if endpoint matches pattern"""

        if pattern == "*":
            return True

        # Simple pattern matching (can be enhanced with regex)
        if pattern.endswith("*"):
            return endpoint.startswith(pattern[:-1])

        return endpoint == pattern

    def _check_conditions(
        self, data: Dict[str, Any], conditions: Dict[str, Any]
    ) -> bool:
        """Check if conditions are met"""

        if not conditions:
            return True

        # Simple condition checking (can be enhanced)
        for key, expected_value in conditions.items():
            if key not in data or data[key] != expected_value:
                return False

        return True

    def _apply_transformations(
        self, data: Dict[str, Any], transformations: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply transformations to data"""

        transformed = data.copy()

        for transform_type, config in transformations.items():
            if transform_type == "rename_fields":
                for old_name, new_name in config.items():
                    if old_name in transformed:
                        transformed[new_name] = transformed.pop(old_name)

            elif transform_type == "add_fields":
                transformed.update(config)

            elif transform_type == "remove_fields":
                for field in config:
                    transformed.pop(field, None)

            elif transform_type == "modify_values":
                for field, modification in config.items():
                    if field in transformed:
                        if modification["type"] == "multiply":
                            transformed[field] *= modification["value"]
                        elif modification["type"] == "format":
                            transformed[field] = modification["format"].format(
                                transformed[field]
                            )

        return transformed


class MetricsCollector:
    """Prometheus metrics collection"""

    def __init__(self):
        # Request metrics
        self.request_count = Counter(
            "api_requests_total",
            "Total API requests",
            ["method", "endpoint", "status_code"],
        )

        self.request_duration = Histogram(
            "api_request_duration_seconds",
            "Request duration in seconds",
            ["method", "endpoint"],
        )

        self.rate_limit_hits = Counter(
            "api_rate_limit_hits_total",
            "Total rate limit hits",
            ["endpoint", "rule_id"],
        )

        # System metrics
        self.active_connections = Gauge(
            "api_active_connections", "Active API connections"
        )

        self.api_keys_active = Gauge("api_keys_active_total", "Total active API keys")

    def record_request(
        self, method: str, endpoint: str, status_code: int, duration: float
    ):
        """Record request metrics"""
        self.request_count.labels(
            method=method, endpoint=endpoint, status_code=status_code
        ).inc()
        self.request_duration.labels(method=method, endpoint=endpoint).observe(duration)

    def record_rate_limit_hit(self, endpoint: str, rule_id: str):
        """Record rate limit hit"""
        self.rate_limit_hits.labels(endpoint=endpoint, rule_id=rule_id).inc()

    def update_active_connections(self, count: int):
        """Update active connections count"""
        self.active_connections.set(count)

    def update_active_api_keys(self, count: int):
        """Update active API keys count"""
        self.api_keys_active.set(count)


class APIGatewayMiddleware(BaseHTTPMiddleware):
    """Main API Gateway middleware"""

    def __init__(self, app, gateway):
        super().__init__(app)
        self.gateway = gateway

    async def dispatch(self, request: Request, call_next):
        """Process request through gateway"""

        start_time = time.time()
        request_id = str(uuid.uuid4())

        # Extract client information
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent", "")

        # Log request
        logger.info(
            f"Request {request_id}: {request.method} {request.url} from {client_ip}"
        )

        try:
            # Authentication
            api_key_obj = None
            jwt_payload = None

            # Check API key authentication
            api_key = request.headers.get("X-API-Key") or request.query_params.get(
                "api_key"
            )
            if api_key:
                api_key_obj = await self.gateway.auth_manager.authenticate_api_key(
                    api_key, client_ip
                )
                if not api_key_obj:
                    return JSONResponse(
                        status_code=401, content={"error": "Invalid API key"}
                    )

            # Check JWT authentication
            auth_header = request.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header[7:]
                jwt_payload = self.gateway.auth_manager.verify_jwt_token(token)
                if not jwt_payload:
                    return JSONResponse(
                        status_code=401, content={"error": "Invalid or expired token"}
                    )

            # Rate limiting
            rate_limit_key = self._get_rate_limit_key(
                request, api_key_obj, jwt_payload, client_ip
            )

            for rule_id, rule in self.gateway.rate_limiter.rules.items():
                if self._rule_applies_to_endpoint(rule, str(request.url.path)):
                    allowed, limit_info = (
                        await self.gateway.rate_limiter.check_rate_limit(
                            rate_limit_key, rule_id
                        )
                    )

                    if not allowed:
                        self.gateway.metrics.record_rate_limit_hit(
                            str(request.url.path), rule_id
                        )

                        return JSONResponse(
                            status_code=429,
                            content={
                                "error": "Rate limit exceeded",
                                "limit_info": limit_info,
                            },
                            headers={
                                "X-RateLimit-Limit": str(rule.requests_per_minute),
                                "X-RateLimit-Remaining": str(
                                    limit_info.get("requests_remaining", 0)
                                ),
                                "X-RateLimit-Reset": str(
                                    limit_info.get("reset_time", 0)
                                ),
                            },
                        )

            # Request transformation
            if request.method in ["POST", "PUT", "PATCH"]:
                try:
                    body = await request.body()
                    if body:
                        request_data = json.loads(body)
                        transformed_data = self.gateway.transformer.transform_request(
                            str(request.url.path), request_data
                        )

                        # Modify request with transformed data
                        request._body = json.dumps(transformed_data).encode()
                except Exception as e:
                    logger.warning(f"Request transformation failed: {e}")

            # Add context to request
            request.state.request_id = request_id
            request.state.api_key_obj = api_key_obj
            request.state.jwt_payload = jwt_payload
            request.state.client_ip = client_ip

            # Call next middleware/endpoint
            response = await call_next(request)

            # Response transformation
            if (
                response.status_code == 200
                and "application/json" in response.headers.get("content-type", "")
            ):
                try:
                    response_body = b""
                    async for chunk in response.body_iterator:
                        response_body += chunk

                    response_data = json.loads(response_body)
                    transformed_response = self.gateway.transformer.transform_response(
                        str(request.url.path), response_data
                    )

                    response = JSONResponse(
                        content=transformed_response,
                        status_code=response.status_code,
                        headers=dict(response.headers),
                    )

                except Exception as e:
                    logger.warning(f"Response transformation failed: {e}")

            # Record metrics
            duration = time.time() - start_time
            self.gateway.metrics.record_request(
                request.method, str(request.url.path), response.status_code, duration
            )

            # Log request completion
            self._log_request(
                request_id, request, response, duration, api_key_obj, jwt_payload
            )

            return response

        except Exception as e:
            logger.error(f"Gateway error for request {request_id}: {e}")

            # Record error metrics
            duration = time.time() - start_time
            self.gateway.metrics.record_request(
                request.method, str(request.url.path), 500, duration
            )

            return JSONResponse(
                status_code=500, content={"error": "Internal server error"}
            )

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address"""
        # Check for forwarded headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        return request.client.host if request.client else "unknown"

    def _get_rate_limit_key(
        self,
        request: Request,
        api_key_obj: Optional[APIKey],
        jwt_payload: Optional[Dict],
        client_ip: str,
    ) -> str:
        """Generate rate limit key based on scope"""

        if api_key_obj:
            return f"api_key:{api_key_obj.key_id}"
        elif jwt_payload:
            return f"user:{jwt_payload['user_id']}"
        else:
            return f"ip:{client_ip}"

    def _rule_applies_to_endpoint(self, rule: RateLimitRule, endpoint: str) -> bool:
        """Check if rate limit rule applies to endpoint"""

        if "*" in rule.endpoints:
            return True

        return any(
            endpoint.startswith(pattern.rstrip("*")) for pattern in rule.endpoints
        )

    def _log_request(
        self,
        request_id: str,
        request: Request,
        response: Response,
        duration: float,
        api_key_obj: Optional[APIKey],
        jwt_payload: Optional[Dict],
    ):
        """Log request details"""

        log_entry = RequestLog(
            request_id=request_id,
            timestamp=datetime.now(timezone.utc),
            method=request.method,
            endpoint=str(request.url.path),
            client_ip=getattr(request.state, "client_ip", "unknown"),
            user_agent=request.headers.get("user-agent", ""),
            api_key_id=api_key_obj.key_id if api_key_obj else None,
            user_id=jwt_payload.get("user_id") if jwt_payload else None,
            request_size=int(request.headers.get("content-length", 0)),
            response_size=0,  # TODO: Calculate response size
            response_time_ms=duration * 1000,
            status_code=response.status_code,
            error_message=None,
            rate_limited=False,
        )

        # Store log entry (implement storage mechanism)
        logger.info(
            f"Request completed: {request_id} - {response.status_code} - {duration:.3f}s"
        )


class APIGateway:
    """Advanced API Gateway for Fraud Detection System"""

    def __init__(self, redis_url: str = "redis://localhost:6379"):

        # Initialize components
        self.redis_client = redis.from_url(redis_url)
        self.rate_limiter = RateLimiter(self.redis_client)
        self.auth_manager = AuthenticationManager(self.redis_client)
        self.transformer = RequestTransformer()
        self.metrics = MetricsCollector()

        # FastAPI app
        self.app = FastAPI(
            title="Fraud Detection API Gateway",
            description="Advanced API Gateway with rate limiting, authentication, and transformation",
            version="1.0.0",
        )

        # Add middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        self.app.add_middleware(APIGatewayMiddleware, gateway=self)

        # Configuration
        self.config = {
            "default_rate_limits": {
                "requests_per_minute": 100,
                "requests_per_hour": 1000,
                "requests_per_day": 10000,
                "burst_limit": 20,
            },
            "jwt_settings": {
                "secret_key": "your-secret-key",
                "algorithm": "HS256",
                "expire_minutes": 60,
            },
        }

        # Setup default routes
        self._setup_routes()

        # Load configuration
        self._load_configuration()

    def _setup_routes(self):
        """Setup gateway management routes"""

        @self.app.get("/gateway/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "version": "1.0.0",
            }

        @self.app.get("/gateway/metrics")
        async def get_metrics():
            """Prometheus metrics endpoint"""
            return Response(content=generate_latest(), media_type="text/plain")

        @self.app.post("/gateway/api-keys")
        async def create_api_key(key_data: Dict[str, Any]):
            """Create new API key"""
            api_key = self.auth_manager.create_api_key(key_data)
            return {"api_key": api_key}

        @self.app.get("/gateway/api-keys")
        async def list_api_keys():
            """List API keys"""
            keys = []
            for api_key, key_obj in self.auth_manager.api_keys.items():
                keys.append(
                    {
                        "key_id": key_obj.key_id,
                        "name": key_obj.name,
                        "organization": key_obj.organization,
                        "scopes": key_obj.scopes,
                        "active": key_obj.active,
                        "created_at": key_obj.created_at.isoformat(),
                        "last_used": (
                            key_obj.last_used.isoformat() if key_obj.last_used else None
                        ),
                    }
                )
            return {"api_keys": keys}

        @self.app.post("/gateway/auth/token")
        async def create_token(credentials: Dict[str, Any]):
            """Create JWT token"""
            # TODO: Implement proper user authentication
            user_id = credentials.get("user_id")
            scopes = credentials.get("scopes", ["read"])

            token = self.auth_manager.create_jwt_token(user_id, scopes)
            return {"access_token": token, "token_type": "bearer"}

        @self.app.post("/gateway/rate-limits")
        async def create_rate_limit_rule(rule_data: Dict[str, Any]):
            """Create rate limiting rule"""
            rule = RateLimitRule(
                rule_id=str(uuid.uuid4()),
                name=rule_data["name"],
                requests_per_minute=rule_data["requests_per_minute"],
                requests_per_hour=rule_data["requests_per_hour"],
                requests_per_day=rule_data["requests_per_day"],
                burst_limit=rule_data["burst_limit"],
                scope=rule_data["scope"],
                endpoints=rule_data["endpoints"],
                priority=rule_data.get("priority", 1),
                active=True,
                created_at=datetime.now(timezone.utc),
            )

            self.rate_limiter.add_rule(rule)
            return {"rule_id": rule.rule_id}

        @self.app.post("/gateway/transformations")
        async def create_transformation_rule(rule_data: Dict[str, Any]):
            """Create transformation rule"""
            rule = TransformationRule(
                rule_id=str(uuid.uuid4()),
                name=rule_data["name"],
                endpoint_pattern=rule_data["endpoint_pattern"],
                transform_type=rule_data["transform_type"],
                transformations=rule_data["transformations"],
                conditions=rule_data.get("conditions", {}),
                active=True,
                created_at=datetime.now(timezone.utc),
            )

            self.transformer.add_transformation_rule(rule)
            return {"rule_id": rule.rule_id}

    def _load_configuration(self):
        """Load gateway configuration"""

        # Create default rate limiting rules
        default_rule = RateLimitRule(
            rule_id="default",
            name="Default Rate Limit",
            requests_per_minute=self.config["default_rate_limits"][
                "requests_per_minute"
            ],
            requests_per_hour=self.config["default_rate_limits"]["requests_per_hour"],
            requests_per_day=self.config["default_rate_limits"]["requests_per_day"],
            burst_limit=self.config["default_rate_limits"]["burst_limit"],
            scope="ip",
            endpoints=["*"],
            priority=1,
            active=True,
            created_at=datetime.now(timezone.utc),
        )

        self.rate_limiter.add_rule(default_rule)

        # Example transformation rule
        example_transform = TransformationRule(
            rule_id="example",
            name="Example Response Transform",
            endpoint_pattern="/api/v1/fraud/analyze",
            transform_type="response",
            transformations={
                "add_fields": {
                    "api_version": "1.0",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            },
            conditions={},
            active=True,
            created_at=datetime.now(timezone.utc),
        )

        self.transformer.add_transformation_rule(example_transform)

    def add_upstream_service(self, prefix: str, target_url: str):
        """Add upstream service proxy"""

        @self.app.api_route(
            f"{prefix}/{{path:path}}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"]
        )
        async def proxy_to_upstream(path: str, request: Request):
            """Proxy request to upstream service"""

            import httpx

            # Build target URL
            target = f"{target_url.rstrip('/')}/{path}"
            if request.query_params:
                target += f"?{request.query_params}"

            # Prepare headers
            headers = dict(request.headers)
            headers.pop("host", None)  # Remove host header

            # Prepare request body
            body = None
            if request.method in ["POST", "PUT", "PATCH"]:
                body = await request.body()

            # Make upstream request
            async with httpx.AsyncClient() as client:
                response = await client.request(
                    method=request.method,
                    url=target,
                    headers=headers,
                    content=body,
                    timeout=30.0,
                )

            # Return response
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=dict(response.headers),
            )

    def run(self, host: str = "0.0.0.0", port: int = 8080, debug: bool = False):
        """Run the API gateway"""

        logger.info(f"Starting API Gateway on {host}:{port}")

        uvicorn.run(
            self.app,
            host=host,
            port=port,
            debug=debug,
            access_log=False,  # We handle logging in middleware
        )


# =====================================================
# USAGE EXAMPLE
# =====================================================


def main():
    """Example usage of API Gateway"""

    # Initialize gateway
    gateway = APIGateway()

    # Add upstream services
    gateway.add_upstream_service("/api/v1/fraud", "http://localhost:8001")
    gateway.add_upstream_service("/api/v1/ml", "http://localhost:8002")
    gateway.add_upstream_service("/api/v1/analytics", "http://localhost:8003")

    # Run gateway
    gateway.run(host="0.0.0.0", port=8080, debug=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
