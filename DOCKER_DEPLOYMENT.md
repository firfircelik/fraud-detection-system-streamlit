# 🐳 Docker Deployment Guide

## Overview

This guide explains how to deploy the Fraud Detection System using Docker and how the GitHub Actions CI/CD pipeline integrates with Docker and databases.

## 🚀 Quick Start with Docker

### Prerequisites
- Docker Engine 20.10+
- Docker Compose 2.0+
- 4GB+ RAM available
- 10GB+ disk space

### Local Development

```bash
# Clone the repository
git clone <repository-url>
cd fraud-detection-system-streamlit

# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f fraud-api
docker-compose logs -f fraud-frontend
```

### Service URLs

- **Streamlit Frontend**: http://localhost:8501
- **API Backend**: http://localhost:8080
- **PostgreSQL**: localhost:5432
- **Redis**: localhost:6379
- **Neo4j Browser**: http://localhost:7474
- **Grafana**: http://localhost:3002
- **Prometheus**: http://localhost:9090

## 🏗️ Architecture

### Core Services

1. **fraud-api** (Backend API)
   - FastAPI application
   - ML model serving
   - Database connections
   - Redis caching

2. **fraud-frontend** (Streamlit UI)
   - Interactive dashboard
   - Real-time monitoring
   - Data visualization

### Databases

1. **PostgreSQL** - Primary OLTP database
2. **Redis** - Caching and real-time features
3. **Neo4j** - Graph database for fraud ring detection
4. **TimescaleDB** - Time-series metrics
5. **Elasticsearch** - Search and analytics

### Monitoring

1. **Prometheus** - Metrics collection
2. **Grafana** - Visualization dashboards

## 🔧 Configuration

### Environment Variables

#### API Service
```env
POSTGRES_URL=postgresql://fraud_admin:FraudDetection2024!@postgres:5432/fraud_detection
REDIS_URL=redis://:RedisStack2024!@redis:6379
NEO4J_URL=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=FraudGraph2024!
API_ENV=production
LOG_LEVEL=INFO
```

#### Frontend Service
```env
API_URL=http://fraud-api:8080
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

### Database Credentials

| Service | Username | Password | Database |
|---------|----------|----------|---------|
| PostgreSQL | fraud_admin | FraudDetection2024! | fraud_detection |
| Neo4j | neo4j | FraudGraph2024! | neo4j |
| TimescaleDB | timescale_admin | TimeScale2024! | fraud_metrics |
| Redis | - | RedisStack2024! | - |

## 🚀 GitHub Actions CI/CD

### Pipeline Overview

Our CI/CD pipeline automatically:

1. **Code Quality Checks**
   - Black formatting
   - isort import sorting
   - Flake8 linting
   - MyPy type checking

2. **Security Scanning**
   - Bandit security analysis
   - Safety dependency scanning
   - CodeQL analysis
   - Trivy container scanning

3. **Testing**
   - Unit tests with pytest
   - Integration tests
   - Database connectivity tests
   - API endpoint tests

4. **Docker Build & Push**
   - Multi-stage Docker builds
   - Container security scanning
   - Push to GitHub Container Registry
   - Database integration testing

### Docker Images

Images are automatically built and pushed to GitHub Container Registry:

```bash
# Pull latest images
docker pull ghcr.io/your-username/fraud-detection-system-streamlit/fraud-detection-api:latest
docker pull ghcr.io/your-username/fraud-detection-system-streamlit/fraud-detection-streamlit:latest

# Pull specific version
docker pull ghcr.io/your-username/fraud-detection-system-streamlit/fraud-detection-api:commit-sha
```

### Database Integration

The CI/CD pipeline includes:

- **PostgreSQL Service**: Automated database setup with test data
- **Redis Service**: Caching layer for tests
- **Database Migrations**: Automatic schema updates
- **Connection Testing**: Validates database connectivity

## 🛠️ Development Workflow

### Local Development

```bash
# Start only databases
docker-compose up -d postgres redis

# Run API locally
cd backend
uvicorn api.main:app --reload --port 8080

# Run Streamlit locally
streamlit run streamlit_app.py
```

### Testing

```bash
# Run tests in Docker
docker-compose exec fraud-api pytest tests/ -v

# Run specific test
docker-compose exec fraud-api pytest tests/test_api.py::TestModelStatus -v

# Check test coverage
docker-compose exec fraud-api pytest --cov=api tests/
```

### Database Management

```bash
# Connect to PostgreSQL
docker-compose exec postgres psql -U fraud_admin -d fraud_detection

# Connect to Redis
docker-compose exec redis redis-cli -a RedisStack2024!

# Connect to Neo4j
docker-compose exec neo4j cypher-shell -u neo4j -p FraudGraph2024!
```

## 🔒 Security

### Container Security

- **Non-root users**: All containers run as non-root
- **Minimal base images**: Using Alpine/slim images
- **Security scanning**: Trivy scans for vulnerabilities
- **Secrets management**: Environment variables for credentials

### Network Security

- **Internal network**: Services communicate via Docker network
- **Port exposure**: Only necessary ports exposed
- **TLS encryption**: Database connections use TLS

## 📊 Monitoring

### Health Checks

All services include health checks:

```bash
# Check service health
docker-compose ps

# View health check logs
docker inspect fraud_api_enhanced --format='{{.State.Health.Status}}'
```

### Metrics

- **Prometheus**: Collects application and system metrics
- **Grafana**: Visualizes metrics with pre-built dashboards
- **Application logs**: Centralized logging with structured format

## 🚨 Troubleshooting

### Common Issues

1. **Port conflicts**
   ```bash
   # Check port usage
   netstat -tulpn | grep :8080
   
   # Stop conflicting services
   docker-compose down
   ```

2. **Database connection issues**
   ```bash
   # Check database logs
   docker-compose logs postgres
   
   # Test connection
   docker-compose exec fraud-api python -c "import psycopg2; print('DB OK')"
   ```

3. **Memory issues**
   ```bash
   # Check resource usage
   docker stats
   
   # Increase Docker memory limit
   # Docker Desktop > Settings > Resources > Memory
   ```

### Logs

```bash
# View all logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f fraud-api
docker-compose logs -f postgres

# Follow logs with timestamps
docker-compose logs -f -t fraud-api
```

## 🔄 Updates and Maintenance

### Updating Images

```bash
# Pull latest images
docker-compose pull

# Restart with new images
docker-compose up -d

# Remove old images
docker image prune
```

### Database Backups

```bash
# Backup PostgreSQL
docker-compose exec postgres pg_dump -U fraud_admin fraud_detection > backup.sql

# Restore PostgreSQL
docker-compose exec -T postgres psql -U fraud_admin fraud_detection < backup.sql
```

### Scaling

```bash
# Scale API service
docker-compose up -d --scale fraud-api=3

# Scale with load balancer
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

## 📝 Production Deployment

For production deployment:

1. **Use production profile**:
   ```bash
   docker-compose --profile production up -d
   ```

2. **Configure external databases**:
   - Use managed PostgreSQL (AWS RDS, Google Cloud SQL)
   - Use managed Redis (AWS ElastiCache, Google Memorystore)

3. **Set up monitoring**:
   - Configure Prometheus alerts
   - Set up log aggregation
   - Monitor resource usage

4. **Security hardening**:
   - Use secrets management
   - Enable TLS/SSL
   - Configure firewalls
   - Regular security updates

## 🤝 Contributing

When contributing:

1. Test locally with Docker
2. Ensure all tests pass in CI/CD
3. Update documentation
4. Follow security best practices

For more information, see the main README.md file.