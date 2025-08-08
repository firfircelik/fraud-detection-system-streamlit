#!/bin/bash

# 🚀 Fraud Detection System - Local Environment Setup (Docker-free)
# Bu script Docker kullanmadan sistemi yerel olarak çalıştırmak için gerekli servisleri kurar

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Functions
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
    exit 1
}

warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

# Header
echo -e "${PURPLE}"
echo "  ███████╗██████╗  █████╗ ██╗   ██╗██████╗ "
echo "  ██╔════╝██╔══██╗██╔══██╗██║   ██║██╔══██╗"
echo "  █████╗  ██████╔╝███████║██║   ██║██║  ██║"
echo "  ██╔══╝  ██╔══██╗██╔══██║██║   ██║██║  ██║"
echo "  ██║     ██║  ██║██║  ██║╚██████╔╝██████╔╝"
echo "  ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═════╝ "
echo -e "${NC}"
echo "🚀 Local Environment Setup (Docker-free)"
echo "======================================="

# Check if Homebrew is installed (macOS)
check_homebrew() {
    if ! command -v brew &> /dev/null; then
        log "Homebrew not found. Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        success "Homebrew installed"
    else
        success "Homebrew already installed"
    fi
}

# Install PostgreSQL
install_postgresql() {
    log "Installing PostgreSQL..."
    if ! command -v psql &> /dev/null; then
        brew install postgresql@15
        brew services start postgresql@15
        success "PostgreSQL installed and started"
    else
        success "PostgreSQL already installed"
        # Start if not running
        if ! brew services list | grep postgresql | grep started &> /dev/null; then
            brew services start postgresql@15
            log "PostgreSQL started"
        fi
    fi
    
    # Create database and user
    log "Setting up fraud detection database..."
    createdb fraud_detection 2>/dev/null || true
    psql -d fraud_detection -c "CREATE USER fraud_admin WITH PASSWORD 'FraudDetection2024!';" 2>/dev/null || true
    psql -d fraud_detection -c "GRANT ALL PRIVILEGES ON DATABASE fraud_detection TO fraud_admin;" 2>/dev/null || true
    psql -d fraud_detection -c "ALTER USER fraud_admin CREATEDB;" 2>/dev/null || true
    
    # Run schema setup
    if [ -f "database/enterprise-schema.sql" ]; then
        psql -d fraud_detection -f database/enterprise-schema.sql
        success "Database schema created"
    fi
}

# Install Redis
install_redis() {
    log "Installing Redis..."
    if ! command -v redis-server &> /dev/null; then
        brew install redis
        brew services start redis
        success "Redis installed and started"
    else
        success "Redis already installed"
        # Start if not running
        if ! brew services list | grep redis | grep started &> /dev/null; then
            brew services start redis
            log "Redis started"
        fi
    fi
}

# Install Python dependencies
setup_python_environment() {
    log "Setting up Python environment..."
    
    # Check Python version
    if ! command -v python3 &> /dev/null; then
        error "Python 3 is required. Please install Python 3.8 or higher."
    fi
    
    # Create virtual environment for backend
    log "Setting up backend environment..."
    cd backend
    if [ ! -d "fraud-env" ]; then
        python3 -m venv fraud-env
    fi
    source fraud-env/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    deactivate
    cd ..
    
    # Create virtual environment for frontend
    log "Setting up frontend environment..."
    if [ ! -d "streamlit-env" ]; then
        python3 -m venv streamlit-env
    fi
    source streamlit-env/bin/activate
    pip install --upgrade pip
    pip install -r streamlit_requirements.txt
    deactivate
    
    success "Python environments created"
}

# Create environment file
setup_environment_file() {
    log "Creating environment configuration..."
    
    if [ ! -f ".env" ]; then
        cat > .env << EOF
# Database Configuration
POSTGRES_URL=postgresql://fraud_admin:FraudDetection2024!@localhost:5432/fraud_detection
REDIS_URL=redis://localhost:6379

# API Configuration
API_HOST=localhost
API_PORT=8000
STREAMLIT_PORT=8501

# Environment
ENVIRONMENT=local
DEBUG=true
EOF
        success "Environment file created"
    else
        warning "Environment file already exists"
    fi
}

# Create startup script
create_startup_script() {
    log "Creating startup script..."
    
    cat > start-local.sh << 'EOF'
#!/bin/bash

# Start Fraud Detection System Locally

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

# Check if services are running
check_services() {
    log "Checking required services..."
    
    # Check PostgreSQL
    if ! pg_isready -h localhost -p 5432 &> /dev/null; then
        warning "PostgreSQL is not running. Starting..."
        brew services start postgresql@15
        sleep 3
    fi
    
    # Check Redis
    if ! redis-cli ping &> /dev/null; then
        warning "Redis is not running. Starting..."
        brew services start redis
        sleep 2
    fi
    
    success "All services are running"
}

# Start backend
start_backend() {
    log "Starting backend API..."
    cd backend
    source fraud-env/bin/activate
    
    # Set environment variables
    export POSTGRES_URL="postgresql://fraud_admin:FraudDetection2024!@localhost:5432/fraud_detection"
    export REDIS_URL="redis://localhost:6379"
    
    # Start FastAPI server
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload &
    BACKEND_PID=$!
    echo $BACKEND_PID > ../backend.pid
    
    cd ..
    success "Backend started on http://localhost:8000"
}

# Start frontend
start_frontend() {
    log "Starting Streamlit frontend..."
    source streamlit-env/bin/activate
    
    # Start Streamlit
    streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0 &
    FRONTEND_PID=$!
    echo $FRONTEND_PID > frontend.pid
    
    success "Frontend started on http://localhost:8501"
}

# Cleanup function
cleanup() {
    log "Stopping services..."
    
    if [ -f "backend.pid" ]; then
        kill $(cat backend.pid) 2>/dev/null || true
        rm backend.pid
    fi
    
    if [ -f "frontend.pid" ]; then
        kill $(cat frontend.pid) 2>/dev/null || true
        rm frontend.pid
    fi
    
    success "Services stopped"
    exit 0
}

# Set trap for cleanup
trap cleanup INT TERM

# Main execution
log "Starting Fraud Detection System locally..."

check_services
start_backend
sleep 5  # Wait for backend to start
start_frontend

echo ""
success "🚀 Fraud Detection System is running!"
echo -e "${GREEN}Frontend: http://localhost:8501${NC}"
echo -e "${GREEN}Backend API: http://localhost:8000${NC}"
echo -e "${GREEN}API Docs: http://localhost:8000/docs${NC}"
echo ""
warning "Press Ctrl+C to stop all services"

# Wait for interrupt
wait
EOF

    chmod +x start-local.sh
    success "Startup script created: start-local.sh"
}

# Create stop script
create_stop_script() {
    log "Creating stop script..."
    
    cat > stop-local.sh << 'EOF'
#!/bin/bash

# Stop Fraud Detection System

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log "Stopping Fraud Detection System..."

# Stop backend
if [ -f "backend.pid" ]; then
    kill $(cat backend.pid) 2>/dev/null || true
    rm backend.pid
    log "Backend stopped"
fi

# Stop frontend
if [ -f "frontend.pid" ]; then
    kill $(cat frontend.pid) 2>/dev/null || true
    rm frontend.pid
    log "Frontend stopped"
fi

# Kill any remaining processes
pkill -f "uvicorn api.main:app" 2>/dev/null || true
pkill -f "streamlit run streamlit_app.py" 2>/dev/null || true

success "All services stopped"
EOF

    chmod +x stop-local.sh
    success "Stop script created: stop-local.sh"
}

# Main installation
main() {
    log "Starting local environment setup..."
    
    # Check if we're on macOS
    if [[ "$OSTYPE" != "darwin"* ]]; then
        warning "This script is designed for macOS. For other systems, please install PostgreSQL and Redis manually."
    fi
    
    check_homebrew
    install_postgresql
    install_redis
    setup_python_environment
    setup_environment_file
    create_startup_script
    create_stop_script
    
    echo ""
    success "🎉 Local environment setup complete!"
    echo ""
    echo -e "${GREEN}To start the system:${NC}"
    echo -e "${BLUE}  ./start-local.sh${NC}"
    echo ""
    echo -e "${GREEN}To stop the system:${NC}"
    echo -e "${BLUE}  ./stop-local.sh${NC}"
    echo ""
    echo -e "${YELLOW}Note: This setup uses local PostgreSQL and Redis instead of Docker containers.${NC}"
    echo -e "${YELLOW}The system will be available at:${NC}"
    echo -e "${GREEN}  - Frontend: http://localhost:8501${NC}"
    echo -e "${GREEN}  - Backend API: http://localhost:8000${NC}"
    echo -e "${GREEN}  - API Documentation: http://localhost:8000/docs${NC}"
}

# Run setup
main