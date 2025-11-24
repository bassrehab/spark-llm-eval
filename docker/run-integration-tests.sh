#!/bin/bash
# Integration test runner script for spark-llm-eval
# Usage: ./run-integration-tests.sh [test-options]
#
# Web UIs available after cluster starts:
#   - Spark Master:    http://localhost:8080
#   - Spark History:   http://localhost:18080
#   - MLflow:          http://localhost:5000
#   - Spark App UI:    http://localhost:4040 (during job execution)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== spark-llm-eval Integration Tests ===${NC}"
echo ""

# Check for .env file
if [ ! -f "$PROJECT_ROOT/.env" ]; then
    echo -e "${YELLOW}Warning: .env file not found. Creating template...${NC}"
    cat > "$PROJECT_ROOT/.env" << 'EOF'
# API Keys for LLM providers
# Get keys from:
# - OpenAI: https://platform.openai.com/api-keys
# - Anthropic: https://console.anthropic.com/
# - Google: https://aistudio.google.com/apikey (FREE tier available)

OPENAI_API_KEY=your-openai-key-here
ANTHROPIC_API_KEY=your-anthropic-key-here
GOOGLE_API_KEY=your-google-key-here
EOF
    echo -e "${RED}Please edit .env with your API keys before running tests.${NC}"
    exit 1
fi

# Load environment variables
set -a
source "$PROJECT_ROOT/.env"
set +a

# Verify at least one API key is set
if [ -z "$OPENAI_API_KEY" ] || [ "$OPENAI_API_KEY" = "your-openai-key-here" ]; then
    if [ -z "$ANTHROPIC_API_KEY" ] || [ "$ANTHROPIC_API_KEY" = "your-anthropic-key-here" ]; then
        if [ -z "$GOOGLE_API_KEY" ] || [ "$GOOGLE_API_KEY" = "your-google-key-here" ]; then
            echo -e "${RED}Error: No valid API keys found in .env${NC}"
            exit 1
        fi
    fi
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Error: Docker is not running. Please start Docker Desktop.${NC}"
    exit 1
fi

cd "$SCRIPT_DIR"

# Build custom image if needed
echo -e "${GREEN}Building custom Spark image with Python 3.11...${NC}"
docker-compose build --quiet

# Stop any existing containers
echo -e "${YELLOW}Stopping any existing containers...${NC}"
docker-compose down --remove-orphans 2>/dev/null || true

# Start Spark cluster
echo -e "${GREEN}Starting Spark cluster with MLflow and History Server...${NC}"
docker-compose up -d

# Wait for services to be ready
echo -e "${YELLOW}Waiting for services to start...${NC}"
sleep 15

# Check cluster status
echo ""
echo -e "${GREEN}Cluster status:${NC}"
docker-compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}"

echo ""
echo -e "${BLUE}=== Web UIs ===${NC}"
echo -e "  Spark Master:     ${GREEN}http://localhost:8080${NC}"
echo -e "  Spark History:    ${GREEN}http://localhost:18080${NC}"
echo -e "  MLflow Tracking:  ${GREEN}http://localhost:5000${NC}"
echo -e "  Spark App UI:     ${GREEN}http://localhost:4040${NC} (during job execution)"
echo ""

# Verify workers are registered
echo -e "${YELLOW}Checking worker registration...${NC}"
sleep 5
WORKERS=$(docker logs spark-master 2>&1 | grep -c "Registering worker" || echo "0")
echo -e "Workers registered: ${GREEN}$WORKERS${NC}"

# Install the package in test-runner
echo -e "${GREEN}Installing spark-llm-eval in test container...${NC}"
docker exec test-runner pip install -q -e /app 2>/dev/null

# Run tests inside the test-runner container
echo ""
echo -e "${GREEN}Running integration tests on Spark cluster...${NC}"
echo ""

# Run pytest with passed arguments or defaults
if [ $# -eq 0 ]; then
    # Default: run all integration tests with verbose output
    docker exec \
        -e OPENAI_API_KEY="$OPENAI_API_KEY" \
        -e ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
        -e GOOGLE_API_KEY="$GOOGLE_API_KEY" \
        -e SPARK_MASTER="spark://spark-master:7077" \
        -e MLFLOW_TRACKING_URI="http://mlflow:5000" \
        test-runner \
        pytest /app/tests/integration -v --tb=short
else
    # Run with provided arguments
    docker exec \
        -e OPENAI_API_KEY="$OPENAI_API_KEY" \
        -e ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
        -e GOOGLE_API_KEY="$GOOGLE_API_KEY" \
        -e SPARK_MASTER="spark://spark-master:7077" \
        -e MLFLOW_TRACKING_URI="http://mlflow:5000" \
        test-runner \
        pytest /app/tests/integration "$@"
fi

TEST_EXIT_CODE=$?

echo ""
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}=== All tests passed! ===${NC}"
else
    echo -e "${RED}=== Some tests failed (exit code: $TEST_EXIT_CODE) ===${NC}"
fi

echo ""
echo -e "${BLUE}View job history at: ${GREEN}http://localhost:18080${NC}"
echo -e "${BLUE}View MLflow runs at: ${GREEN}http://localhost:5000${NC}"
echo ""

# Ask if user wants to stop the cluster
read -p "Stop Spark cluster? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Stopping Spark cluster...${NC}"
    docker-compose down
    echo -e "${GREEN}Done.${NC}"
else
    echo -e "${YELLOW}Cluster still running. Access UIs at:${NC}"
    echo -e "  - Spark Master:  http://localhost:8080"
    echo -e "  - History:       http://localhost:18080"
    echo -e "  - MLflow:        http://localhost:5000"
    echo -e "${YELLOW}Stop with: cd docker && docker-compose down${NC}"
fi

exit $TEST_EXIT_CODE
