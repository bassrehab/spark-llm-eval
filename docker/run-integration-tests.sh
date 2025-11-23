#!/bin/bash
# Integration test runner script
# Usage: ./run-integration-tests.sh [test-options]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== spark-llm-eval Integration Tests ===${NC}"

# Check for .env file
if [ ! -f "$PROJECT_ROOT/.env" ]; then
    echo -e "${YELLOW}Warning: .env file not found. Creating template...${NC}"
    cat > "$PROJECT_ROOT/.env" << 'EOF'
# API Keys for LLM providers
# Get keys from:
# - OpenAI: https://platform.openai.com/api-keys
# - Anthropic: https://console.anthropic.com/
# - Google: https://aistudio.google.com/apikey

OPENAI_API_KEY=your-openai-key-here
ANTHROPIC_API_KEY=your-anthropic-key-here
GOOGLE_API_KEY=your-google-key-here
EOF
    echo -e "${RED}Please edit .env with your API keys before running tests.${NC}"
    exit 1
fi

# Load environment variables
export $(grep -v '^#' "$PROJECT_ROOT/.env" | xargs)

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

# Start Spark cluster
echo -e "${GREEN}Starting Spark cluster...${NC}"
cd "$SCRIPT_DIR"
docker-compose up -d

# Wait for Spark master to be ready
echo -e "${YELLOW}Waiting for Spark cluster to be ready...${NC}"
sleep 10

# Check cluster status
echo -e "${GREEN}Cluster status:${NC}"
docker-compose ps

# Run tests inside the test-runner container
echo -e "${GREEN}Running integration tests...${NC}"

# Install package and dependencies in container
docker exec test-runner bash -c "
    cd /app
    pip install -q -e '.[dev]' 2>/dev/null
    pip install -q openai anthropic google-generativeai 2>/dev/null
"

# Run pytest with passed arguments or defaults
if [ $# -eq 0 ]; then
    # Default: run all integration tests with verbose output
    docker exec -e OPENAI_API_KEY="$OPENAI_API_KEY" \
                -e ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
                -e GOOGLE_API_KEY="$GOOGLE_API_KEY" \
                -e SPARK_MASTER="spark://spark-master:7077" \
                test-runner \
                pytest /app/tests/integration -v --tb=short
else
    # Run with provided arguments
    docker exec -e OPENAI_API_KEY="$OPENAI_API_KEY" \
                -e ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
                -e GOOGLE_API_KEY="$GOOGLE_API_KEY" \
                -e SPARK_MASTER="spark://spark-master:7077" \
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

# Ask if user wants to stop the cluster
echo ""
read -p "Stop Spark cluster? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Stopping Spark cluster...${NC}"
    docker-compose down
    echo -e "${GREEN}Done.${NC}"
else
    echo -e "${YELLOW}Cluster still running. Stop with: cd docker && docker-compose down${NC}"
fi

exit $TEST_EXIT_CODE
