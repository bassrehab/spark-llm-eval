# Integration Testing Guide

This guide explains how to set up and run integration tests for spark-llm-eval using a local Docker-based Apache Spark cluster.

## Prerequisites

- **Docker Desktop** (v4.0+) with at least 8GB RAM allocated
- **Docker Compose** (v2.0+)
- **Python 3.10+**
- **API Keys** for LLM providers (see below)

## Obtaining API Keys

### OpenAI API Key
1. Go to [OpenAI Platform](https://platform.openai.com/)
2. Sign in or create an account
3. Navigate to **API Keys** in the left sidebar
4. Click **Create new secret key**
5. Copy the key (starts with `sk-`)

**Pricing**: Pay-as-you-go. GPT-4o costs ~$5/1M input tokens, ~$15/1M output tokens.

### Anthropic (Claude) API Key
1. Go to [Anthropic Console](https://console.anthropic.com/)
2. Sign in or create an account
3. Navigate to **API Keys**
4. Click **Create Key**
5. Copy the key (starts with `sk-ant-`)

**Pricing**: Pay-as-you-go. Claude 3.5 Sonnet costs ~$3/1M input tokens, ~$15/1M output tokens.

### Google (Gemini) API Key
1. Go to [Google AI Studio](https://aistudio.google.com/apikey)
2. Sign in with your Google account
3. Click **Create API Key**
4. Select or create a Google Cloud project
5. Copy the key

**Pricing**: Free tier available (60 requests/minute for Gemini 1.5 Flash). Paid tier for higher limits.

## Environment Setup

### 1. Create Environment File

Create a `.env` file in the project root (this file is gitignored):

```bash
# .env
OPENAI_API_KEY=sk-your-openai-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
GOOGLE_API_KEY=your-google-api-key-here
```

### 2. Verify Docker Installation

```bash
# Check Docker is running
docker --version
docker-compose --version

# Check available resources (should have at least 8GB RAM)
docker system info | grep -i memory
```

### 3. Install Python Dependencies

```bash
# Install the package with all dependencies
pip install -e ".[dev]"

# Install provider-specific packages
pip install openai anthropic google-generativeai
```

## Starting the Spark Cluster

### 1. Start the Cluster

```bash
cd docker

# Start all services (detached mode)
docker-compose up -d

# Verify all containers are running
docker-compose ps
```

Expected output:
```
NAME             STATUS    PORTS
spark-master    running   0.0.0.0:7077->7077/tcp, 0.0.0.0:8080->8080/tcp
spark-worker-1  running
spark-worker-2  running
test-runner     running
```

### 2. Verify Spark Cluster

Open [http://localhost:8080](http://localhost:8080) in your browser to see the Spark Master UI.

You should see:
- 2 workers registered
- Each worker with 2 cores and 2GB memory

### 3. Test Spark Connectivity

```bash
# Enter the test-runner container
docker exec -it test-runner bash

# Test PySpark
python3 -c "
from pyspark.sql import SparkSession
spark = SparkSession.builder \
    .appName('test') \
    .master('spark://spark-master:7077') \
    .getOrCreate()
print(f'Spark version: {spark.version}')
print(f'Executors: {spark.sparkContext.defaultParallelism}')
spark.stop()
"
```

## Running Integration Tests

### Option 1: Run Inside Container (Recommended)

```bash
# Enter the test-runner container
docker exec -it test-runner bash

# Install the package
cd /app
pip install -e ".[dev]"
pip install openai anthropic google-generativeai

# Run integration tests
pytest tests/integration -v --tb=short

# Run specific test
pytest tests/integration/test_full_pipeline.py -v
```

### Option 2: Run from Host (Requires Spark on Host)

```bash
# Set environment variables
export OPENAI_API_KEY=your-key
export ANTHROPIC_API_KEY=your-key
export GOOGLE_API_KEY=your-key

# Run tests pointing to Docker cluster
SPARK_MASTER=spark://localhost:7077 pytest tests/integration -v
```

## Integration Test Categories

### 1. Inference Engine Tests (`test_inference_engines.py`)

Tests real API calls to each provider:
- OpenAI GPT-4o-mini
- Anthropic Claude 3 Haiku
- Google Gemini 1.5 Flash

**Estimated cost**: ~$0.01 per full run (uses cheapest models)

### 2. Spark UDF Tests (`test_spark_udfs.py`)

Tests Pandas UDFs executing on Spark workers:
- Batch inference across partitions
- Rate limiting behavior
- Error handling and retries

### 3. Delta Lake Tests (`test_delta_integration.py`)

Tests Delta Lake read/write operations:
- Write evaluation results
- Read with time travel
- Schema evolution

### 4. Full Pipeline Tests (`test_full_pipeline.py`)

End-to-end evaluation workflow:
- Load dataset
- Run distributed inference
- Compute metrics
- Store results

## Test Configuration

### Adjusting Test Parameters

Edit `tests/integration/conftest.py` to adjust:

```python
# Number of test examples (reduce for faster/cheaper tests)
TEST_SAMPLE_SIZE = 10

# Models to test (use cheaper models for routine testing)
TEST_MODELS = {
    "openai": "gpt-4o-mini",      # Cheapest OpenAI
    "anthropic": "claude-3-haiku-20240307",  # Cheapest Claude
    "google": "gemini-1.5-flash", # Free tier available
}

# Skip expensive tests
SKIP_EXPENSIVE_TESTS = True
```

### Running Subset of Tests

```bash
# Only test OpenAI
pytest tests/integration -v -k "openai"

# Only test inference (no Delta)
pytest tests/integration -v -k "inference"

# Skip slow tests
pytest tests/integration -v -m "not slow"
```

## Monitoring and Debugging

### View Spark Application UI

During test execution, open [http://localhost:4040](http://localhost:4040) to see:
- Active jobs and stages
- Task distribution across workers
- Memory usage
- Event timeline

### View Container Logs

```bash
# All containers
docker-compose logs -f

# Specific container
docker-compose logs -f spark-master
docker-compose logs -f test-runner
```

### Check API Rate Limits

If tests fail with rate limit errors:

```bash
# Check current usage (OpenAI)
curl https://api.openai.com/v1/usage -H "Authorization: Bearer $OPENAI_API_KEY"
```

## Cleanup

### Stop the Cluster

```bash
cd docker

# Stop containers (preserves data)
docker-compose stop

# Stop and remove containers
docker-compose down

# Remove all data volumes
docker-compose down -v
```

### Reset Everything

```bash
# Remove all containers, networks, and volumes
docker-compose down -v --rmi all

# Prune unused Docker resources
docker system prune -a
```

## Troubleshooting

### Issue: Workers Not Connecting

```bash
# Check network connectivity
docker exec spark-worker-1 ping spark-master

# Check Spark logs
docker logs spark-master 2>&1 | grep -i error
```

### Issue: Out of Memory

Increase Docker memory allocation in Docker Desktop settings, or reduce worker memory:

```yaml
# In docker-compose.yml
SPARK_WORKER_MEMORY=1G
```

### Issue: API Key Not Found

```bash
# Verify environment variables are passed
docker exec test-runner env | grep API_KEY
```

### Issue: Module Not Found

```bash
# Reinstall package in container
docker exec -it test-runner pip install -e /app[dev]
```

## Cost Estimation

Running the full integration test suite:

| Provider | Model | Requests | Est. Tokens | Est. Cost |
|----------|-------|----------|-------------|-----------|
| OpenAI | gpt-4o-mini | 10 | ~2K | $0.001 |
| Anthropic | claude-3-haiku | 10 | ~2K | $0.001 |
| Google | gemini-1.5-flash | 10 | ~2K | Free |

**Total estimated cost per run: < $0.01**

For development, you can run tests against a single provider to minimize costs.

## CI/CD Integration

For automated testing in CI pipelines, see `.github/workflows/integration-tests.yml` (if available) or set up:

```yaml
# Example GitHub Actions workflow
jobs:
  integration-tests:
    runs-on: ubuntu-latest
    services:
      spark:
        image: bitnami/spark:3.5.0
    env:
      OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
      ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
      GOOGLE_API_KEY: ${{ secrets.GOOGLE_API_KEY }}
    steps:
      - uses: actions/checkout@v4
      - name: Run integration tests
        run: pytest tests/integration -v
```
