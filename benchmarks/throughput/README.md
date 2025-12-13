# Throughput Benchmarks

Benchmarks measuring spark-llm-eval throughput at various scales.

## Overview

These benchmarks measure:
- **Examples per minute**: How many evaluation examples can be processed
- **Scaling efficiency**: How throughput scales with executor count
- **Comparison**: Performance vs single-machine sequential processing

## Running Benchmarks

### Basic Benchmark

```bash
python benchmark_throughput.py --num-examples 1000
```

### Scaling Benchmark

```bash
python benchmark_throughput.py --scaling --output results.json
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `--num-examples` | Number of examples to evaluate | 100 |
| `--batch-size` | Batch size for inference | 32 |
| `--model` | Model to use | gpt-4o-mini |
| `--output` | Output file for results (JSON) | None |
| `--scaling` | Run at multiple scales | False |

## Expected Results

Target performance metrics:
- **Throughput**: >10,000 examples/minute (API-limited)
- **Scaling**: Linear with executor count
- **Speedup**: 5-10x vs single-machine for large datasets

## Benchmark Scenarios

### 1. Small Scale (100-1K examples)
- Validates correctness
- Measures baseline latency

### 2. Medium Scale (1K-10K examples)
- Tests batching efficiency
- Measures rate limiting behavior

### 3. Large Scale (10K-100K examples)
- Validates linear scaling
- Tests fault tolerance

### 4. Enterprise Scale (100K+ examples)
- Production-realistic workload
- Cost optimization testing

## Results Format

Results are saved as JSON:

```json
{
  "benchmark_name": "spark-llm-eval",
  "num_examples": 10000,
  "num_executors": 8,
  "total_time_seconds": 120.5,
  "examples_per_second": 83.0,
  "examples_per_minute": 4980.0,
  "avg_latency_ms": 450.0,
  "p95_latency_ms": 890.0,
  "timestamp": "2025-11-30T10:30:00"
}
```

## TODO

- [ ] Implement synthetic dataset generation
- [ ] Add comparison with lm-evaluation-harness
- [ ] Add comparison with ragas
- [ ] Add cost tracking
- [ ] Create visualization scripts
