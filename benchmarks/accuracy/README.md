# Accuracy Benchmarks

Validates that spark-llm-eval metrics match reference implementations.

## Overview

These benchmarks ensure our metric implementations produce the same (or statistically equivalent) results as established reference implementations:

| Metric | Reference Implementation |
|--------|-------------------------|
| exact_match | String comparison |
| f1 | SQuAD evaluation script |
| bleu | sacrebleu / nltk |
| rouge_l | rouge_score |
| bertscore | bert_score |

## Running Benchmarks

### Validate All Metrics

```bash
python benchmark_accuracy.py --all
```

### Validate Specific Metric

```bash
python benchmark_accuracy.py --metric exact_match
python benchmark_accuracy.py --metric f1
python benchmark_accuracy.py --metric bleu
```

### Save Results

```bash
python benchmark_accuracy.py --all --output results.json
```

## Pass Criteria

A metric validation passes if:
- **Max difference** from reference is below threshold
- **Correlation** with reference is > 0.99

| Metric | Threshold | Notes |
|--------|-----------|-------|
| exact_match | 0.001 | Should be identical |
| f1 | 0.01 | Minor tokenization differences allowed |
| bleu | 0.01 | Smoothing variations |
| rouge_l | 0.01 | Stemming variations |
| bertscore | 0.02 | Model loading variations |

## Results Format

```json
{
  "metric_name": "f1",
  "num_samples": 100,
  "max_difference": 0.0023,
  "mean_difference": 0.0008,
  "correlation": 0.9987,
  "passed": true,
  "threshold": 0.01
}
```

## Test Cases

Test cases cover edge cases:
- Exact matches
- Case differences
- Empty strings
- Partial overlaps
- Long text
- Special characters

## TODO

- [ ] Add more test cases from standard benchmarks
- [ ] Validate against SQuAD evaluation script
- [ ] Validate against sacrebleu
- [ ] Add statistical significance testing
- [ ] Create CI integration
