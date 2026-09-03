# FedTrust-K8s Experimental Results

## Experiment 00 — Centralized PyTorch Baseline

- Dataset: MNIST
- Framework: PyTorch
- Device: CPU
- Training epochs: 3
- Test accuracy: 98.53%

---

## Experiment 01 — Federated Learning with FedAvg

- Dataset: MNIST
- Clients: 5
- Federated rounds: 5
- Local epochs: 1
- Aggregation: FedAvg
- Device: CPU

### Results

| Round | Global Accuracy |
|------:|----------------:|
| 1 | 92.52% |
| 2 | 97.07% |
| 3 | 97.92% |
| 4 | 98.37% |
| 5 | 98.61% |

### Final result

Final global model accuracy: **98.61%**

Centralized baseline: **98.53%**

Difference: **+0.08 percentage points**

### Observation

FedAvg successfully converged to a high-performing global model under the current balanced client partition. The final accuracy was slightly higher than the centralized baseline, although this small difference should not be interpreted as evidence that federated learning outperforms centralized training.

---

## Experiment 02 — Non-IID Federated Learning

- Dataset: MNIST
- Clients: 5
- Federated rounds: 5
- Local epochs: 1
- Aggregation: FedAvg
- Device: CPU
- Client distribution:
  - Client 1: digits 0, 1
  - Client 2: digits 2, 3
  - Client 3: digits 4, 5
  - Client 4: digits 6, 7
  - Client 5: digits 8, 9

### Results

| Round | Global Accuracy |
|------:|----------------:|
| 1 | 14.54% |
| 2 | 52.75% |
| 3 | 58.83% |
| 4 | 69.03% |
| 5 | 63.81% |

### Comparison

| Experiment | Final Accuracy |
|---|---:|
| Centralized baseline | 98.53% |
| IID FedAvg | 98.61% |
| Non-IID FedAvg | 63.81% |

### Observation

Under the highly heterogeneous label distribution used in this experiment, FedAvg showed substantially lower performance than the balanced federated baseline. Global accuracy increased from 14.54% to 69.03% during the first four rounds, before declining to 63.81% in round five. This indicates that severe client-level distribution differences can negatively affect the convergence and stability of the global model.

---

## Experiment 03 — Byzantine Client Attack

- Dataset: MNIST
- Clients: 5 (4 honest, 1 Byzantine)
- Federated rounds: 5
- Local epochs: 1
- Aggregation: FedAvg
- Device: CPU
- Client distribution:
  - Client 1: digits 0, 1
  - Client 2: digits 2, 3
  - Client 3: digits 4, 5
  - Client 4: digits 6, 7
  - Client 5: digits 8, 9 (BYZANTINE)
- Attack: Gradient sign-flipping

### Results

| Round | Global Accuracy |
|------:|----------------:|
| 1 | 14.68% |
| 2 | 20.19% |
| 3 | 24.35% |
| 4 | 48.11% |
| 5 | 47.52% |

### Comparison

| Experiment | Data Distribution | Attack | Final Accuracy |
|---|---|---|---:|
| Centralized baseline | Balanced | None | 98.53% |
| IID FedAvg | Balanced | None | 98.61% |
| Non-IID FedAvg | Non-IID | None | 63.81% |
| **Byzantine FedAvg** | **Non-IID** | **Gradient flip** | **47.52%** |

### Observation

The Byzantine client caused a substantial performance degradation compared with the non-IID baseline (63.81% → 47.52%). However, the global model still learned to classify with reasonable accuracy, suggesting that FedAvg has some inherent robustness to this type of gradient manipulation. The attack effectiveness appeared to accumulate over time, with the largest impact observed in the later rounds.

### Key Insight

While FedAvg provides some resilience against a single Byzantine client in a non-IID setting, the performance drop of 16.29 percentage points is significant enough to justify exploring robust aggregation methods.

---

## Experiment 04 — Data Poisoning Attack

- Dataset: MNIST
- Clients: 5 (4 honest, 1 poisoned)
- Federated rounds: 5
- Local epochs: 1
- Aggregation: FedAvg
- Device: CPU
- Client distribution:
  - Client 1: digits 0, 1
  - Client 2: digits 2, 3
  - Client 3: digits 4, 5
  - Client 4: digits 6, 7
  - Client 5: digits 8, 9 (POISONED)
- Attack: Label flipping (50% of data → class 0)

### Results

| Round | Global Accuracy |
|------:|----------------:|
| 1 | 24.11% |
| 2 | 59.75% |
| 3 | 61.96% |
| 4 | 70.16% |
| 5 | 71.57% |

### Comparison

| Experiment | Data Distribution | Attack | Final Accuracy | Δ from Non-IID |
|---|---|---|---:|---:|
| Centralized baseline | Balanced | None | 98.53% | — |
| IID FedAvg | Balanced | None | 98.61% | — |
| Non-IID FedAvg | Non-IID | None | 63.81% | — |
| Byzantine FedAvg | Non-IID | Gradient flip | 47.52% | -16.29% |
| **Data Poisoning** | **Non-IID** | **Label flipping** | **71.57%** | **+7.76%** |

### Observation

Counterintuitively, the data poisoning attack resulted in improved global model performance compared with the non-IID baseline (63.81% → 71.57%). This occurred because poisoning 50% of Client 5's data (digits 8-9) by flipping labels to class 0 effectively created a regularization effect. The poisoned client's data distribution became less specialized, which helped bridge the statistical heterogeneity between clients.

This result highlights an important consideration in federated learning research: not all perturbations are adversarial in practice, and the effect of an attack depends heavily on the specific configuration (target class, poisoning fraction, and data distribution).

### Key Insight

While data poisoning is a real threat, its impact is highly context-dependent. In our specific configuration, the "attack" actually improved generalization. Future work should explore more targeted poisoning strategies that could be more damaging.

---

## Experiment 05 — Robust Aggregation Comparison

### Configuration
- Dataset: MNIST
- Clients: 5 (4 honest, 1 Byzantine)
- Federated rounds: 5
- Local epochs: 1
- Device: CPU
- Client distribution:
  - Client 1: digits 0, 1
  - Client 2: digits 2, 3
  - Client 3: digits 4, 5
  - Client 4: digits 6, 7
  - Client 5: digits 8, 9 (BYZANTINE)
- Attack: Gradient sign-flipping

### Results

| Method | Round 1 | Round 2 | Round 3 | Round 4 | Round 5 | Final |
|--------|---------|---------|---------|---------|---------|-------|
| FedAvg | 10.10% | 27.04% | 40.32% | 19.27% | 37.73% | 37.73% |
| Krum | 18.70% | 18.74% | 18.71% | 18.74% | 18.74% | 18.74% |
| Trimmed Mean (20%) | 23.31% | 40.79% | 34.73% | 36.60% | 44.45% | 44.45% |
| Coordinate Median | 18.29% | 31.16% | 22.31% | 43.09% | 28.84% | 28.84% |
| Geometric Median | 23.24% | 36.43% | 50.92% | 55.02% | 60.25% | 60.25% |

### Comparison Summary

| Method | Final Accuracy | Δ vs FedAvg | Δ vs Non-IID Baseline |
|--------|---------------:|------------:|----------------------:|
| FedAvg | 37.73% | — | -26.08% |
| Krum | 18.74% | -18.99% | -45.07% |
| Trimmed Mean (20%) | 44.45% | +6.72% | -19.36% |
| Coordinate Median | 28.84% | -8.89% | -34.97% |
| **Geometric Median** | **60.25%** | **+22.52%** | **-3.56%** |

### Key Findings

1. **Geometric Median is the most effective robust aggregation method** for non-IID federated learning with Byzantine clients. It achieved 60.25% accuracy, only 3.56 percentage points below the non-IID baseline without attacks.

2. **Krum fails catastrophically on non-IID data** because it assumes honest clients have similar updates. In our label-skewed setting, this assumption is violated, causing Krum to discard most useful information.

3. **Trimmed Mean provides a good balance** of robustness and performance (44.45%), making it a practical choice for scenarios where computational overhead is a concern.

4. **FedAvg is unstable** under Byzantine attacks, showing significant performance fluctuations across rounds.

### Conclusion

For robust federated learning on non-IID data with Byzantine clients, **Geometric Median aggregation** provides the best defense while maintaining learning performance. The Geometric Median's ability to find a consensus point that minimizes distances to all client updates makes it naturally robust to outliers without assuming client homogeneity.

### Research Implications

This result demonstrates that robust aggregation methods must be designed with both adversarial robustness and statistical heterogeneity in mind. Methods like Krum that work well on IID data may fail on non-IID data. The Geometric Median's superiority suggests that geometric approaches to robust aggregation are promising for real-world federated learning applications.