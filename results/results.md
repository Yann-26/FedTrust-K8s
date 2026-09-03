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