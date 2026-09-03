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
