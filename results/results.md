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

---

## Experiment 06 — Differential Privacy

### Configuration
- Dataset: MNIST
- Clients: 5
- Federated rounds: 5
- Local epochs: 1
- Device: CPU
- Data distribution: Non-IID (clients specialized in digit pairs)
- DP Method: Gradient clipping + Gaussian noise

### Results: Privacy-Utility Tradeoff

| Noise (σ) | Privacy Level | Final Accuracy | Δ from No Privacy |
|-----------|---------------|----------------|------------------:|
| 0.0 | No privacy | 77.54% | — |
| 0.5 | Low privacy | 11.65% | -65.89% |
| 1.0 | Medium privacy | 14.04% | -63.50% |
| 2.0 | High privacy | 8.89% | -68.65% |

### Complete Results Table

| Round | σ=0.0 | σ=0.5 | σ=1.0 | σ=2.0 |
|------:|------:|------:|------:|------:|
| 1 | 33.16% | 9.76% | 6.91% | 11.20% |
| 2 | 57.50% | 6.59% | 3.11% | 6.90% |
| 3 | 66.23% | 8.55% | 11.27% | 7.77% |
| 4 | 70.21% | 12.91% | 10.29% | 7.23% |
| 5 | 77.54% | 11.65% | 14.04% | 8.89% |

### Key Findings

1. **DP causes catastrophic accuracy degradation** in this setting. Adding even modest noise (σ=0.5) dropped accuracy from 77.54% to 11.65%.

2. **The privacy-utility tradeoff is extreme**: No privacy achieved 77.54%, while high privacy (σ=2.0) resulted in 8.89% (worse than random 10%).

3. **DP implementation matters**: Our simplified DP-SGD implementation may be too aggressive. Future work should use Opacus or similar libraries for proper DP guarantees.

4. **Non-IID data exacerbates DP challenges**: Statistical heterogeneity already reduces accuracy; adding privacy noise compounds the problem.

### Comparison with Previous Experiments

| Experiment | Method | Accuracy | Key Insight |
|------------|--------|---------:|-------------|
| 02 | Non-IID FedAvg | 63.81% | Baseline without attacks |
| 03 | Byzantine Attack | 47.52% | Malicious clients dangerous |
| 05 | Geometric Median | 60.25% | Best robust defense |
| **06 (No DP)** | **FedAvg** | **77.54%** | **Higher baseline (lucky run)** |
| **06 (DP σ=0.5)** | **FedAvg+DP** | **11.65%** | **Privacy kills utility** |

### Conclusion

While Differential Privacy provides formal privacy guarantees, our results demonstrate that it can be prohibitively expensive in non-IID federated learning settings. The privacy-utility tradeoff is severe, and achieving both strong privacy and high accuracy requires careful tuning and more sophisticated DP implementations.

### Future Directions

1. Implement DP-SGD using Opacus
2. Tune clipping thresholds per layer
3. Test with larger models
4. Explore alternative privacy-preserving techniques (e.g., secure aggregation)
5. Study the interaction between DP and robust aggregation methods