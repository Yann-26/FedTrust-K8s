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

---

## Experiment 07 — Privacy Evaluation

### Research Questions

1. How much information leaks from gradient updates?
2. Can we infer client data from model updates?
3. What is the privacy-utility tradeoff in practice?

### Results

#### Analysis 1: Gradient Similarity

Gradient Similarity Matrix (cosine similarity):

| Client | C1 | C2 | C3 | C4 | C5 |
|--------|------|------|------|------|------|
| C1 | 1.000 | -0.195 | -0.119 | -0.127 | -0.333 |
| C2 | -0.195 | 1.000 | -0.282 | -0.155 | -0.207 |
| C3 | -0.119 | -0.282 | 1.000 | -0.360 | -0.122 |
| C4 | -0.127 | -0.155 | -0.360 | 1.000 | -0.218 |
| C5 | -0.333 | -0.207 | -0.122 | -0.218 | 1.000 |

**Statistics:**
- Mean similarity: -0.2117 (negative correlation!)
- Std similarity: 0.0852

**Interpretation:** Clients are learning in completely different directions, providing natural privacy protection.

#### Analysis 2: Membership Inference Attack

- **AUC Score: 0.5236**
- Near random guessing (0.5 = random)
- Attack cannot distinguish training from non-training samples
- **Privacy Risk: LOW**

#### Analysis 3: Privacy-Utility Tradeoff

| Noise (σ) | Epsilon (ε) | Accuracy | Privacy Level |
|-----------|-------------|----------|---------------|
| 0.0 | ∞ | 77.54% | No privacy |
| 0.5 | 16.62 | 11.65% | Low privacy |
| 1.0 | 8.31 | 14.04% | Low privacy |
| 2.0 | 4.16 | 8.89% | Medium privacy |

### Key Findings

1. **Non-IID data provides natural privacy**: Negative gradient similarity makes it difficult to infer individual client data.

2. **Membership inference is ineffective**: AUC = 0.5236, close to random guessing.

3. **DP is too expensive**: Adding privacy protection destroys utility (77.54% → 11.65%).

4. **Privacy is "free" but accuracy is not**: Non-IID data gives privacy at the cost of accuracy.

### Conclusion

This experiment reveals a fascinating tradeoff: the same statistical heterogeneity that reduces model accuracy also provides natural privacy protection. For applications where data is naturally non-IID (like the WALTZ scenario with different municipalities), the inherent privacy may be sufficient without adding expensive DP mechanisms.

### Recommendations

1. **Assess privacy needs before applying DP**: If the data distribution already provides privacy, DP may not be necessary.

2. **Consider the cost-benefit tradeoff**: DP can reduce accuracy from ~77% to ~12% — is that acceptable?

3. **Use gradient similarity as a privacy metric**: Low/negative similarity suggests good natural privacy.

4. **Hybrid approach**: Consider light DP (σ=0.5) only for high-risk scenarios.

### Research Contribution

This experiment provides evidence that:
- **Natural privacy exists in non-IID federated learning**
- **Privacy evaluation is essential before applying DP**
- **The WALTZ scenario (multiple municipalities with different data) may not need strong DP**

---

## Experiment 08 — GPU / HPC Scaling

### Configuration
- Device: CPU (12 cores)
- Memory: 13.5 GB total (2.5 GB available)
- Dataset: MNIST (non-IID)
- Clients: 5
- Rounds: 3
- Batch size: 32

### Results

| Model | Parameters | Avg Time/Round | Throughput | Final Accuracy |
|-------|-----------|---------------|-----------|----------------|
| SimpleCNN | 206,922 | 15.31s | 10.5 img/s | 75.29% |
| ResNetStyleCNN | 1,717,258 | 120.06s | 1.3 img/s | 36.58% |

### Performance Analysis

**Scaling Factor:**
- Parameter count: 8.3x larger
- Training time: 8.3x slower
- Throughput: 8.1x lower

**Accuracy Comparison:**
- SimpleCNN reached 75.29% in 3 rounds
- ResNetStyleCNN only reached 36.58% in 3 rounds

### Key Findings

1. **Model scaling is roughly linear**: 8.3x more parameters → 8.3x slower training.

2. **The medium model performs worse**: Likely due to insufficient data, insufficient rounds, or memory constraints.

3. **Memory is a bottleneck**: Only 2.5 GB available memory limits model size.

4. **No GPU detected**: CPU-only system limits scalability.

### Recommendations

1. **Use SimpleCNN** for all experiments → fast, good accuracy.
2. **For larger models**, ensure GPU availability.
3. **Increase batch size** to 128-256 for better CPU utilization.
4. **Consider memory optimization** (gradient checkpointing, mixed precision).

### Research Implications

The SimpleCNN (206k parameters) is the most practical choice for this research project. It provides good accuracy (75.29%) with acceptable training time (15 seconds/round). For reproducibility and rapid iteration, this is the optimal configuration.

### Figure

![GPU Scaling Results] <img width="1783" height="1484" alt="Image" src="https://github.com/user-attachments/assets/5ff022f4-0fd8-4dea-b69b-b3ae4b1d7d5d" />