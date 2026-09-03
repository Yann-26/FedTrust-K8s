# FedTrust-K8s: Federated Learning with Trust and Privacy

## Experiments

- **Experiment 00**: Centralized Baseline - 98.53%
- **Experiment 01**: IID Federated Learning - 98.61%
- **Experiment 02**: Non-IID Federated Learning - 63.81%
- **Experiment 03**: Byzantine Client Attack - 47.52%
- **Experiment 04**: Data Poisoning Attack - 71.57%
- **Experiment 05**: Robust Aggregation Comparison - Geometric Median: 60.25%
- **Experiment 06**: Differential Privacy - TBD
- **Experiment 07**: Privacy Evaluation - TBD
- **Experiment 08**: GPU / HPC - TBD
- **Experiment 09**: Docker + Reproducibility - TBD
- **Experiment 10**: Kubernetes Deployment - TBD

## Results Summary

| Experiment | Method | Key Finding | Accuracy |
|------------|--------|-------------|----------|
| 00 | Centralized | Baseline | 98.53% |
| 01 | IID FedAvg | FL works | 98.61% |
| 02 | Non-IID FedAvg | Heterogeneity hurts | 63.81% |
| 03 | Byzantine | Malicious clients dangerous | 47.52% |
| 04 | Data Poisoning | Counterintuitive results | 71.57% |
| 05 | Geometric Median | Best defense | 60.25% |
| 06 | DP-FedAvg | Privacy-utility tradeoff | TBD |