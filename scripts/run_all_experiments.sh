#!/bin/bash
# Run all experiments sequentially (useful for reproducibility)

set -e

echo "=================================================="
echo "FedTrust-K8s: Running All Experiments"
echo "=================================================="

EXPERIMENTS=(
    "00_baseline.py"
    "01_federated_fedavg.py"
    "02_non_iid_fedavg.py"
    "03_byzantine_client.py"
    "04_data_poisoning.py"
    "05_robust_aggregation.py"
    "06_differential_privacy.py"
    "07_privacy_evaluation.py"
    "08_gpu_hpc_scaling.py"
)

for exp in "${EXPERIMENTS[@]}"; do
    echo ""
    echo "=================================================="
    echo "Running: $exp"
    echo "=================================================="
    
    docker run --rm \
        -v $(pwd)/datasets:/app/datasets \
        -v $(pwd)/results:/app/results \
        fedtrust-k8s:latest \
        experiments/$exp
    
    echo ""
    echo "✅ Completed: $exp"
    echo "=================================================="
done

echo ""
echo "=================================================="
echo "✅ All experiments completed!"
echo "📊 Results saved in ./results/"
echo "=================================================="
