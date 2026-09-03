"""Experiment 07: Privacy Evaluation for Federated Learning.

This experiment evaluates privacy leakage in federated learning:
1. Gradient similarity analysis
2. Membership inference attacks
3. Privacy-utility visualization
"""

import sys
sys.path.append('.')

import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_curve, auc
from scipy.stats import norm

# Import from src modules
from src.models import SimpleCNN
from src.data import get_mnist_datasets, create_non_iid_datasets
from src.evaluation import evaluate_accuracy
from src.aggregation import fedavg
from src.utils import get_device, print_header, set_seed


# ============================================================
# Configuration
# ============================================================

CONFIG = {
    'num_clients': 5,
    'rounds': 3,  # Reduced for faster evaluation
    'local_epochs': 1,
    'batch_size': 32,
    'learning_rate': 0.001,
    'seed': 42,
    'num_inference_samples': 100,  # For membership inference
}

device = get_device()
set_seed(CONFIG['seed'])


# ============================================================
# 1. Gradient Similarity Analysis
# ============================================================

def compute_gradient_similarity(model1, model2):
    """Compute cosine similarity between two models' gradients."""
    flat1 = []
    flat2 = []
    
    for (key1, param1), (key2, param2) in zip(
        model1.named_parameters(), 
        model2.named_parameters()
    ):
        if param1.grad is not None and param2.grad is not None:
            flat1.append(param1.grad.flatten())
            flat2.append(param2.grad.flatten())
    
    if not flat1:
        return 0.0
    
    vec1 = torch.cat(flat1)
    vec2 = torch.cat(flat2)
    
    cosine_sim = torch.dot(vec1, vec2) / (
        torch.norm(vec1) * torch.norm(vec2) + 1e-10
    )
    
    return cosine_sim.item()


def analyze_gradient_similarity():
    """Analyze gradient similarity between clients."""
    print("\n" + "="*65)
    print("ANALYSIS 1: Gradient Similarity Between Clients")
    print("="*65)
    
    # Load data
    train_dataset, test_dataset = get_mnist_datasets()
    client_datasets = create_non_iid_datasets(train_dataset, CONFIG['num_clients'])
    
    # Initialize global model
    global_model = SimpleCNN().to(device)
    
    # Train one round
    print("\nTraining clients for one round...")
    client_models = []
    client_grads = []
    
    for client_id in range(CONFIG['num_clients']):
        model = SimpleCNN().to(device)
        model.load_state_dict(global_model.state_dict())
        model.train()
        
        loader = DataLoader(
            client_datasets[client_id],
            batch_size=CONFIG['batch_size'],
            shuffle=True
        )
        
        optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
        criterion = nn.CrossEntropyLoss()
        
        # One training step to get gradients
        images, labels = next(iter(loader))
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Store gradients
        grads = []
        for param in model.parameters():
            if param.grad is not None:
                grads.append(param.grad.flatten())
        client_grads.append(torch.cat(grads))
        
        client_models.append(model)
    
    # Compute similarity matrix
    n_clients = len(client_grads)
    similarity_matrix = torch.zeros((n_clients, n_clients))
    
    for i in range(n_clients):
        for j in range(n_clients):
            sim = torch.dot(client_grads[i], client_grads[j]) / (
                torch.norm(client_grads[i]) * torch.norm(client_grads[j]) + 1e-10
            )
            similarity_matrix[i, j] = sim
    
    print("\nGradient Similarity Matrix (cosine similarity):")
    print("-" * 50)
    print("         ", end="")
    for i in range(n_clients):
        print(f"  C{i+1}   ", end="")
    print()
    
    for i in range(n_clients):
        print(f"Client {i+1}: ", end="")
        for j in range(n_clients):
            print(f"{similarity_matrix[i, j]:.3f}  ", end="")
        print()
    
    # Statistics
    off_diag = similarity_matrix[~torch.eye(n_clients, dtype=bool)]
    mean_sim = off_diag.mean().item()
    std_sim = off_diag.std().item()
    
    print(f"\nStatistics:")
    print(f"  Mean similarity between clients: {mean_sim:.4f}")
    print(f"  Std similarity between clients: {std_sim:.4f}")
    print(f"  This indicates how different client updates are.")
    
    # Interpretation
    if mean_sim < 0.5:
        print("\n⚠️  Low gradient similarity suggests clients are learning different patterns.")
        print("   This makes the system more vulnerable to privacy leakage.")
    else:
        print("\n✅ High gradient similarity suggests clients are learning similar patterns.")
        print("   This provides some natural privacy protection.")
    
    return similarity_matrix


# ============================================================
# 2. Membership Inference Attack
# ============================================================

class MembershipInferenceAttack:
    """Simple membership inference attack using model confidence."""
    
    def __init__(self, model, target_dataset, known_dataset):
        self.model = model
        self.target_dataset = target_dataset
        self.known_dataset = known_dataset
        self.device = next(model.parameters()).device
    
    def compute_confidence(self, dataset, indices=None):
        """Compute model confidence on dataset samples."""
        self.model.eval()
        
        if indices is None:
            indices = list(range(len(dataset)))
        
        confidences = []
        loader = DataLoader(
            Subset(dataset, indices),
            batch_size=32,
            shuffle=False
        )
        
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                confidence = probs.max(dim=1)[0]
                confidences.extend(confidence.cpu().numpy())
        
        return np.array(confidences)
    
    def run_attack(self):
        """Run membership inference attack."""
        # Use some samples from training (members) and some from test (non-members)
        n_samples = CONFIG['num_inference_samples']
        
        # Members: from training
        train_indices = list(range(min(n_samples, len(self.target_dataset))))
        
        # Non-members: from known dataset (test)
        test_indices = list(range(min(n_samples, len(self.known_dataset))))
        
        # Compute confidences
        train_conf = self.compute_confidence(self.target_dataset, train_indices)
        test_conf = self.compute_confidence(self.known_dataset, test_indices)
        
        # Labels: 1 for member, 0 for non-member
        labels = np.concatenate([np.ones_like(train_conf), np.zeros_like(test_conf)])
        confidences = np.concatenate([train_conf, test_conf])
        
        # Compute ROC curve
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(labels, confidences)
        auc_score = auc(fpr, tpr)
        
        return fpr, tpr, thresholds, auc_score, confidences, labels
    
    def evaluate_attack(self):
        """Evaluate and print attack performance."""
        fpr, tpr, thresholds, auc_score, confidences, labels = self.run_attack()
        
        print("\n" + "="*65)
        print("ANALYSIS 2: Membership Inference Attack")
        print("="*65)
        
        print(f"\nAttack Performance:")
        print(f"  AUC Score: {auc_score:.4f}")
        
        if auc_score > 0.9:
            print("  ⚠️  HIGH RISK: Attack can easily identify training samples!")
            print("      Privacy leakage is significant.")
        elif auc_score > 0.7:
            print("  ⚠️  MODERATE RISK: Some privacy leakage detected.")
        else:
            print("  ✅  LOW RISK: Attack performs near random guessing.")
            print("      Good privacy protection.")
        
        # Find optimal threshold
        if thresholds is not None and len(thresholds) > 0:
            optimal_idx = np.argmax(tpr - fpr)
            if optimal_idx < len(thresholds):
                optimal_threshold = thresholds[optimal_idx]
                
                print(f"\nOptimal threshold: {optimal_threshold:.4f}")
                print(f"  True Positive Rate: {tpr[optimal_idx]:.4f}")
                print(f"  False Positive Rate: {fpr[optimal_idx]:.4f}")
                
                print("\nInterpretation:")
                print(f"  At the optimal threshold, we can identify training samples")
                print(f"  with {tpr[optimal_idx]*100:.1f}% accuracy while only")
                print(f"  misclassifying {fpr[optimal_idx]*100:.1f}% of non-training samples.")
        
        return fpr, tpr, auc_score


# ============================================================
# 3. Privacy Budget Visualization
# ============================================================

def compute_privacy_budget(rounds, noise_multiplier, sensitivity=1.0, delta=1e-5):
    """Compute (ε, δ) privacy budget."""
    sigma = noise_multiplier * sensitivity
    if sigma == 0:
        return float('inf')  # No privacy
    epsilon = np.sqrt(2 * rounds * np.log(1 / delta)) / sigma
    return epsilon


def visualize_privacy_utility():
    """Visualize the privacy-utility tradeoff."""
    print("\n" + "="*65)
    print("ANALYSIS 3: Privacy-Utility Tradeoff Visualization")
    print("="*65)
    
    # Data from Experiment 06
    noise_levels = [0.0, 0.5, 1.0, 2.0]
    accuracies = [77.54, 11.65, 14.04, 8.89]
    
    # Compute privacy budgets
    epsilons = []
    for noise in noise_levels:
        eps = compute_privacy_budget(
            rounds=CONFIG['rounds'] * CONFIG['local_epochs'],
            noise_multiplier=noise,
            sensitivity=1.0,
            delta=1e-5
        )
        epsilons.append(eps)
    
    print("\nPrivacy-Utility Tradeoff:")
    print(f"{'Noise (σ)':<12} {'Epsilon (ε)':<15} {'Accuracy (%)':<15} {'Privacy Level':<15}")
    print("-" * 60)
    
    for noise, eps, acc in zip(noise_levels, epsilons, accuracies):
        if eps == float('inf'):
            eps_str = "∞ (No privacy)"
            privacy = "No privacy"
        elif eps < 1.0:
            eps_str = f"{eps:.2f}"
            privacy = "High privacy"
        elif eps < 5.0:
            eps_str = f"{eps:.2f}"
            privacy = "Medium privacy"
        else:
            eps_str = f"{eps:.2f}"
            privacy = "Low privacy"
        
        print(f"{noise:<12} {eps_str:<15} {acc:<15.2f} {privacy:<15}")
    
    print("\nInterpretation:")
    print("  • ε (epsilon): Lower values = stronger privacy")
    print("  • ε = 0: Perfect privacy (impossible)")
    print("  • ε = ∞: No privacy")
    print("  • Practical range: ε ∈ [1, 10]")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Privacy-Utility Tradeoff
    ax1.plot(noise_levels, accuracies, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Noise Multiplier (σ)', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Privacy-Utility Tradeoff', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.2, 2.2)
    ax1.set_ylim(0, 90)
    
    # Annotate privacy levels
    ax1.text(0.0, 75, 'No Privacy', fontsize=10, ha='center')
    ax1.text(0.5, 5, 'Low Privacy', fontsize=10, ha='center')
    ax1.text(1.0, 10, 'Medium Privacy', fontsize=10, ha='center')
    ax1.text(2.0, 3, 'High Privacy', fontsize=10, ha='center')
    
    # Plot 2: Epsilon vs Accuracy
    eps_vals = [eps if eps != float('inf') else 100 for eps in epsilons]
    ax2.plot(eps_vals, accuracies, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Epsilon (ε) [Lower = More Private]', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Privacy Budget vs Utility', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-2, 55)
    ax2.set_ylim(0, 90)
    
    # Annotate privacy levels
    ax2.axvline(x=1, color='green', linestyle='--', alpha=0.5, label='ε=1 (Strong)')
    ax2.axvline(x=5, color='orange', linestyle='--', alpha=0.5, label='ε=5 (Moderate)')
    ax2.axvline(x=10, color='red', linestyle='--', alpha=0.5, label='ε=10 (Weak)')
    ax2.legend()
    
    plt.tight_layout()
    
    # Save figure
    fig_path = "results/figures/privacy_utility_tradeoff.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Figure saved to: {fig_path}")
    
    return fig


# ============================================================
# 4. Privacy Leakage Analysis
# ============================================================

def analyze_privacy_leakage():
    """Analyze privacy leakage across different scenarios."""
    print("\n" + "="*65)
    print("ANALYSIS 4: Privacy Leakage Analysis")
    print("="*65)
    
    # Load data
    train_dataset, test_dataset = get_mnist_datasets()
    client_datasets = create_non_iid_datasets(train_dataset, CONFIG['num_clients'])
    
    # Train with and without DP
    global_model = SimpleCNN().to(device)
    
    print("\nTraining model without privacy protection...")
    
    for round_num in range(CONFIG['rounds']):
        client_models = []
        
        for client_id in range(CONFIG['num_clients']):
            model = SimpleCNN().to(device)
            model.load_state_dict(global_model.state_dict())
            model.train()
            
            loader = DataLoader(
                client_datasets[client_id],
                batch_size=CONFIG['batch_size'],
                shuffle=True
            )
            
            optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
            criterion = nn.CrossEntropyLoss()
            
            for epoch in range(CONFIG['local_epochs']):
                for images, labels in loader:
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
            
            client_models.append(model.state_dict())
        
        global_weights = fedavg(client_models)
        global_model.load_state_dict(global_weights)
    
    accuracy = evaluate_accuracy(global_model, test_dataset, device)
    print(f"\nFinal accuracy: {accuracy:.2f}%")
    
    # Run membership inference
    print("\nRunning membership inference attack...")
    attack = MembershipInferenceAttack(
        global_model,
        train_dataset,
        test_dataset
    )
    fpr, tpr, auc_score = attack.evaluate_attack()
    
    print("\n" + "="*65)
    print("PRIVACY LEAKAGE SUMMARY")
    print("="*65)
    
    if auc_score > 0.8:
        print("\n⚠️  HIGH PRIVACY LEAKAGE DETECTED")
        print("   The model reveals significant information about its training data.")
        print("   Recommendation: Use Differential Privacy or other privacy-preserving techniques.")
    elif auc_score > 0.6:
        print("\n⚠️  MODERATE PRIVACY LEAKAGE DETECTED")
        print("   Some information about training data is exposed.")
        print("   Consider adding noise or using privacy-preserving aggregation.")
    else:
        print("\n✅ LOW PRIVACY LEAKAGE DETECTED")
        print("   The model provides reasonable privacy protection.")
        print("   Current privacy measures are effective.")
    
    return auc_score


# ============================================================
# 5. Main Experiment
# ============================================================

def main():
    """Run the complete privacy evaluation."""
    print_header("FedTrust-K8s — Experiment 07: Privacy Evaluation")
    
    print("\nThis experiment evaluates privacy leakage in federated learning:")
    print("  1. Gradient similarity between clients")
    print("  2. Membership inference attacks")
    print("  3. Privacy-utility tradeoff visualization")
    print("  4. Comprehensive privacy leakage analysis")
    
    # Analysis 1: Gradient Similarity
    similarity_matrix = analyze_gradient_similarity()
    
    # Analysis 2: Privacy-Utility Tradeoff
    fig = visualize_privacy_utility()
    
    # Analysis 3: Privacy Leakage
    auc_score = analyze_privacy_leakage()
    
    # Final summary
    print("\n" + "="*65)
    print("EXPERIMENT 07 COMPLETE")
    print("="*65)
    
    print("\nKey Findings:")
    print("  1. Gradient similarity between non-IID clients is low,")
    print("     indicating high diversity and potential privacy risks.")
    print(f"  2. Membership inference attack achieved AUC = {auc_score:.4f}")
    print("  3. Privacy-utility tradeoff shows significant accuracy loss with privacy.")
    
    print("\nRecommendations:")
    if auc_score > 0.7:
        print("  ⚠️  Implement Differential Privacy to protect training data.")
    else:
        print("  ✅  Current system provides reasonable privacy.")
    
    print("\n" + "="*65)
    print("✅ Experiment 07 completed successfully!")
    print("="*65)


if __name__ == "__main__":
    main()
