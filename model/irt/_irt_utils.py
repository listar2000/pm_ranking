from typing import Dict

import torch
import matplotlib.pyplot as plt

def plot_posterior_samples(posterior_samples: Dict[str, torch.Tensor], base_path: str):
    """
    Plot histograms of posterior samples for theta, a, b, and p parameters.
    Creates 4 separate PNG files.
    
    Args:
        posterior_samples: Dictionary of posterior samples from MCMC
    """
    base_path = "/net/scratch2/listar2000/pm_ranking/model/irt/images"
    
    # Plot theta samples (first 10)
    if 'theta' in posterior_samples:
        plt.figure(figsize=(10, 6))
        theta_samples = posterior_samples['theta'].detach().cpu().numpy()
        for i in range(min(10, theta_samples.shape[1])):
            plt.hist(theta_samples[:, i], bins=30, alpha=0.7, label=f'θ_{i}')
        plt.title('Theta Parameters (First 10)', fontsize=14, fontweight='bold')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{base_path}/theta_samples.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Theta samples plot saved to: {base_path}/theta_samples.png")
    
    # Plot a samples (first 10)
    if 'a' in posterior_samples:
        plt.figure(figsize=(10, 6))
        a_samples = posterior_samples['a'].detach().cpu().numpy()
        for i in range(min(10, a_samples.shape[1])):
            plt.hist(a_samples[:, i], bins=30, alpha=0.7, label=f'a_{i}')
        plt.title('Discrimination Parameters (First 10)', fontsize=14, fontweight='bold')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{base_path}/a_samples.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Discrimination samples plot saved to: {base_path}/a_samples.png")
    
    # Plot b samples (first 10)
    if 'b' in posterior_samples:
        plt.figure(figsize=(10, 6))
        b_samples = posterior_samples['b'].detach().cpu().numpy()
        for i in range(min(10, b_samples.shape[1])):
            plt.hist(b_samples[:, i], bins=30, alpha=0.7, label=f'b_{i}')
        plt.title('Difficulty Parameters (First 10)', fontsize=14, fontweight='bold')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{base_path}/b_samples.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Difficulty samples plot saved to: {base_path}/b_samples.png")
    
    # Plot p samples (all)
    if 'p' in posterior_samples:
        plt.figure(figsize=(10, 6))
        p_samples = posterior_samples['p'].detach().cpu().numpy()
        for i in range(p_samples.shape[1]):
            plt.hist(p_samples[:, i], bins=30, alpha=0.7, label=f'p_{i}')
        plt.title('Category Parameters (All)', fontsize=14, fontweight='bold')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{base_path}/p_samples.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Category samples plot saved to: {base_path}/p_samples.png")