"""
Metrics Module
==============

Evaluation metrics for QCBM performance.
"""

import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass

from src.training.loss_functions import (
    mmd_loss,
    kl_divergence,
    total_variation_distance,
    hellinger_distance,
    fidelity
)


@dataclass
class MetricsResult:
    """Container for evaluation metrics."""
    mmd: float
    kl_divergence: float
    total_variation: float
    hellinger: float
    fidelity: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'MMD': self.mmd,
            'KL Divergence': self.kl_divergence,
            'Total Variation': self.total_variation,
            'Hellinger Distance': self.hellinger,
            'Fidelity': self.fidelity
        }
    
    def __repr__(self) -> str:
        return (
            f"MetricsResult(\n"
            f"  MMD: {self.mmd:.6f}\n"
            f"  KL Divergence: {self.kl_divergence:.6f}\n"
            f"  Total Variation: {self.total_variation:.6f}\n"
            f"  Hellinger Distance: {self.hellinger:.6f}\n"
            f"  Fidelity: {self.fidelity:.6f}\n"
            f")"
        )


def compute_all_metrics(
    model_dist: np.ndarray,
    target_dist: np.ndarray,
    mmd_bandwidth: float = 0.1
) -> MetricsResult:
    """
    Compute all evaluation metrics between model and target distributions.
    
    Parameters
    ----------
    model_dist : ndarray
        Model (QCBM) probability distribution.
    target_dist : ndarray
        Target probability distribution.
    mmd_bandwidth : float
        Bandwidth for MMD RBF kernel.
    
    Returns
    -------
    MetricsResult
        All computed metrics.
    
    Example
    -------
    >>> model = np.array([0.4, 0.3, 0.2, 0.1])
    >>> target = np.array([0.25, 0.25, 0.25, 0.25])
    >>> metrics = compute_all_metrics(model, target)
    >>> print(metrics)
    """
    return MetricsResult(
        mmd=mmd_loss(model_dist, target_dist, bandwidth=mmd_bandwidth),
        kl_divergence=kl_divergence(model_dist, target_dist),
        total_variation=total_variation_distance(model_dist, target_dist),
        hellinger=hellinger_distance(model_dist, target_dist),
        fidelity=fidelity(model_dist, target_dist)
    )


def format_metrics_table(
    metrics_dict: Dict[str, MetricsResult],
    highlight_best: bool = True
) -> str:
    """
    Format multiple experiment metrics as a comparison table.
    
    Parameters
    ----------
    metrics_dict : dict
        Dictionary mapping experiment names to MetricsResult.
    highlight_best : bool
        Whether to highlight best values.
    
    Returns
    -------
    str
        Formatted table string.
    """
    if not metrics_dict:
        return "No metrics to display."
    
    # Get all metric names
    metric_names = ['MMD', 'KL Divergence', 'Total Variation', 'Hellinger Distance', 'Fidelity']
    exp_names = list(metrics_dict.keys())
    
    # Build table
    header = "| Experiment | " + " | ".join(metric_names) + " |"
    separator = "|" + "|".join(["-" * 15] * (len(metric_names) + 1)) + "|"
    
    rows = []
    for exp_name in exp_names:
        metrics = metrics_dict[exp_name]
        values = metrics.to_dict()
        row = f"| {exp_name:13} | "
        row += " | ".join([f"{values[m]:.6f}" for m in metric_names])
        row += " |"
        rows.append(row)
    
    return "\n".join([header, separator] + rows)


def compute_convergence_speed(
    loss_history: list,
    threshold: float = 0.1
) -> Optional[int]:
    """
    Compute the iteration at which loss first drops below threshold.
    
    Parameters
    ----------
    loss_history : list
        Training loss history.
    threshold : float
        Loss threshold for convergence.
    
    Returns
    -------
    int or None
        Iteration of convergence, or None if not converged.
    """
    for i, loss in enumerate(loss_history):
        if loss < threshold:
            return i
    return None


def compute_stability(loss_history: list, window: int = 20) -> float:
    """
    Compute training stability as the std of loss in the final window.
    
    Lower values indicate more stable training.
    
    Parameters
    ----------
    loss_history : list
        Training loss history.
    window : int
        Number of final iterations to consider.
    
    Returns
    -------
    float
        Standard deviation of loss in final window.
    """
    if len(loss_history) < window:
        window = len(loss_history)
    
    final_losses = loss_history[-window:]
    return float(np.std(final_losses))


def compute_entropy(distribution: np.ndarray) -> float:
    """
    Compute Shannon entropy of a distribution.
    
    Parameters
    ----------
    distribution : ndarray
        Probability distribution.
    
    Returns
    -------
    float
        Entropy in bits.
    """
    eps = 1e-10
    p = distribution + eps
    return float(-np.sum(p * np.log2(p)))


def compute_effective_dimension(distribution: np.ndarray, threshold: float = 0.01) -> int:
    """
    Compute effective dimension (number of states with significant probability).
    
    Parameters
    ----------
    distribution : ndarray
        Probability distribution.
    threshold : float
        Minimum probability to count as significant.
    
    Returns
    -------
    int
        Number of significant states.
    """
    return int(np.sum(distribution > threshold))
