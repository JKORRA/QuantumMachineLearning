"""
Loss Functions Module
=====================

Loss functions for training QCBM to match target distributions.
"""

import numpy as np
from typing import Callable, Optional
from enum import Enum


class LossFunction(Enum):
    """Available loss functions."""
    MMD = 'mmd'
    KL_DIVERGENCE = 'kl'
    TOTAL_VARIATION = 'tv'
    HELLINGER = 'hellinger'


def mmd_loss(
    p: np.ndarray,
    q: np.ndarray,
    bandwidth: float = 0.1
) -> float:
    """
    Maximum Mean Discrepancy (MMD) loss with RBF kernel.
    
    MMD measures the difference between two probability distributions
    in a Reproducing Kernel Hilbert Space (RKHS).
    
    For distributions p and q:
        MMD²(p, q) = E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]
    
    where k is the RBF kernel: k(x,y) = exp(-||x-y||² / (2σ²))
    
    Parameters
    ----------
    p : ndarray
        Model distribution (from QCBM).
    q : ndarray
        Target distribution.
    bandwidth : float
        RBF kernel bandwidth (σ).
    
    Returns
    -------
    float
        MMD squared loss value.
    
    Notes
    -----
    MMD is preferred over KL divergence for QCBM training because:
    1. It's symmetric
    2. It doesn't require support overlap (no log(0) issues)
    3. It's smooth and well-behaved for optimization
    
    Example
    -------
    >>> p = np.array([0.5, 0.3, 0.2])
    >>> q = np.array([0.33, 0.33, 0.34])
    >>> loss = mmd_loss(p, q)
    """
    n = len(p)
    
    # Create state indices for computing pairwise distances
    indices = np.arange(n)
    
    # Compute pairwise distances (for discrete states, distance = |i - j|)
    # Using toroidal distance to respect periodicity if applicable
    dists = np.abs(indices[:, None] - indices[None, :])
    
    # RBF kernel matrix
    K = np.exp(-dists ** 2 / (2 * bandwidth ** 2 * n ** 2))
    
    # MMD^2 = E_p[K] + E_q[K] - 2 E_pq[K]
    # = p^T K p + q^T K q - 2 p^T K q
    mmd_sq = p @ K @ p + q @ K @ q - 2 * p @ K @ q
    
    return float(mmd_sq)


def mmd_loss_multiscale(
    p: np.ndarray,
    q: np.ndarray,
    bandwidths: Optional[list] = None
) -> float:
    """
    Multi-scale MMD loss using multiple kernel bandwidths.
    
    Combines multiple RBF kernels at different scales for more
    robust distribution matching.
    
    Parameters
    ----------
    p : ndarray
        Model distribution.
    q : ndarray
        Target distribution.
    bandwidths : list of float, optional
        Kernel bandwidths. Default: [0.01, 0.1, 1.0].
    
    Returns
    -------
    float
        Combined MMD loss.
    """
    if bandwidths is None:
        bandwidths = [0.01, 0.1, 1.0]
    
    total_loss = 0.0
    for bw in bandwidths:
        total_loss += mmd_loss(p, q, bandwidth=bw)
    
    return total_loss / len(bandwidths)


def kl_divergence(
    p: np.ndarray,
    q: np.ndarray,
    epsilon: float = 1e-10
) -> float:
    """
    Kullback-Leibler (KL) Divergence.
    
    Measures how distribution p diverges from target distribution q:
        KL(p || q) = Σ p(x) log(p(x) / q(x))
    
    Parameters
    ----------
    p : ndarray
        Model distribution.
    q : ndarray
        Target distribution.
    epsilon : float
        Small constant for numerical stability.
    
    Returns
    -------
    float
        KL divergence value.
    
    Warning
    -------
    KL divergence can be problematic when:
    - p has support where q is zero (results in infinity)
    - p and q have disjoint supports
    
    Consider using MMD for QCBM training instead.
    """
    # Add epsilon for numerical stability
    p_safe = np.maximum(p, epsilon)
    q_safe = np.maximum(q, epsilon)
    
    # Renormalize
    p_safe = p_safe / p_safe.sum()
    q_safe = q_safe / q_safe.sum()
    
    return float(np.sum(p_safe * np.log(p_safe / q_safe)))


def reverse_kl_divergence(
    p: np.ndarray,
    q: np.ndarray,
    epsilon: float = 1e-10
) -> float:
    """
    Reverse KL Divergence: KL(q || p).
    
    Measures how target q diverges from model p.
    This is mode-seeking (p tries to cover high-probability regions of q).
    
    Parameters
    ----------
    p : ndarray
        Model distribution.
    q : ndarray
        Target distribution.
    epsilon : float
        Small constant for numerical stability.
    
    Returns
    -------
    float
        Reverse KL divergence value.
    """
    return kl_divergence(q, p, epsilon)


def total_variation_distance(p: np.ndarray, q: np.ndarray) -> float:
    """
    Total Variation (TV) Distance.
    
    The maximum difference in probability assigned to any event:
        TV(p, q) = 0.5 * Σ |p(x) - q(x)|
    
    Parameters
    ----------
    p : ndarray
        Model distribution.
    q : ndarray
        Target distribution.
    
    Returns
    -------
    float
        TV distance (between 0 and 1).
    """
    return float(0.5 * np.sum(np.abs(p - q)))


def hellinger_distance(
    p: np.ndarray,
    q: np.ndarray
) -> float:
    """
    Hellinger Distance.
    
    A symmetric measure of distribution similarity:
        H(p, q) = (1/√2) * √(Σ (√p(x) - √q(x))²)
    
    Parameters
    ----------
    p : ndarray
        Model distribution.
    q : ndarray
        Target distribution.
    
    Returns
    -------
    float
        Hellinger distance (between 0 and 1).
    """
    return float(np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)))


def fidelity(p: np.ndarray, q: np.ndarray) -> float:
    """
    Classical Fidelity (Bhattacharyya coefficient).
    
    Measures the overlap between two distributions:
        F(p, q) = Σ √(p(x) * q(x))
    
    Perfect match → F = 1
    No overlap → F = 0
    
    Parameters
    ----------
    p : ndarray
        Model distribution.
    q : ndarray
        Target distribution.
    
    Returns
    -------
    float
        Fidelity value (between 0 and 1).
    """
    return float(np.sum(np.sqrt(p * q)))


def get_loss_function(name: str, **kwargs) -> Callable:
    """
    Get a loss function by name.
    
    Parameters
    ----------
    name : str
        Loss function name: 'mmd', 'kl', 'tv', 'hellinger'.
    **kwargs
        Additional arguments for the loss function.
    
    Returns
    -------
    callable
        Loss function that takes (p, q) and returns float.
    """
    loss_functions = {
        'mmd': lambda p, q: mmd_loss(p, q, **kwargs),
        'mmd_multiscale': lambda p, q: mmd_loss_multiscale(p, q, **kwargs),
        'kl': lambda p, q: kl_divergence(p, q, **kwargs),
        'reverse_kl': lambda p, q: reverse_kl_divergence(p, q, **kwargs),
        'tv': total_variation_distance,
        'hellinger': hellinger_distance,
    }
    
    if name not in loss_functions:
        raise ValueError(f"Unknown loss function: {name}. Available: {list(loss_functions.keys())}")
    
    return loss_functions[name]
