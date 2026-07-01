"""
Visualization Module
====================

Professional plotting functions for QCBM experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path

# Import config for colors
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import COLORS, FIGURE_DPI, FIGURE_SIZE_DEFAULT, FIGURE_SIZE_WIDE


def set_style():
    """Set the plotting style for consistent professional figures."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': FIGURE_DPI,
        'savefig.dpi': FIGURE_DPI,
        'savefig.bbox': 'tight',
    })


def plot_distribution(
    distribution: np.ndarray,
    title: str = "Probability Distribution",
    ax: Optional[plt.Axes] = None,
    color: str = COLORS['quantum'],
    xlabel: str = "State",
    ylabel: str = "Probability",
    show_top_k: int = 0,
    figsize: Tuple[int, int] = FIGURE_SIZE_DEFAULT
) -> plt.Axes:
    """
    Plot a probability distribution as a bar chart.
    
    Parameters
    ----------
    distribution : ndarray
        Probability distribution to plot.
    title : str
        Plot title.
    ax : matplotlib Axes, optional
        Axes to plot on. If None, creates new figure.
    color : str
        Bar color.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    show_top_k : int
        If > 0, annotate top k states with their indices.
    figsize : tuple
        Figure size.
    
    Returns
    -------
    matplotlib Axes
        The axes with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    n_states = len(distribution)
    indices = np.arange(n_states)
    
    ax.bar(indices, distribution, color=color, alpha=0.8, edgecolor='white')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    # Show x-ticks for small distributions
    if n_states <= 32:
        ax.set_xticks(indices)
        if n_states > 16:
            ax.set_xticklabels([f'{i}' for i in indices], rotation=45)
    
    # Annotate top k states
    if show_top_k > 0:
        top_indices = np.argsort(distribution)[-show_top_k:][::-1]
        for idx in top_indices:
            ax.annotate(
                f'{idx}',
                xy=(idx, distribution[idx]),
                xytext=(0, 5),
                textcoords='offset points',
                ha='center',
                fontsize=8
            )
    
    ax.set_xlim(-0.5, n_states - 0.5)
    ax.set_ylim(0, max(distribution) * 1.1)
    
    return ax


def plot_distributions_comparison(
    target: np.ndarray,
    model: np.ndarray,
    title: str = "Target vs Model Distribution",
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = FIGURE_SIZE_DEFAULT,
    target_label: str = "Target",
    model_label: str = "Model"
) -> plt.Axes:
    """
    Plot target and model distributions side by side.
    
    Parameters
    ----------
    target : ndarray
        Target probability distribution.
    model : ndarray
        Model (QCBM) probability distribution.
    title : str
        Plot title.
    ax : matplotlib Axes, optional
        Axes to plot on.
    figsize : tuple
        Figure size.
    target_label : str
        Label for target distribution.
    model_label : str
        Label for model distribution.
    
    Returns
    -------
    matplotlib Axes
        The axes with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    n_states = len(target)
    indices = np.arange(n_states)
    width = 0.35
    
    ax.bar(
        indices - width/2, target, width,
        label=target_label, color=COLORS['target'], alpha=0.8
    )
    ax.bar(
        indices + width/2, model, width,
        label=model_label, color=COLORS['quantum'], alpha=0.8
    )
    
    ax.set_xlabel("State")
    ax.set_ylabel("Probability")
    ax.set_title(title)
    ax.legend()
    
    if n_states <= 32:
        ax.set_xticks(indices)
    
    ax.set_xlim(-0.5, n_states - 0.5)
    
    return ax


def plot_loss_curve(
    loss_history: List[float],
    title: str = "Training Loss",
    ax: Optional[plt.Axes] = None,
    color: str = COLORS['quantum'],
    xlabel: str = "Iteration",
    ylabel: str = "Loss",
    log_scale: bool = False,
    figsize: Tuple[int, int] = FIGURE_SIZE_DEFAULT,
    label: Optional[str] = None
) -> plt.Axes:
    """
    Plot training loss curve.
    
    Parameters
    ----------
    loss_history : list of float
        Loss values over training.
    title : str
        Plot title.
    ax : matplotlib Axes, optional
        Axes to plot on.
    color : str
        Line color.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    log_scale : bool
        Whether to use log scale for y-axis.
    figsize : tuple
        Figure size.
    label : str, optional
        Legend label.
    
    Returns
    -------
    matplotlib Axes
        The axes with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    iterations = np.arange(len(loss_history))
    ax.plot(iterations, loss_history, color=color, linewidth=2, label=label)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    if log_scale:
        ax.set_yscale('log')
    
    if label is not None:
        ax.legend()
    
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_multiple_loss_curves(
    loss_histories: Dict[str, List[float]],
    title: str = "Training Comparison",
    colors: Optional[Dict[str, str]] = None,
    figsize: Tuple[int, int] = FIGURE_SIZE_WIDE,
    log_scale: bool = False
) -> plt.Figure:
    """
    Plot multiple loss curves for comparison.
    
    Parameters
    ----------
    loss_histories : dict
        Dictionary mapping experiment names to loss histories.
    title : str
        Plot title.
    colors : dict, optional
        Dictionary mapping experiment names to colors.
    figsize : tuple
        Figure size.
    log_scale : bool
        Whether to use log scale.
    
    Returns
    -------
    matplotlib Figure
        The figure with the plot.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    default_colors = list(plt.cm.tab10.colors)
    
    for i, (name, history) in enumerate(loss_histories.items()):
        color = colors.get(name) if colors else default_colors[i % len(default_colors)]
        ax.plot(history, label=name, color=color, linewidth=2)
    
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    if log_scale:
        ax.set_yscale('log')
    
    plt.tight_layout()
    return fig


def plot_experiment_comparison(
    results: Dict[str, Any],
    metric: str = 'final_loss',
    title: str = "Experiment Comparison",
    figsize: Tuple[int, int] = FIGURE_SIZE_DEFAULT
) -> plt.Figure:
    """
    Create bar chart comparing experiments on a metric.
    
    Parameters
    ----------
    results : dict
        Dictionary mapping experiment names to TrainingResult objects or dicts.
    metric : str
        Metric to compare: 'final_loss', 'final_fidelity', 'total_time'.
    title : str
        Plot title.
    figsize : tuple
        Figure size.
    
    Returns
    -------
    matplotlib Figure
        The figure with the plot.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    names = list(results.keys())
    values = []
    
    for name, result in results.items():
        if hasattr(result, metric):
            values.append(getattr(result, metric))
        elif isinstance(result, dict) and metric in result:
            values.append(result[metric])
        else:
            values.append(0)
    
    colors = [plt.cm.viridis(i / len(names)) for i in range(len(names))]
    
    bars = ax.bar(names, values, color=colors, edgecolor='white')
    
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(title)
    
    # Rotate labels if many experiments
    if len(names) > 4:
        plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(
            f'{val:.4f}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords='offset points',
            ha='center', va='bottom',
            fontsize=9
        )
    
    plt.tight_layout()
    return fig


def plot_2d_distribution(
    distribution: np.ndarray,
    n_rows: int,
    n_cols: int,
    title: str = "Joint Distribution",
    xlabel: str = "Pitch",
    ylabel: str = "Velocity",
    figsize: Tuple[int, int] = (8, 6),
    cmap: str = 'viridis'
) -> plt.Figure:
    """
    Plot a 2D joint distribution as a heatmap.
    
    Parameters
    ----------
    distribution : ndarray
        1D probability distribution (will be reshaped).
    n_rows : int
        Number of rows (first dimension).
    n_cols : int
        Number of columns (second dimension).
    title : str
        Plot title.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    figsize : tuple
        Figure size.
    cmap : str
        Colormap name.
    
    Returns
    -------
    matplotlib Figure
        The figure with the heatmap.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Reshape to 2D
    dist_2d = distribution.reshape(n_rows, n_cols)
    
    im = ax.imshow(dist_2d, aspect='auto', cmap=cmap, origin='lower')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    plt.colorbar(im, ax=ax, label='Probability')
    
    plt.tight_layout()
    return fig


def save_figure(
    fig: plt.Figure,
    filename: str,
    output_dir: Optional[Path] = None,
    formats: List[str] = ['png', 'pdf']
) -> None:
    """
    Save figure to file in multiple formats.
    
    Parameters
    ----------
    fig : matplotlib Figure
        Figure to save.
    filename : str
        Base filename (without extension).
    output_dir : Path, optional
        Output directory. If None, uses current directory.
    formats : list of str
        File formats to save.
    """
    if output_dir is None:
        output_dir = Path('.')
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    for fmt in formats:
        filepath = output_dir / f"{filename}.{fmt}"
        fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Saved: {filepath}")
