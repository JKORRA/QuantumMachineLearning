"""
Figure Saver Utility
====================

Centralized figure saving for LaTeX report generation.
All figures are saved as high-quality PNG files.
"""

import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import FIGURES_DIR, FIGURE_DPI


def save_figure(
    fig: plt.Figure,
    name: str,
    experiment: Optional[str] = None,
    dpi: int = FIGURE_DPI,
    format: str = 'png',
    tight: bool = True
) -> Path:
    """
    Save a figure to the figures directory.
    
    Parameters
    ----------
    fig : matplotlib Figure
        The figure to save.
    name : str
        Base name for the file (without extension).
    experiment : str, optional
        Experiment prefix (e.g., 'exp01', 'exp06').
    dpi : int
        Resolution in dots per inch.
    format : str
        File format ('png', 'pdf', 'svg').
    tight : bool
        Whether to use tight bounding box.
    
    Returns
    -------
    Path
        Path to the saved figure.
    
    Example
    -------
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3])
    >>> path = save_figure(fig, 'training_loss', experiment='exp03')
    >>> print(f"Saved to: {path}")
    """
    # Build filename
    if experiment:
        filename = f"{experiment}_{name}.{format}"
    else:
        filename = f"{name}.{format}"
    
    filepath = FIGURES_DIR / filename
    
    # Save
    bbox = 'tight' if tight else None
    fig.savefig(filepath, dpi=dpi, format=format, bbox_inches=bbox, 
                facecolor='white', edgecolor='none')
    
    print(f" Figure saved: {filepath}")
    return filepath


def save_current_figure(
    name: str,
    experiment: Optional[str] = None,
    **kwargs
) -> Path:
    """
    Save the current matplotlib figure.
    
    Convenience wrapper around save_figure for the current figure.
    """
    fig = plt.gcf()
    return save_figure(fig, name, experiment, **kwargs)


# Dictionary of all expected figures for the report
REPORT_FIGURES = {
    'exp01': [
        'mario_pitch_histogram',
        'mario_velocity_histogram', 
        'mario_pitch_velocity_scatter',
        'mario_note_duration_histogram',
    ],
    'exp02': [
        'qcbm_circuit_diagram',
        'hardware_efficient_ansatz',
        'entanglement_topologies',
    ],
    'exp03': [
        'baseline_training_curve',
        'baseline_distribution_comparison',
        'separable_vs_entangled',
    ],
    'exp04': [
        'scalability_training_time',
        'scalability_fidelity_comparison',
        '4qubit_vs_8qubit',
    ],
    'exp05': [
        'noise_fidelity_degradation',
        'noise_levels_comparison',
        'noise_threshold_analysis',
    ],
    'exp06': [
        'topology_battle_curves',
        'topology_final_comparison',
        'topology_heatmap',
    ],
    'exp07': [
        'optimizer_battle_curves',
        'optimizer_convergence',
        'optimizer_final_comparison',
    ],
    'exp08': [
        'loss_function_comparison',
        'mmd_vs_kl_training',
        'loss_landscape',
    ],
    'exp09': [
        'final_validation_curves',
        'champion_vs_baseline',
        'final_2d_heatmaps',
        'final_metrics_table',
    ],
    'exp10': [
        'learned_distribution',
        'pitch_histogram_comparison',
        'piano_roll_comparison',
        'generated_samples_analysis',
    ],
}


def list_missing_figures() -> dict:
    """
    Check which figures are missing from the figures directory.
    
    Returns
    -------
    dict
        Dictionary mapping experiment to list of missing figures.
    """
    missing = {}
    
    for exp, figures in REPORT_FIGURES.items():
        exp_missing = []
        for fig_name in figures:
            filepath = FIGURES_DIR / f"{exp}_{fig_name}.png"
            if not filepath.exists():
                exp_missing.append(fig_name)
        
        if exp_missing:
            missing[exp] = exp_missing
    
    return missing


def print_figure_status():
    """Print status of all expected figures."""
    print("\n FIGURE STATUS FOR LATEX REPORT")
    print("=" * 60)
    
    total = 0
    found = 0
    
    for exp, figures in REPORT_FIGURES.items():
        print(f"\n{exp.upper()}:")
        for fig_name in figures:
            total += 1
            filepath = FIGURES_DIR / f"{exp}_{fig_name}.png"
            if filepath.exists():
                print(f"   {fig_name}")
                found += 1
            else:
                print(f"   {fig_name}")
    
    print(f"\n{'=' * 60}")
    print(f"Total: {found}/{total} figures ({100*found/total:.1f}%)")
