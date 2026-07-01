"""
Global Configuration for QCBM Music Generation Project
======================================================

This module contains all hyperparameters and settings used across the project.
Centralizing configuration ensures reproducibility and easy experimentation.
"""

import os
from pathlib import Path

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.absolute()
DATA_DIR = PROJECT_ROOT / "data"
MIDI_DIR = DATA_DIR / "midi"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = RESULTS_DIR / "models"
LOGS_DIR = RESULTS_DIR / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, MIDI_DIR, RESULTS_DIR, FIGURES_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# =============================================================================
# QUANTUM CIRCUIT CONFIGURATION
# =============================================================================

# Number of qubits for different experiments
N_QUBITS_SIMPLE = 4      # 2^4 = 16 states (pitch only, binned)
N_QUBITS_FULL = 8        # 2^8 = 256 states (pitch + velocity)

# Ansatz configuration
N_LAYERS_DEFAULT = 3     # Number of variational layers
ENTANGLEMENT_TOPOLOGIES = ['none', 'linear', 'circular', 'full']

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

# Training iterations for different phases
N_ITERATIONS_EXPLORATION = 200   # Quick exploration experiments
N_ITERATIONS_BATTLE = 300        # Optimization tournaments
N_ITERATIONS_VALIDATION = 500    # Final validation experiments

# Optimizer settings
OPTIMIZER_DEFAULT = 'Powell'
OPTIMIZER_OPTIONS = {
    'COBYLA': {'maxiter': 500, 'rhobeg': 0.5},
    'Powell': {'maxiter': 500},
    'SLSQP': {'maxiter': 500},
    'L-BFGS-B': {'maxiter': 500},
}

# Loss function settings
LOSS_DEFAULT = 'mmd'
MMD_KERNEL_BANDWIDTH = 0.1  # RBF kernel bandwidth for MMD

# =============================================================================
# NOISE CONFIGURATION (NISQ Simulation)
# =============================================================================

NOISE_LEVELS = [0.0, 0.05, 0.10, 0.15]  # Depolarizing noise probabilities
NOISE_DEFAULT = 0.0

# =============================================================================
# DATA ENCODING CONFIGURATION
# =============================================================================

# Pitch binning for 4-qubit encoding
PITCH_MIN = 36   # C2 (typical bass range)
PITCH_MAX = 84   # C6 (typical treble range)
N_PITCH_BINS = 16  # 4 qubits

# Velocity binning for 4-qubit encoding (when using 8 qubits total)
VELOCITY_MIN = 0
VELOCITY_MAX = 127
N_VELOCITY_BINS = 16  # 4 qubits

# =============================================================================
# VISUALIZATION CONFIGURATION
# =============================================================================

# Plot style
PLOT_STYLE = 'seaborn-v0_8-whitegrid'
FIGURE_DPI = 150
FIGURE_SIZE_DEFAULT = (10, 6)
FIGURE_SIZE_WIDE = (14, 6)
FIGURE_SIZE_SQUARE = (8, 8)

# Color palette
COLORS = {
    'quantum': '#6366F1',      # Indigo (primary quantum color)
    'classical': '#10B981',    # Emerald (classical baseline)
    'entangled': '#8B5CF6',    # Purple (entangled circuits)
    'separable': '#F59E0B',    # Amber (separable circuits)
    'noise': '#EF4444',        # Red (noise effects)
    'target': '#1F2937',       # Dark gray (target distribution)
}

# =============================================================================
# RANDOM SEED (for reproducibility)
# =============================================================================

RANDOM_SEED = 42

# =============================================================================
# MIDI GENERATION
# =============================================================================

# Default MIDI settings for generation
MIDI_TICKS_PER_BEAT = 480
MIDI_DEFAULT_VELOCITY = 100
MIDI_DEFAULT_DURATION = 480  # Quarter note
MIDI_DEFAULT_TEMPO = 500000  # 120 BPM

# =============================================================================
# EXPERIMENT NAMING
# =============================================================================

def get_experiment_name(phase: str, exp_num: int, description: str) -> str:
    """Generate a standardized experiment name."""
    return f"phase{phase}_exp{exp_num:02d}_{description}"
