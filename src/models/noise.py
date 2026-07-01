"""
Noise Module
============

Noise models for simulating NISQ hardware errors.
"""

import pennylane as qml
import numpy as np
from typing import Optional, List
from abc import ABC, abstractmethod


class NoiseModel(ABC):
    """Abstract base class for noise models."""
    
    @abstractmethod
    def apply(self, wires: List[int]) -> None:
        """Apply noise to specified wires."""
        pass


class DepolarizingNoise(NoiseModel):
    """
    Depolarizing noise channel.
    
    With probability p, replaces the qubit state with the maximally mixed state.
    This is a common noise model for NISQ devices.
    
    Parameters
    ----------
    probability : float
        Depolarizing probability (0 to 1).
    
    Example
    -------
    >>> noise = DepolarizingNoise(probability=0.05)
    >>> # In a PennyLane circuit:
    >>> noise.apply(wires=[0, 1, 2, 3])
    """
    
    def __init__(self, probability: float = 0.01):
        if not 0 <= probability <= 1:
            raise ValueError(f"Probability must be in [0, 1], got {probability}")
        self.probability = probability
    
    def apply(self, wires: List[int]) -> None:
        """Apply depolarizing noise to each wire."""
        if self.probability > 0:
            for wire in wires:
                qml.DepolarizingChannel(self.probability, wires=wire)
    
    def __repr__(self) -> str:
        return f"DepolarizingNoise(probability={self.probability})"


class BitFlipNoise(NoiseModel):
    """
    Bit-flip noise channel.
    
    With probability p, applies an X gate (bit flip) to the qubit.
    
    Parameters
    ----------
    probability : float
        Bit-flip probability (0 to 1).
    """
    
    def __init__(self, probability: float = 0.01):
        if not 0 <= probability <= 1:
            raise ValueError(f"Probability must be in [0, 1], got {probability}")
        self.probability = probability
    
    def apply(self, wires: List[int]) -> None:
        """Apply bit-flip noise to each wire."""
        if self.probability > 0:
            for wire in wires:
                qml.BitFlip(self.probability, wires=wire)
    
    def __repr__(self) -> str:
        return f"BitFlipNoise(probability={self.probability})"


class PhaseFlipNoise(NoiseModel):
    """
    Phase-flip noise channel.
    
    With probability p, applies a Z gate (phase flip) to the qubit.
    
    Parameters
    ----------
    probability : float
        Phase-flip probability (0 to 1).
    """
    
    def __init__(self, probability: float = 0.01):
        if not 0 <= probability <= 1:
            raise ValueError(f"Probability must be in [0, 1], got {probability}")
        self.probability = probability
    
    def apply(self, wires: List[int]) -> None:
        """Apply phase-flip noise to each wire."""
        if self.probability > 0:
            for wire in wires:
                qml.PhaseFlip(self.probability, wires=wire)
    
    def __repr__(self) -> str:
        return f"PhaseFlipNoise(probability={self.probability})"


class CompositeNoise(NoiseModel):
    """
    Composite noise model combining multiple noise sources.
    
    Parameters
    ----------
    noise_models : list of NoiseModel
        List of noise models to apply sequentially.
    """
    
    def __init__(self, noise_models: List[NoiseModel]):
        self.noise_models = noise_models
    
    def apply(self, wires: List[int]) -> None:
        """Apply all noise models to each wire."""
        for noise_model in self.noise_models:
            noise_model.apply(wires)
    
    def __repr__(self) -> str:
        models_str = ", ".join(repr(m) for m in self.noise_models)
        return f"CompositeNoise([{models_str}])"


def create_noise_model(
    noise_type: str = 'depolarizing',
    probability: float = 0.01
) -> Optional[NoiseModel]:
    """
    Factory function to create a noise model.
    
    Parameters
    ----------
    noise_type : str
        Type of noise: 'depolarizing', 'bitflip', 'phaseflip', or 'none'.
    probability : float
        Noise probability.
    
    Returns
    -------
    NoiseModel or None
        Configured noise model, or None if noise_type is 'none'.
    """
    if noise_type == 'none' or probability == 0:
        return None
    elif noise_type == 'depolarizing':
        return DepolarizingNoise(probability)
    elif noise_type == 'bitflip':
        return BitFlipNoise(probability)
    elif noise_type == 'phaseflip':
        return PhaseFlipNoise(probability)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")


# =============================================================================
# NISQ Hardware Noise Profiles
# =============================================================================

def get_ibm_noise_profile() -> CompositeNoise:
    """
    Approximate noise profile for IBM Quantum hardware.
    
    Based on typical error rates for IBM quantum computers:
    - Single-qubit gate error: ~0.1%
    - Two-qubit gate error: ~1%
    - Readout error: ~1%
    
    Returns
    -------
    CompositeNoise
        Composite noise model approximating IBM hardware.
    """
    return CompositeNoise([
        DepolarizingNoise(probability=0.01),
        BitFlipNoise(probability=0.001),
    ])


def get_rigetti_noise_profile() -> CompositeNoise:
    """
    Approximate noise profile for Rigetti hardware.
    
    Returns
    -------
    CompositeNoise
        Composite noise model approximating Rigetti hardware.
    """
    return CompositeNoise([
        DepolarizingNoise(probability=0.02),
        PhaseFlipNoise(probability=0.005),
    ])
