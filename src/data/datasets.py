"""
Dataset Module
==============

Creates training datasets (target distributions) for QCBM experiments.
Includes both simple (uncorrelated) and complex (correlated) datasets.
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass

from .encoder import PitchEncoder, PitchVelocityEncoder
from .midi_parser import MidiParser


@dataclass
class Dataset:
    """Container for a dataset with target distribution."""
    name: str
    distribution: np.ndarray
    n_qubits: int
    description: str
    encoder: Optional[object] = None
    
    @property
    def n_states(self) -> int:
        """Number of possible states."""
        return len(self.distribution)
    
    @property
    def entropy(self) -> float:
        """Shannon entropy of the distribution (in bits)."""
        eps = 1e-10
        p = self.distribution + eps
        return -np.sum(p * np.log2(p))
    
    @property
    def sparsity(self) -> float:
        """Fraction of states with near-zero probability."""
        return np.mean(self.distribution < 1e-6)


class SimpleDataset:
    """
    Creates a simple dataset from MIDI with uncorrelated features.
    
    This is ideal for baseline experiments where entanglement should NOT
    provide an advantage.
    
    Parameters
    ----------
    midi_path : str
        Path to MIDI file.
    n_qubits : int
        Number of qubits for encoding.
    
    Example
    -------
    >>> dataset = SimpleDataset('mario.mid', n_qubits=4)
    >>> target = dataset.create()
    >>> print(f"Entropy: {target.entropy:.2f} bits")
    """
    
    def __init__(self, midi_path: str, n_qubits: int = 4):
        self.midi_path = midi_path
        self.n_qubits = n_qubits
        self.parser = MidiParser(midi_path)
        self.encoder = PitchEncoder(n_qubits=n_qubits)
    
    def create(self) -> Dataset:
        """Create the dataset with target distribution."""
        self.parser.parse()
        pitches = self.parser.get_pitches()
        
        # Adjust encoder range to data
        self.encoder = PitchEncoder(
            n_qubits=self.n_qubits,
            pitch_min=int(pitches.min()),
            pitch_max=int(pitches.max()) + 1
        )
        
        distribution = self.encoder.get_distribution(pitches)
        
        return Dataset(
            name=f"Simple_{self.n_qubits}q",
            distribution=distribution,
            n_qubits=self.n_qubits,
            description=f"Simple pitch distribution from {self.midi_path}",
            encoder=self.encoder
        )


class ComplexDataset:
    """
    Creates a complex dataset with correlated pitch-velocity structure.
    
    This dataset has strong correlations between pitch and velocity,
    requiring quantum entanglement to model properly.
    
    The correlation pattern can be:
    - 'linear': velocity increases linearly with pitch
    - 'quadratic': velocity peaks at middle pitches
    - 'bimodal': two distinct pitch-velocity clusters
    
    Parameters
    ----------
    n_qubits_pitch : int
        Number of qubits for pitch.
    n_qubits_velocity : int
        Number of qubits for velocity.
    correlation_type : str
        Type of correlation ('linear', 'quadratic', 'bimodal').
    noise_level : float
        Amount of noise to add (0 to 1).
    
    Example
    -------
    >>> dataset = ComplexDataset(n_qubits_pitch=4, n_qubits_velocity=4)
    >>> target = dataset.create(correlation_type='linear')
    >>> print(f"Mutual Information: {dataset.mutual_information:.2f} bits")
    """
    
    def __init__(
        self,
        n_qubits_pitch: int = 4,
        n_qubits_velocity: int = 4,
        correlation_type: str = 'linear',
        noise_level: float = 0.1
    ):
        self.n_qubits_pitch = n_qubits_pitch
        self.n_qubits_velocity = n_qubits_velocity
        self.n_qubits_total = n_qubits_pitch + n_qubits_velocity
        self.correlation_type = correlation_type
        self.noise_level = noise_level
        
        self.encoder = PitchVelocityEncoder(
            n_qubits_pitch=n_qubits_pitch,
            n_qubits_velocity=n_qubits_velocity
        )
        
        self.mutual_information = 0.0
    
    def create(self) -> Dataset:
        """Create the correlated dataset."""
        n_pitch = 2 ** self.n_qubits_pitch
        n_vel = 2 ** self.n_qubits_velocity
        
        # Create joint distribution based on correlation type
        if self.correlation_type == 'linear':
            joint = self._create_linear_correlation(n_pitch, n_vel)
        elif self.correlation_type == 'quadratic':
            joint = self._create_quadratic_correlation(n_pitch, n_vel)
        elif self.correlation_type == 'bimodal':
            joint = self._create_bimodal_correlation(n_pitch, n_vel)
        else:
            raise ValueError(f"Unknown correlation type: {self.correlation_type}")
        
        # Add noise
        if self.noise_level > 0:
            uniform = np.ones_like(joint) / joint.size
            joint = (1 - self.noise_level) * joint + self.noise_level * uniform
        
        # Normalize
        distribution = joint.flatten()
        distribution = distribution / distribution.sum()
        
        # Compute mutual information
        self.mutual_information = self.encoder.compute_mutual_information(distribution)
        
        return Dataset(
            name=f"Complex_{self.correlation_type}_{self.n_qubits_total}q",
            distribution=distribution,
            n_qubits=self.n_qubits_total,
            description=f"Correlated pitch-velocity with {self.correlation_type} pattern",
            encoder=self.encoder
        )
    
    def _create_linear_correlation(self, n_pitch: int, n_vel: int) -> np.ndarray:
        """Create linear correlation: high pitch → high velocity."""
        joint = np.zeros((n_pitch, n_vel))
        
        for p in range(n_pitch):
            # Expected velocity proportional to pitch
            mean_vel = int((p / n_pitch) * n_vel)
            sigma = max(1, n_vel // 8)  # Spread around the mean
            
            for v in range(n_vel):
                joint[p, v] = np.exp(-0.5 * ((v - mean_vel) / sigma) ** 2)
        
        return joint / joint.sum()
    
    def _create_quadratic_correlation(self, n_pitch: int, n_vel: int) -> np.ndarray:
        """Create quadratic correlation: middle pitch → high velocity."""
        joint = np.zeros((n_pitch, n_vel))
        
        for p in range(n_pitch):
            # Velocity peaks at middle pitches
            distance_from_center = abs(p - n_pitch // 2) / (n_pitch // 2)
            mean_vel = int((1 - distance_from_center) * (n_vel - 1))
            sigma = max(1, n_vel // 8)
            
            for v in range(n_vel):
                joint[p, v] = np.exp(-0.5 * ((v - mean_vel) / sigma) ** 2)
        
        return joint / joint.sum()
    
    def _create_bimodal_correlation(self, n_pitch: int, n_vel: int) -> np.ndarray:
        """Create bimodal: two distinct clusters (low-soft, high-loud)."""
        joint = np.zeros((n_pitch, n_vel))
        
        # Cluster 1: Low pitch, low velocity
        p1, v1 = n_pitch // 4, n_vel // 4
        sigma_p, sigma_v = n_pitch // 8, n_vel // 8
        
        # Cluster 2: High pitch, high velocity
        p2, v2 = 3 * n_pitch // 4, 3 * n_vel // 4
        
        for p in range(n_pitch):
            for v in range(n_vel):
                # Two Gaussian clusters
                g1 = np.exp(-0.5 * (((p - p1) / sigma_p) ** 2 + ((v - v1) / sigma_v) ** 2))
                g2 = np.exp(-0.5 * (((p - p2) / sigma_p) ** 2 + ((v - v2) / sigma_v) ** 2))
                joint[p, v] = g1 + g2
        
        return joint / joint.sum()


def create_target_distribution(
    source: str = 'simple',
    midi_path: Optional[str] = None,
    n_qubits: int = 4,
    correlation_type: str = 'linear',
    **kwargs
) -> Dataset:
    """
    Factory function to create target distributions.
    
    Parameters
    ----------
    source : str
        'simple' for MIDI-based, 'complex' for synthetic correlated.
    midi_path : str, optional
        Path to MIDI file (required if source='simple').
    n_qubits : int
        Number of qubits for simple, or per-feature for complex.
    correlation_type : str
        Type of correlation for complex datasets.
    
    Returns
    -------
    Dataset
        Dataset object with target distribution.
    """
    if source == 'simple':
        if midi_path is None:
            raise ValueError("midi_path required for simple dataset")
        return SimpleDataset(midi_path, n_qubits).create()
    
    elif source == 'complex':
        return ComplexDataset(
            n_qubits_pitch=n_qubits,
            n_qubits_velocity=n_qubits,
            correlation_type=correlation_type,
            **kwargs
        ).create()
    
    else:
        raise ValueError(f"Unknown source: {source}")


# =============================================================================
# Predefined datasets for experiments
# =============================================================================

def get_mario_dataset(n_qubits: int = 4, midi_path: str = None) -> Dataset:
    """Get the Super Mario Bros dataset (simple, uncorrelated)."""
    if midi_path is None:
        from pathlib import Path
        midi_path = Path(__file__).parent.parent.parent / "mario.mid"
    return create_target_distribution('simple', str(midi_path), n_qubits)


def get_trap_dataset(n_qubits: int = 4) -> Dataset:
    """Get the Trap dataset (complex, strongly correlated)."""
    return create_target_distribution(
        'complex', 
        n_qubits=n_qubits,
        correlation_type='bimodal',
        noise_level=0.05
    )
