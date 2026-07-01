"""
Data Encoder Module
===================

Encodes musical features (pitch, velocity) into quantum-compatible binary representations.
"""

import numpy as np
from typing import Tuple, Optional
from abc import ABC, abstractmethod


class BaseEncoder(ABC):
    """Abstract base class for data encoders."""
    
    @abstractmethod
    def encode(self, data: np.ndarray) -> np.ndarray:
        """Encode continuous/discrete data to binary indices."""
        pass
    
    @abstractmethod
    def decode(self, indices: np.ndarray) -> np.ndarray:
        """Decode binary indices back to original space."""
        pass
    
    @abstractmethod
    def get_distribution(self, data: np.ndarray) -> np.ndarray:
        """Compute probability distribution from data."""
        pass


class PitchEncoder(BaseEncoder):
    """
    Encodes MIDI pitches into discrete bins for quantum representation.
    
    For n_qubits qubits, we have 2^n_qubits bins.
    Pitches outside the range are clipped.
    
    Parameters
    ----------
    n_qubits : int
        Number of qubits (determines number of bins as 2^n_qubits).
    pitch_min : int
        Minimum pitch value (default: 36, C2).
    pitch_max : int
        Maximum pitch value (default: 84, C6).
    
    Example
    -------
    >>> encoder = PitchEncoder(n_qubits=4)  # 16 bins
    >>> pitches = np.array([60, 65, 72, 48])
    >>> encoded = encoder.encode(pitches)
    >>> distribution = encoder.get_distribution(pitches)
    """
    
    def __init__(
        self,
        n_qubits: int = 4,
        pitch_min: Optional[int] = None,
        pitch_max: Optional[int] = None
    ):
        self.n_qubits = n_qubits
        self.n_bins = 2 ** n_qubits
        self.pitch_min = pitch_min if pitch_min is not None else 36
        self.pitch_max = pitch_max if pitch_max is not None else 84
        self._fitted = False
        
        # Create bin edges (will be updated if fit() is called)
        self._update_bins()
    
    def _update_bins(self):
        """Update bin edges and centers based on current pitch range."""
        self.bin_edges = np.linspace(self.pitch_min, self.pitch_max, self.n_bins + 1)
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
        self.bin_width = (self.pitch_max - self.pitch_min) / self.n_bins
    
    def fit(self, pitches: np.ndarray):
        """
        Fit the encoder to the pitch range in the data.
        
        Parameters
        ----------
        pitches : ndarray or list
            Array of MIDI pitch values.
        
        Returns
        -------
        self
            Returns self for method chaining.
        """
        pitches = np.asarray(pitches)
        self.pitch_min = int(np.min(pitches))
        self.pitch_max = int(np.max(pitches)) + 1  # +1 to include max value
        self._update_bins()
        self._fitted = True
        return self
    
    def encode(self, pitches: np.ndarray) -> np.ndarray:
        """
        Encode pitches to bin indices.
        
        Parameters
        ----------
        pitches : ndarray
            Array of MIDI pitch values.
        
        Returns
        -------
        ndarray
            Array of bin indices (0 to n_bins-1).
        """
        # Clip to valid range
        clipped = np.clip(pitches, self.pitch_min, self.pitch_max - 0.01)
        
        # Compute bin indices
        indices = np.digitize(clipped, self.bin_edges) - 1
        
        # Ensure indices are in valid range
        indices = np.clip(indices, 0, self.n_bins - 1)
        
        return indices.astype(int)
    
    def decode(self, indices: np.ndarray) -> np.ndarray:
        """
        Decode bin indices back to pitch values (using bin centers).
        
        Parameters
        ----------
        indices : ndarray
            Array of bin indices.
        
        Returns
        -------
        ndarray
            Array of pitch values (bin centers).
        """
        indices = np.clip(indices, 0, self.n_bins - 1)
        return self.bin_centers[indices]
    
    def get_distribution(self, pitches: np.ndarray) -> np.ndarray:
        """
        Compute probability distribution over bins.
        
        Parameters
        ----------
        pitches : ndarray
            Array of MIDI pitch values.
        
        Returns
        -------
        ndarray
            Probability distribution of shape (n_bins,).
        """
        indices = self.encode(pitches)
        counts = np.bincount(indices, minlength=self.n_bins)
        distribution = counts / counts.sum()
        return distribution
    
    def index_to_binary(self, index: int) -> str:
        """Convert bin index to binary string representation."""
        return format(index, f'0{self.n_qubits}b')
    
    def binary_to_index(self, binary: str) -> int:
        """Convert binary string to bin index."""
        return int(binary, 2)


class PitchVelocityEncoder(BaseEncoder):
    """
    Encodes both pitch and velocity for joint quantum representation.
    
    Uses n_qubits_pitch qubits for pitch and n_qubits_velocity qubits for velocity,
    creating a joint distribution over 2^(n_qubits_pitch + n_qubits_velocity) states.
    
    Parameters
    ----------
    n_qubits_pitch : int
        Number of qubits for pitch encoding.
    n_qubits_velocity : int
        Number of qubits for velocity encoding.
    pitch_min : int
        Minimum pitch value.
    pitch_max : int
        Maximum pitch value.
    velocity_min : int
        Minimum velocity value.
    velocity_max : int
        Maximum velocity value.
    
    Example
    -------
    >>> encoder = PitchVelocityEncoder(n_qubits_pitch=4, n_qubits_velocity=4)
    >>> data = np.array([[60, 100], [65, 80], [72, 120]])
    >>> distribution = encoder.get_distribution(data)
    """
    
    def __init__(
        self,
        n_qubits_pitch: int = 4,
        n_qubits_velocity: int = 4,
        pitch_min: int = 36,
        pitch_max: int = 84,
        velocity_min: int = 0,
        velocity_max: int = 127
    ):
        self.n_qubits_pitch = n_qubits_pitch
        self.n_qubits_velocity = n_qubits_velocity
        self.n_qubits_total = n_qubits_pitch + n_qubits_velocity
        
        self.n_bins_pitch = 2 ** n_qubits_pitch
        self.n_bins_velocity = 2 ** n_qubits_velocity
        self.n_bins_total = 2 ** self.n_qubits_total
        
        # Pitch encoding
        self.pitch_encoder = PitchEncoder(
            n_qubits=n_qubits_pitch,
            pitch_min=pitch_min,
            pitch_max=pitch_max
        )
        
        # Velocity encoding
        self.velocity_min = velocity_min
        self.velocity_max = velocity_max
        self.velocity_bin_edges = np.linspace(
            velocity_min, velocity_max, self.n_bins_velocity + 1
        )
        self.velocity_bin_centers = (
            self.velocity_bin_edges[:-1] + self.velocity_bin_edges[1:]
        ) / 2
    
    def encode(self, data: np.ndarray) -> np.ndarray:
        """
        Encode (pitch, velocity) pairs to joint indices.
        
        Parameters
        ----------
        data : ndarray
            Array of shape (n_samples, 2) with [pitch, velocity] pairs.
        
        Returns
        -------
        ndarray
            Array of joint indices (0 to n_bins_total-1).
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        pitches = data[:, 0]
        velocities = data[:, 1]
        
        # Encode pitch
        pitch_indices = self.pitch_encoder.encode(pitches)
        
        # Encode velocity
        clipped_vel = np.clip(velocities, self.velocity_min, self.velocity_max - 0.01)
        velocity_indices = np.digitize(clipped_vel, self.velocity_bin_edges) - 1
        velocity_indices = np.clip(velocity_indices, 0, self.n_bins_velocity - 1)
        
        # Combine: joint_index = pitch_index * n_bins_velocity + velocity_index
        joint_indices = pitch_indices * self.n_bins_velocity + velocity_indices
        
        return joint_indices.astype(int)
    
    def decode(self, indices: np.ndarray) -> np.ndarray:
        """
        Decode joint indices back to (pitch, velocity) pairs.
        
        Parameters
        ----------
        indices : ndarray
            Array of joint indices.
        
        Returns
        -------
        ndarray
            Array of shape (n_samples, 2) with [pitch, velocity] pairs.
        """
        indices = np.asarray(indices)
        
        pitch_indices = indices // self.n_bins_velocity
        velocity_indices = indices % self.n_bins_velocity
        
        pitches = self.pitch_encoder.decode(pitch_indices)
        velocities = self.velocity_bin_centers[velocity_indices]
        
        return np.column_stack([pitches, velocities])
    
    def get_distribution(self, data: np.ndarray) -> np.ndarray:
        """
        Compute joint probability distribution.
        
        Parameters
        ----------
        data : ndarray
            Array of shape (n_samples, 2) with [pitch, velocity] pairs.
        
        Returns
        -------
        ndarray
            Probability distribution of shape (n_bins_total,).
        """
        indices = self.encode(data)
        counts = np.bincount(indices, minlength=self.n_bins_total)
        distribution = counts / counts.sum()
        return distribution
    
    def get_marginal_pitch(self, joint_dist: np.ndarray) -> np.ndarray:
        """Compute marginal distribution over pitch from joint distribution."""
        joint_2d = joint_dist.reshape(self.n_bins_pitch, self.n_bins_velocity)
        return joint_2d.sum(axis=1)
    
    def get_marginal_velocity(self, joint_dist: np.ndarray) -> np.ndarray:
        """Compute marginal distribution over velocity from joint distribution."""
        joint_2d = joint_dist.reshape(self.n_bins_pitch, self.n_bins_velocity)
        return joint_2d.sum(axis=0)
    
    def compute_mutual_information(self, joint_dist: np.ndarray) -> float:
        """
        Compute mutual information between pitch and velocity.
        
        Higher values indicate stronger correlation (entanglement useful).
        
        Parameters
        ----------
        joint_dist : ndarray
            Joint probability distribution.
        
        Returns
        -------
        float
            Mutual information in bits.
        """
        eps = 1e-10
        
        p_xy = joint_dist.reshape(self.n_bins_pitch, self.n_bins_velocity)
        p_x = p_xy.sum(axis=1, keepdims=True)
        p_y = p_xy.sum(axis=0, keepdims=True)
        
        # I(X;Y) = sum p(x,y) * log(p(x,y) / (p(x)*p(y)))
        with np.errstate(divide='ignore', invalid='ignore'):
            mi_matrix = p_xy * np.log2((p_xy + eps) / (p_x * p_y + eps))
            mi_matrix = np.where(p_xy > eps, mi_matrix, 0)
        
        return float(np.sum(mi_matrix))
