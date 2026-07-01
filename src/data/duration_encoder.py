"""
Duration Encoder Module
=======================

Extends encoding to include note duration alongside pitch.
Uses 6 qubits: 4 for pitch + 2 for duration (4 duration bins).
"""

import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class DurationBin:
    """Represents a duration bin with its range and label."""
    index: int
    min_duration: float  # In beats
    max_duration: float
    label: str
    
    @property
    def center(self) -> float:
        return (self.min_duration + self.max_duration) / 2


# Standard duration bins (in beats, where 1 beat = quarter note)
DURATION_BINS = [
    DurationBin(0, 0.0, 0.25, "sixteenth"),      # 1/16 note
    DurationBin(1, 0.25, 0.5, "eighth"),          # 1/8 note  
    DurationBin(2, 0.5, 1.0, "quarter"),          # 1/4 note
    DurationBin(3, 1.0, float('inf'), "half+"),   # 1/2 note or longer
]


class DurationEncoder:
    """
    Encodes note durations into discrete bins for quantum representation.
    
    Uses 2 qubits for 4 duration categories:
    - 0: Sixteenth note (very short)
    - 1: Eighth note (short)
    - 2: Quarter note (medium)
    - 3: Half note or longer (long)
    
    Parameters
    ----------
    n_qubits : int
        Number of qubits for duration encoding (default: 2 for 4 bins).
    tempo : int
        Tempo in BPM for converting seconds to beats.
    
    Example
    -------
    >>> encoder = DurationEncoder(n_qubits=2, tempo=120)
    >>> durations = [0.25, 0.5, 1.0, 2.0]  # in seconds
    >>> encoded = encoder.encode(durations)
    >>> print(encoded)  # [1, 2, 3, 3]
    """
    
    def __init__(self, n_qubits: int = 2, tempo: int = 120):
        self.n_qubits = n_qubits
        self.n_bins = 2 ** n_qubits
        self.tempo = tempo
        self.seconds_per_beat = 60.0 / tempo
        
        # Create duration bins
        if self.n_bins == 4:
            self.bins = DURATION_BINS
        else:
            # Create custom bins for different qubit counts
            self.bins = self._create_bins()
    
    def _create_bins(self) -> List[DurationBin]:
        """Create duration bins for arbitrary number of qubits."""
        bins = []
        # Logarithmic spacing from 1/16 to 2 beats
        edges = np.logspace(np.log10(0.0625), np.log10(2.0), self.n_bins + 1)
        
        for i in range(self.n_bins):
            label = f"bin_{i}"
            bins.append(DurationBin(i, edges[i], edges[i+1], label))
        
        # Last bin extends to infinity
        bins[-1] = DurationBin(
            self.n_bins - 1, 
            bins[-1].min_duration, 
            float('inf'), 
            "long"
        )
        return bins
    
    def seconds_to_beats(self, seconds: float) -> float:
        """Convert duration from seconds to beats."""
        return seconds / self.seconds_per_beat
    
    def beats_to_seconds(self, beats: float) -> float:
        """Convert duration from beats to seconds."""
        return beats * self.seconds_per_beat
    
    def encode(self, durations: np.ndarray) -> np.ndarray:
        """
        Encode durations (in seconds) to bin indices.
        
        Parameters
        ----------
        durations : ndarray
            Array of durations in seconds.
        
        Returns
        -------
        ndarray
            Array of bin indices (0 to n_bins-1).
        """
        durations = np.atleast_1d(durations)
        # Convert to beats
        beats = durations / self.seconds_per_beat
        
        # Assign to bins
        indices = np.zeros(len(beats), dtype=int)
        for i, beat_dur in enumerate(beats):
            for bin_obj in self.bins:
                if bin_obj.min_duration <= beat_dur < bin_obj.max_duration:
                    indices[i] = bin_obj.index
                    break
            else:
                # Duration exceeds all bins - assign to last
                indices[i] = self.n_bins - 1
        
        return indices
    
    def decode(self, indices: np.ndarray) -> np.ndarray:
        """
        Decode bin indices back to durations (in seconds).
        
        Uses bin center values.
        
        Parameters
        ----------
        indices : ndarray
            Array of bin indices.
        
        Returns
        -------
        ndarray
            Array of durations in seconds.
        """
        indices = np.atleast_1d(indices)
        indices = np.clip(indices, 0, self.n_bins - 1)
        
        # Use bin centers (cap the "infinite" bin)
        durations_beats = np.array([
            min(self.bins[int(idx)].center, 2.0) for idx in indices
        ])
        
        return durations_beats * self.seconds_per_beat
    
    def get_distribution(self, durations: np.ndarray) -> np.ndarray:
        """
        Compute probability distribution over duration bins.
        
        Parameters
        ----------
        durations : ndarray
            Array of durations in seconds.
        
        Returns
        -------
        ndarray
            Probability distribution of shape (n_bins,).
        """
        indices = self.encode(durations)
        counts = np.bincount(indices, minlength=self.n_bins)
        return counts / counts.sum()


class PitchDurationEncoder:
    """
    Joint encoder for pitch and duration.
    
    Combines pitch (4 qubits) and duration (2 qubits) into a 6-qubit 
    joint representation with 64 possible states.
    
    State index = pitch_bin * n_duration_bins + duration_bin
    
    Parameters
    ----------
    n_qubits_pitch : int
        Number of qubits for pitch (default: 4).
    n_qubits_duration : int
        Number of qubits for duration (default: 2).
    pitch_min : int
        Minimum MIDI pitch.
    pitch_max : int
        Maximum MIDI pitch.
    tempo : int
        Tempo in BPM.
    
    Example
    -------
    >>> encoder = PitchDurationEncoder(n_qubits_pitch=4, n_qubits_duration=2)
    >>> encoder.fit(pitches, durations)
    >>> joint_indices = encoder.encode(pitches, durations)
    >>> decoded_pitches, decoded_durations = encoder.decode(joint_indices)
    """
    
    def __init__(
        self,
        n_qubits_pitch: int = 4,
        n_qubits_duration: int = 2,
        pitch_min: Optional[int] = None,
        pitch_max: Optional[int] = None,
        tempo: int = 120
    ):
        self.n_qubits_pitch = n_qubits_pitch
        self.n_qubits_duration = n_qubits_duration
        self.n_qubits_total = n_qubits_pitch + n_qubits_duration
        
        self.n_pitch_bins = 2 ** n_qubits_pitch
        self.n_duration_bins = 2 ** n_qubits_duration
        self.n_states = 2 ** self.n_qubits_total
        
        self.pitch_min = pitch_min if pitch_min is not None else 36
        self.pitch_max = pitch_max if pitch_max is not None else 84
        self.tempo = tempo
        
        # Create individual encoders
        self._update_pitch_encoder()
        self.duration_encoder = DurationEncoder(n_qubits_duration, tempo)
        
        self._fitted = False
    
    def _update_pitch_encoder(self):
        """Update pitch binning."""
        self.pitch_bin_edges = np.linspace(
            self.pitch_min, self.pitch_max, self.n_pitch_bins + 1
        )
        self.pitch_bin_centers = (
            self.pitch_bin_edges[:-1] + self.pitch_bin_edges[1:]
        ) / 2
    
    def fit(self, pitches: np.ndarray, durations: Optional[np.ndarray] = None):
        """
        Fit encoder to data.
        
        Parameters
        ----------
        pitches : ndarray
            Array of MIDI pitches.
        durations : ndarray, optional
            Array of durations in seconds (not currently used for fitting).
        
        Returns
        -------
        self
        """
        pitches = np.asarray(pitches)
        self.pitch_min = int(np.min(pitches))
        self.pitch_max = int(np.max(pitches)) + 1
        self._update_pitch_encoder()
        self._fitted = True
        return self
    
    def encode_pitch(self, pitches: np.ndarray) -> np.ndarray:
        """Encode pitches to bin indices."""
        pitches = np.atleast_1d(pitches)
        clipped = np.clip(pitches, self.pitch_min, self.pitch_max - 0.01)
        indices = np.digitize(clipped, self.pitch_bin_edges) - 1
        return np.clip(indices, 0, self.n_pitch_bins - 1).astype(int)
    
    def decode_pitch(self, indices: np.ndarray) -> np.ndarray:
        """Decode pitch bin indices to MIDI pitches."""
        indices = np.clip(indices, 0, self.n_pitch_bins - 1)
        return self.pitch_bin_centers[indices]
    
    def encode(self, pitches: np.ndarray, durations: np.ndarray) -> np.ndarray:
        """
        Encode pitch-duration pairs to joint indices.
        
        Parameters
        ----------
        pitches : ndarray
            MIDI pitches.
        durations : ndarray
            Durations in seconds.
        
        Returns
        -------
        ndarray
            Joint state indices (0 to n_states-1).
        """
        pitch_indices = self.encode_pitch(pitches)
        duration_indices = self.duration_encoder.encode(durations)
        
        # Combine: joint_index = pitch_bin * n_duration_bins + duration_bin
        joint_indices = pitch_indices * self.n_duration_bins + duration_indices
        return joint_indices
    
    def decode(self, joint_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decode joint indices to pitch-duration pairs.
        
        Parameters
        ----------
        joint_indices : ndarray
            Joint state indices.
        
        Returns
        -------
        pitches : ndarray
            Decoded MIDI pitches.
        durations : ndarray
            Decoded durations in seconds.
        """
        joint_indices = np.atleast_1d(joint_indices)
        
        # Extract components
        pitch_indices = joint_indices // self.n_duration_bins
        duration_indices = joint_indices % self.n_duration_bins
        
        pitches = self.decode_pitch(pitch_indices)
        durations = self.duration_encoder.decode(duration_indices)
        
        return pitches, durations
    
    def get_joint_distribution(
        self, 
        pitches: np.ndarray, 
        durations: np.ndarray
    ) -> np.ndarray:
        """
        Compute joint probability distribution over pitch-duration states.
        
        Parameters
        ----------
        pitches : ndarray
            MIDI pitches.
        durations : ndarray
            Durations in seconds.
        
        Returns
        -------
        ndarray
            Joint probability distribution of shape (n_states,).
        """
        joint_indices = self.encode(pitches, durations)
        counts = np.bincount(joint_indices, minlength=self.n_states)
        return counts / counts.sum()
    
    def get_marginal_pitch(self, joint_dist: np.ndarray) -> np.ndarray:
        """Extract marginal pitch distribution from joint."""
        joint_2d = joint_dist.reshape(self.n_pitch_bins, self.n_duration_bins)
        return joint_2d.sum(axis=1)
    
    def get_marginal_duration(self, joint_dist: np.ndarray) -> np.ndarray:
        """Extract marginal duration distribution from joint."""
        joint_2d = joint_dist.reshape(self.n_pitch_bins, self.n_duration_bins)
        return joint_2d.sum(axis=0)
    
    def get_conditional_duration(
        self, 
        joint_dist: np.ndarray, 
        pitch_bin: int
    ) -> np.ndarray:
        """Get P(duration | pitch) for a specific pitch bin."""
        joint_2d = joint_dist.reshape(self.n_pitch_bins, self.n_duration_bins)
        row = joint_2d[pitch_bin]
        return row / (row.sum() + 1e-10)
    
    def index_to_binary(self, index: int) -> str:
        """Convert joint index to binary string."""
        return format(index, f'0{self.n_qubits_total}b')
    
    def visualize_joint_distribution(self, distribution: np.ndarray, ax=None):
        """
        Visualize the joint pitch-duration distribution as heatmap.
        
        Parameters
        ----------
        distribution : ndarray
            Joint distribution of shape (n_states,).
        ax : matplotlib Axes, optional
        
        Returns
        -------
        matplotlib Axes
        """
        import matplotlib.pyplot as plt
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # Reshape to 2D
        joint_2d = distribution.reshape(self.n_pitch_bins, self.n_duration_bins)
        
        im = ax.imshow(joint_2d, aspect='auto', cmap='viridis', origin='lower')
        ax.set_xlabel('Duration Bin')
        ax.set_ylabel('Pitch Bin')
        ax.set_title('Joint Pitch-Duration Distribution')
        
        # Duration labels
        duration_labels = ['16th', '8th', '4th', 'half+'][:self.n_duration_bins]
        ax.set_xticks(range(self.n_duration_bins))
        ax.set_xticklabels(duration_labels)
        
        plt.colorbar(im, ax=ax, label='Probability')
        
        return ax
