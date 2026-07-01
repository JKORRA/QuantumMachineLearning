"""
Temporal Modeling Module
========================

Implements Markov-based temporal structure for QCBM music generation.
Instead of i.i.d. sampling, notes depend on previous notes.
"""

import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class MarkovTransition:
    """Represents a Markov transition matrix."""
    matrix: np.ndarray  # Shape: (n_states, n_states)
    n_states: int
    order: int = 1  # Markov order (1 = depends on previous note only)
    
    def __post_init__(self):
        # Normalize rows to be valid probability distributions
        row_sums = self.matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        self.matrix = self.matrix / row_sums


class MarkovQCBM:
    """
    Markov-enhanced QCBM for temporal music generation.
    
    Combines a base QCBM distribution with learned Markov transitions
    to create musically coherent sequences.
    
    The generation process:
    1. Sample first note from QCBM marginal distribution
    2. For subsequent notes: P(x_t | x_{t-1}) ∝ P_QCBM(x_t) * T(x_t | x_{t-1})
    
    This creates sequences that:
    - Follow the overall pitch distribution (QCBM)
    - Have smooth transitions (Markov)
    
    Parameters
    ----------
    qcbm : QCBM
        Base quantum circuit born machine.
    transition_strength : float
        Weight of Markov transitions vs QCBM (0=pure QCBM, 1=pure Markov).
    
    Example
    -------
    >>> from src.models.qcbm import QCBM
    >>> qcbm = QCBM(n_qubits=4, n_layers=3)
    >>> markov_qcbm = MarkovQCBM(qcbm, transition_strength=0.5)
    >>> markov_qcbm.fit_transitions(training_sequences)
    >>> melody = markov_qcbm.generate_sequence(length=50, params=trained_params)
    """
    
    def __init__(
        self,
        qcbm,  # QCBM instance
        transition_strength: float = 0.5,
        smoothing: float = 0.1
    ):
        self.qcbm = qcbm
        self.transition_strength = transition_strength
        self.smoothing = smoothing  # Laplace smoothing for sparse transitions
        self.n_states = qcbm.n_states
        
        # Initialize uniform transition matrix
        self.transition_matrix = np.ones((self.n_states, self.n_states)) / self.n_states
        self._fitted = False
    
    def fit_transitions(self, sequences: List[np.ndarray]) -> 'MarkovQCBM':
        """
        Learn transition probabilities from training sequences.
        
        Parameters
        ----------
        sequences : list of ndarray
            List of note sequences (each is array of state indices).
        
        Returns
        -------
        self
            Returns self for method chaining.
        """
        # Count transitions
        transition_counts = np.zeros((self.n_states, self.n_states))
        
        for seq in sequences:
            for i in range(len(seq) - 1):
                current_state = int(seq[i])
                next_state = int(seq[i + 1])
                if 0 <= current_state < self.n_states and 0 <= next_state < self.n_states:
                    transition_counts[current_state, next_state] += 1
        
        # Apply Laplace smoothing
        transition_counts += self.smoothing
        
        # Normalize to probabilities
        row_sums = transition_counts.sum(axis=1, keepdims=True)
        self.transition_matrix = transition_counts / row_sums
        
        self._fitted = True
        return self
    
    def fit_from_notes(self, notes: List, encoder) -> 'MarkovQCBM':
        """
        Learn transitions from a list of Note objects.
        
        Parameters
        ----------
        notes : list of Note
            Musical notes with pitch attribute.
        encoder : PitchEncoder
            Encoder to convert pitches to state indices.
        
        Returns
        -------
        self
        """
        # Sort notes by start time
        sorted_notes = sorted(notes, key=lambda n: n.start_time)
        
        # Encode pitches to states
        pitches = [n.pitch for n in sorted_notes]
        states = encoder.encode(np.array(pitches))
        
        # Fit on single sequence
        return self.fit_transitions([states])
    
    def get_combined_distribution(
        self,
        params: np.ndarray,
        previous_state: Optional[int] = None
    ) -> np.ndarray:
        """
        Get combined QCBM + Markov distribution.
        
        Parameters
        ----------
        params : ndarray
            QCBM parameters.
        previous_state : int, optional
            Previous state for Markov conditioning.
        
        Returns
        -------
        ndarray
            Combined probability distribution.
        """
        # Get QCBM distribution
        qcbm_probs = self.qcbm.get_probabilities(params)
        
        if previous_state is None or not self._fitted:
            return qcbm_probs
        
        # Get Markov transition probabilities
        markov_probs = self.transition_matrix[previous_state]
        
        # Combine: weighted geometric mean
        alpha = self.transition_strength
        combined = (qcbm_probs ** (1 - alpha)) * (markov_probs ** alpha)
        
        # Normalize
        combined /= combined.sum()
        
        return combined
    
    def generate_sequence(
        self,
        params: np.ndarray,
        length: int,
        initial_state: Optional[int] = None,
        temperature: float = 1.0
    ) -> np.ndarray:
        """
        Generate a sequence of notes with temporal structure.
        
        Parameters
        ----------
        params : ndarray
            QCBM parameters.
        length : int
            Number of notes to generate.
        initial_state : int, optional
            Starting state. If None, samples from QCBM.
        temperature : float
            Sampling temperature (lower = more deterministic).
        
        Returns
        -------
        ndarray
            Sequence of state indices.
        """
        sequence = []
        
        # Generate first note
        if initial_state is not None:
            current_state = initial_state
        else:
            probs = self.qcbm.get_probabilities(params)
            current_state = np.random.choice(self.n_states, p=probs)
        
        sequence.append(current_state)
        
        # Generate subsequent notes
        for _ in range(length - 1):
            # Get combined distribution
            probs = self.get_combined_distribution(params, current_state)
            
            # Apply temperature
            if temperature != 1.0:
                probs = probs ** (1.0 / temperature)
                probs /= probs.sum()
            
            # Sample next state
            next_state = np.random.choice(self.n_states, p=probs)
            sequence.append(next_state)
            current_state = next_state
        
        return np.array(sequence)
    
    def compute_sequence_likelihood(
        self,
        sequence: np.ndarray,
        params: np.ndarray
    ) -> float:
        """
        Compute log-likelihood of a sequence under the model.
        
        Parameters
        ----------
        sequence : ndarray
            Sequence of state indices.
        params : ndarray
            QCBM parameters.
        
        Returns
        -------
        float
            Log-likelihood.
        """
        log_likelihood = 0.0
        
        # First note likelihood
        qcbm_probs = self.qcbm.get_probabilities(params)
        log_likelihood += np.log(qcbm_probs[sequence[0]] + 1e-10)
        
        # Subsequent notes
        for i in range(1, len(sequence)):
            probs = self.get_combined_distribution(params, sequence[i-1])
            log_likelihood += np.log(probs[sequence[i]] + 1e-10)
        
        return log_likelihood
    
    def get_transition_entropy(self, state: int) -> float:
        """Get entropy of transitions from a given state."""
        probs = self.transition_matrix[state]
        return -np.sum(probs * np.log2(probs + 1e-10))
    
    def get_stationary_distribution(self) -> np.ndarray:
        """
        Compute stationary distribution of the Markov chain.
        
        Returns
        -------
        ndarray
            Stationary distribution.
        """
        # Power iteration method
        pi = np.ones(self.n_states) / self.n_states
        for _ in range(100):
            pi_new = pi @ self.transition_matrix
            if np.allclose(pi, pi_new):
                break
            pi = pi_new
        return pi
    
    def visualize_transitions(self, ax=None, top_k: int = 5):
        """
        Visualize the transition matrix.
        
        Parameters
        ----------
        ax : matplotlib Axes, optional
        top_k : int
            Show only top k transitions per state.
        """
        import matplotlib.pyplot as plt
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot heatmap
        im = ax.imshow(self.transition_matrix, cmap='viridis', aspect='auto')
        ax.set_xlabel('Next State')
        ax.set_ylabel('Current State')
        ax.set_title('Markov Transition Matrix')
        plt.colorbar(im, ax=ax, label='P(next | current)')
        
        return ax


class SecondOrderMarkovQCBM(MarkovQCBM):
    """
    Second-order Markov QCBM for richer temporal dependencies.
    
    P(x_t | x_{t-1}, x_{t-2}) - considers last two notes.
    """
    
    def __init__(self, qcbm, transition_strength: float = 0.5, smoothing: float = 0.1):
        super().__init__(qcbm, transition_strength, smoothing)
        
        # Second-order transition tensor: P(x_t | x_{t-1}, x_{t-2})
        self.transition_tensor = np.ones(
            (self.n_states, self.n_states, self.n_states)
        ) / self.n_states
    
    def fit_transitions(self, sequences: List[np.ndarray]) -> 'SecondOrderMarkovQCBM':
        """Fit second-order transitions."""
        # Also fit first-order for fallback
        super().fit_transitions(sequences)
        
        # Count second-order transitions
        transition_counts = np.zeros((self.n_states, self.n_states, self.n_states))
        
        for seq in sequences:
            for i in range(len(seq) - 2):
                s0 = int(seq[i])
                s1 = int(seq[i + 1])
                s2 = int(seq[i + 2])
                if all(0 <= s < self.n_states for s in [s0, s1, s2]):
                    transition_counts[s0, s1, s2] += 1
        
        # Smooth and normalize
        transition_counts += self.smoothing
        
        # Normalize over third dimension
        for i in range(self.n_states):
            for j in range(self.n_states):
                total = transition_counts[i, j].sum()
                if total > 0:
                    self.transition_tensor[i, j] = transition_counts[i, j] / total
        
        return self
    
    def generate_sequence(
        self,
        params: np.ndarray,
        length: int,
        initial_states: Optional[Tuple[int, int]] = None,
        temperature: float = 1.0
    ) -> np.ndarray:
        """Generate with second-order dependencies."""
        sequence = []
        qcbm_probs = self.qcbm.get_probabilities(params)
        
        # First two notes
        if initial_states is not None:
            sequence.extend(initial_states)
        else:
            # Sample first note from QCBM
            s0 = np.random.choice(self.n_states, p=qcbm_probs)
            sequence.append(s0)
            
            # Sample second from first-order Markov
            probs = self.get_combined_distribution(params, s0)
            s1 = np.random.choice(self.n_states, p=probs)
            sequence.append(s1)
        
        # Generate rest with second-order
        for _ in range(length - 2):
            s_prev2 = sequence[-2]
            s_prev1 = sequence[-1]
            
            # Get second-order Markov probs
            markov_probs = self.transition_tensor[s_prev2, s_prev1]
            
            # Combine with QCBM
            alpha = self.transition_strength
            combined = (qcbm_probs ** (1 - alpha)) * (markov_probs ** alpha)
            combined /= combined.sum()
            
            # Apply temperature
            if temperature != 1.0:
                combined = combined ** (1.0 / temperature)
                combined /= combined.sum()
            
            next_state = np.random.choice(self.n_states, p=combined)
            sequence.append(next_state)
        
        return np.array(sequence)
