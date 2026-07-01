"""
Quantum Circuit Born Machine (QCBM)
===================================

The main quantum generative model for music generation.

The QCBM uses the Born rule to sample from a quantum circuit's output
distribution, making it a natural probability distribution generator.
"""

import pennylane as qml
import numpy as np
from typing import Optional, Callable, Dict, Any, Tuple
from dataclasses import dataclass

from .ansatz import HardwareEfficientAnsatz, TopologyType
from .noise import NoiseModel


@dataclass
class QCBMConfig:
    """Configuration for QCBM."""
    n_qubits: int
    n_layers: int = 3
    topology: TopologyType = 'full'
    noise_model: Optional[NoiseModel] = None
    device_name: str = 'default.qubit'


class QCBM:
    """
    Quantum Circuit Born Machine.
    
    A generative model that uses the Born rule to sample from the output
    distribution of a parameterized quantum circuit.
    
    The probability of measuring a bitstring |x⟩ is given by:
        P(x) = |⟨x|ψ(θ)⟩|²
    
    Parameters
    ----------
    n_qubits : int
        Number of qubits (determines output space size 2^n_qubits).
    n_layers : int
        Number of variational layers in the ansatz.
    topology : str
        Entanglement topology: 'none', 'linear', 'circular', 'full'.
    noise_model : NoiseModel, optional
        Noise model for NISQ simulation.
    device_name : str
        PennyLane device to use.
    
    Attributes
    ----------
    n_params : int
        Number of variational parameters.
    ansatz : HardwareEfficientAnsatz
        The variational ansatz used in the circuit.
    
    Example
    -------
    >>> qcbm = QCBM(n_qubits=4, n_layers=3, topology='full')
    >>> params = qcbm.get_initial_params()
    >>> probs = qcbm.get_probabilities(params)
    >>> samples = qcbm.sample(params, n_samples=1000)
    """
    
    def __init__(
        self,
        n_qubits: int,
        n_layers: int = 3,
        topology: TopologyType = 'full',
        noise_model: Optional[NoiseModel] = None,
        device_name: str = 'default.qubit'
    ):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.topology = topology
        self.noise_model = noise_model
        self.device_name = device_name
        
        # Create ansatz
        self.ansatz = HardwareEfficientAnsatz(
            n_qubits=n_qubits,
            n_layers=n_layers,
            topology=topology
        )
        self.n_params = self.ansatz.n_params
        
        # Output space size
        self.n_states = 2 ** n_qubits
        
        # Create quantum device and circuit
        self._create_circuit()
    
    def _create_circuit(self) -> None:
        """Create the PennyLane quantum circuit."""
        # Choose device based on whether we need noise simulation
        if self.noise_model is not None:
            # Mixed state simulator for noise
            self.device = qml.device('default.mixed', wires=self.n_qubits)
        else:
            self.device = qml.device(self.device_name, wires=self.n_qubits)
        
        @qml.qnode(self.device, interface='autograd')
        def circuit(params):
            # Apply the variational ansatz
            self.ansatz.apply(params)
            
            # Apply noise if present
            if self.noise_model is not None:
                self.noise_model.apply(list(range(self.n_qubits)))
            
            # Return probabilities of all computational basis states
            return qml.probs(wires=range(self.n_qubits))
        
        self._circuit = circuit
    
    def get_probabilities(self, params: np.ndarray) -> np.ndarray:
        """
        Get the output probability distribution.
        
        Parameters
        ----------
        params : ndarray
            Variational parameters.
        
        Returns
        -------
        ndarray
            Probability distribution over 2^n_qubits states.
        """
        return np.asarray(self._circuit(params))
    
    def sample(
        self, 
        params: np.ndarray, 
        n_samples: int = 1000,
        return_counts: bool = False
    ) -> np.ndarray:
        """
        Sample from the QCBM distribution.
        
        Parameters
        ----------
        params : ndarray
            Variational parameters.
        n_samples : int
            Number of samples to generate.
        return_counts : bool
            If True, return histogram counts instead of samples.
        
        Returns
        -------
        ndarray
            Either array of samples (shape: n_samples,) or counts (shape: n_states,).
        """
        probs = self.get_probabilities(params)
        
        if return_counts:
            # Multinomial sampling for counts
            counts = np.random.multinomial(n_samples, probs)
            return counts
        else:
            # Direct sampling
            samples = np.random.choice(self.n_states, size=n_samples, p=probs)
            return samples
    
    def get_initial_params(
        self,
        strategy: str = 'random',
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Get initial parameters for training.
        
        Parameters
        ----------
        strategy : str
            'random', 'zeros', or 'small'.
        seed : int, optional
            Random seed.
        
        Returns
        -------
        ndarray
            Initial parameters.
        """
        return self.ansatz.get_initial_params(strategy=strategy, seed=seed)
    
    def to_config(self) -> QCBMConfig:
        """Convert to configuration dataclass."""
        return QCBMConfig(
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            topology=self.topology,
            noise_model=self.noise_model,
            device_name=self.device_name
        )
    
    @classmethod
    def from_config(cls, config: QCBMConfig) -> 'QCBM':
        """Create QCBM from configuration."""
        return cls(
            n_qubits=config.n_qubits,
            n_layers=config.n_layers,
            topology=config.topology,
            noise_model=config.noise_model,
            device_name=config.device_name
        )
    
    def draw(self) -> str:
        """Draw the circuit structure."""
        params = self.get_initial_params(strategy='zeros')
        return qml.draw(self._circuit)(params)
    
    def __repr__(self) -> str:
        noise_str = f", noise={self.noise_model}" if self.noise_model else ""
        return (
            f"QCBM(n_qubits={self.n_qubits}, n_layers={self.n_layers}, "
            f"topology='{self.topology}'{noise_str})"
        )


# =============================================================================
# Factory functions
# =============================================================================

def create_qcbm(
    n_qubits: int,
    n_layers: int = 3,
    topology: TopologyType = 'full',
    noise_probability: float = 0.0,
    **kwargs
) -> QCBM:
    """
    Factory function to create a QCBM.
    
    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    n_layers : int
        Number of layers.
    topology : str
        Entanglement topology.
    noise_probability : float
        Depolarizing noise probability (0 for noiseless).
    
    Returns
    -------
    QCBM
        Configured QCBM model.
    """
    from .noise import DepolarizingNoise
    
    noise_model = None
    if noise_probability > 0:
        noise_model = DepolarizingNoise(noise_probability)
    
    return QCBM(
        n_qubits=n_qubits,
        n_layers=n_layers,
        topology=topology,
        noise_model=noise_model,
        **kwargs
    )


def create_separable_qcbm(n_qubits: int, n_layers: int = 3, **kwargs) -> QCBM:
    """Create a QCBM without entanglement (classical baseline)."""
    return create_qcbm(n_qubits, n_layers, topology='none', **kwargs)


def create_entangled_qcbm(n_qubits: int, n_layers: int = 3, **kwargs) -> QCBM:
    """Create a QCBM with full entanglement."""
    return create_qcbm(n_qubits, n_layers, topology='full', **kwargs)
