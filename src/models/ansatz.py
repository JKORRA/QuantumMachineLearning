"""
Ansatz Module
=============

Hardware-efficient variational ansatz for QCBM with configurable entanglement topology.
"""

import pennylane as qml
import numpy as np
from typing import List, Literal, Optional, Callable


# Type alias for entanglement topology
TopologyType = Literal['none', 'linear', 'circular', 'full']


class HardwareEfficientAnsatz:
    """
    Hardware-efficient variational ansatz for quantum circuits.
    
    This ansatz uses single-qubit rotations (RY, RZ) followed by an entangling
    layer of CNOT gates. The entanglement topology can be configured.
    
    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    n_layers : int
        Number of variational layers.
    topology : str
        Entanglement topology: 'none', 'linear', 'circular', or 'full'.
    rotation_gates : list of str
        Single-qubit gates to use (default: ['RY', 'RZ']).
    
    Attributes
    ----------
    n_params : int
        Total number of variational parameters.
    
    Example
    -------
    >>> ansatz = HardwareEfficientAnsatz(n_qubits=4, n_layers=3, topology='full')
    >>> print(f"Parameters: {ansatz.n_params}")
    >>> # Use in a PennyLane circuit
    >>> @qml.qnode(dev)
    ... def circuit(params):
    ...     ansatz.apply(params)
    ...     return qml.probs(wires=range(ansatz.n_qubits))
    """
    
    def __init__(
        self,
        n_qubits: int,
        n_layers: int = 3,
        topology: TopologyType = 'full',
        rotation_gates: Optional[List[str]] = None
    ):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.topology = topology
        self.rotation_gates = rotation_gates or ['RY', 'RZ']
        
        # Calculate number of parameters
        n_rotations_per_layer = len(self.rotation_gates) * n_qubits
        self.n_params = n_rotations_per_layer * n_layers
        
        # Get entanglement pairs
        self.entanglement_pairs = self._get_entanglement_pairs()
    
    def _get_entanglement_pairs(self) -> List[tuple]:
        """Get list of qubit pairs for CNOT gates based on topology."""
        n = self.n_qubits
        
        if self.topology == 'none':
            # No entanglement (separable circuit)
            return []
        
        elif self.topology == 'linear':
            # Linear chain: 0-1, 1-2, 2-3, ...
            return [(i, i + 1) for i in range(n - 1)]
        
        elif self.topology == 'circular':
            # Circular: linear + connection from last to first
            pairs = [(i, i + 1) for i in range(n - 1)]
            pairs.append((n - 1, 0))
            return pairs
        
        elif self.topology == 'full':
            # All-to-all connectivity
            return [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        else:
            raise ValueError(f"Unknown topology: {self.topology}")
    
    def apply(self, params: np.ndarray) -> None:
        """
        Apply the ansatz to the quantum circuit.
        
        Must be called within a PennyLane QNode context.
        
        Parameters
        ----------
        params : ndarray
            Variational parameters of shape (n_params,) or (n_layers, n_qubits, n_rotations).
        """
        params = np.asarray(params).flatten()
        
        if len(params) != self.n_params:
            raise ValueError(
                f"Expected {self.n_params} parameters, got {len(params)}"
            )
        
        # Reshape for easier indexing
        n_rot = len(self.rotation_gates)
        params = params.reshape(self.n_layers, self.n_qubits, n_rot)
        
        for layer in range(self.n_layers):
            # Single-qubit rotations
            for qubit in range(self.n_qubits):
                for rot_idx, gate_name in enumerate(self.rotation_gates):
                    angle = params[layer, qubit, rot_idx]
                    gate = getattr(qml, gate_name)
                    gate(angle, wires=qubit)
            
            # Entangling layer
            for control, target in self.entanglement_pairs:
                qml.CNOT(wires=[control, target])
    
    def get_initial_params(
        self, 
        strategy: str = 'random',
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate initial parameters.
        
        Parameters
        ----------
        strategy : str
            'random' for uniform random, 'zeros' for all zeros,
            'small' for small random values.
        seed : int, optional
            Random seed for reproducibility.
        
        Returns
        -------
        ndarray
            Initial parameters of shape (n_params,).
        """
        if seed is not None:
            np.random.seed(seed)
        
        if strategy == 'random':
            return np.random.uniform(0, 2 * np.pi, self.n_params)
        elif strategy == 'zeros':
            return np.zeros(self.n_params)
        elif strategy == 'small':
            return np.random.uniform(-0.1, 0.1, self.n_params)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def __repr__(self) -> str:
        return (
            f"HardwareEfficientAnsatz(n_qubits={self.n_qubits}, "
            f"n_layers={self.n_layers}, topology='{self.topology}', "
            f"n_params={self.n_params})"
        )


def create_ansatz(
    n_qubits: int,
    n_layers: int = 3,
    topology: TopologyType = 'full',
    **kwargs
) -> HardwareEfficientAnsatz:
    """
    Factory function to create an ansatz.
    
    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    n_layers : int
        Number of variational layers.
    topology : str
        Entanglement topology.
    
    Returns
    -------
    HardwareEfficientAnsatz
        Configured ansatz object.
    """
    return HardwareEfficientAnsatz(
        n_qubits=n_qubits,
        n_layers=n_layers,
        topology=topology,
        **kwargs
    )


# =============================================================================
# Visualization utilities
# =============================================================================

def draw_ansatz(ansatz: HardwareEfficientAnsatz, show: bool = True) -> Optional[str]:
    """
    Draw the circuit structure of an ansatz.
    
    Parameters
    ----------
    ansatz : HardwareEfficientAnsatz
        The ansatz to visualize.
    show : bool
        Whether to print the circuit.
    
    Returns
    -------
    str or None
        String representation of the circuit.
    """
    dev = qml.device('default.qubit', wires=ansatz.n_qubits)
    
    @qml.qnode(dev)
    def circuit(params):
        ansatz.apply(params)
        return qml.probs(wires=range(ansatz.n_qubits))
    
    params = ansatz.get_initial_params(strategy='zeros')
    circuit(params)
    
    drawer = qml.draw(circuit)
    circuit_str = drawer(params)
    
    if show:
        print(circuit_str)
    
    return circuit_str
