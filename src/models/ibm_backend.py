"""
IBM Quantum Backend Module
==========================

Integration with IBM Quantum hardware via Qiskit.
Allows running QCBM on real quantum computers.
"""

import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
import warnings

# Check for qiskit availability
try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit import Parameter
    from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Session
    from qiskit_ibm_runtime import Options
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    warnings.warn(
        "Qiskit not installed. Install with: pip install qiskit qiskit-ibm-runtime"
    )


@dataclass
class HardwareConfig:
    """Configuration for IBM Quantum hardware."""
    backend_name: str = "ibm_torino"  # 133 qubit Heron processor
    shots: int = 4096
    optimization_level: int = 3
    resilience_level: int = 1  # Error mitigation
    
    # Alternative backends (2026 Open Plan)
    AVAILABLE_BACKENDS = [
        "ibm_fez",         # 156 qubits, Heron
        "ibm_marrakesh",   # 156 qubits, Heron
        "ibm_torino",      # 133 qubits, Heron
    ]


class QiskitQCBM:
    """
    QCBM implementation using Qiskit for IBM Quantum hardware.
    
    This class provides the same interface as the PennyLane QCBM but
    uses Qiskit circuits that can run on real IBM quantum computers.
    
    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    n_layers : int
        Number of variational layers.
    topology : str
        Entanglement topology.
    hardware_config : HardwareConfig, optional
        Hardware configuration.
    
    Example
    -------
    >>> # Local simulation
    >>> qcbm = QiskitQCBM(n_qubits=4, n_layers=3)
    >>> probs = qcbm.get_probabilities(params)
    
    >>> # Run on real hardware
    >>> qcbm.connect_to_ibm("your-api-token")
    >>> probs = qcbm.run_on_hardware(params)
    """
    
    def __init__(
        self,
        n_qubits: int,
        n_layers: int = 3,
        topology: str = 'full',
        hardware_config: Optional[HardwareConfig] = None
    ):
        if not QISKIT_AVAILABLE:
            raise ImportError(
                "Qiskit is required for IBM Quantum integration. "
                "Install with: pip install qiskit qiskit-ibm-runtime"
            )
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.topology = topology
        self.n_states = 2 ** n_qubits
        self.hardware_config = hardware_config or HardwareConfig()
        
        # Calculate number of parameters
        self.n_params = 2 * n_qubits * n_layers  # RY and RZ per qubit per layer
        
        # Create parameterized circuit
        self._create_circuit()
        
        # IBM Quantum connection
        self.service = None
        self.backend = None
        self._connected = False
    
    def _get_entanglement_pairs(self) -> List[Tuple[int, int]]:
        """Get entanglement pairs based on topology."""
        n = self.n_qubits
        
        if self.topology == 'none':
            return []
        elif self.topology == 'linear':
            return [(i, i + 1) for i in range(n - 1)]
        elif self.topology == 'circular':
            pairs = [(i, i + 1) for i in range(n - 1)]
            pairs.append((n - 1, 0))
            return pairs
        elif self.topology == 'full':
            return [(i, j) for i in range(n) for j in range(i + 1, n)]
        else:
            raise ValueError(f"Unknown topology: {self.topology}")
    
    def _create_circuit(self):
        """Create the parameterized Qiskit circuit."""
        self.params = [
            Parameter(f'θ_{i}') for i in range(self.n_params)
        ]
        
        self.circuit = QuantumCircuit(self.n_qubits)
        
        param_idx = 0
        entanglement_pairs = self._get_entanglement_pairs()
        
        for layer in range(self.n_layers):
            # Single-qubit rotations
            for qubit in range(self.n_qubits):
                self.circuit.ry(self.params[param_idx], qubit)
                param_idx += 1
                self.circuit.rz(self.params[param_idx], qubit)
                param_idx += 1
            
            # Entangling layer
            for control, target in entanglement_pairs:
                self.circuit.cx(control, target)
        
        # Measurement
        self.circuit.measure_all()
    
    def get_bound_circuit(self, params: np.ndarray) -> QuantumCircuit:
        """
        Create circuit with bound parameter values.
        
        Parameters
        ----------
        params : ndarray
            Parameter values.
        
        Returns
        -------
        QuantumCircuit
            Circuit with concrete parameter values.
        """
        param_dict = {self.params[i]: params[i] for i in range(len(params))}
        return self.circuit.assign_parameters(param_dict)
    
    def connect_to_ibm(
        self, 
        token: Optional[str] = None,
        channel: str = "ibm_quantum_platform",
        instance: Optional[str] = None
    ):
        """
        Connect to IBM Quantum services.
        
        Parameters
        ----------
        token : str, optional
            IBM Quantum API token. If None, uses saved credentials.
        channel : str
            Service channel ('ibm_quantum_platform' or 'ibm_cloud').
        instance : str, optional
            Service instance (not needed for ibm_quantum_platform).
        """
        if token:
            self.service = QiskitRuntimeService(
                channel=channel,
                token=token
            )
        else:
            # Try to load saved credentials
            self.service = QiskitRuntimeService()
        
        # Get backend
        self.backend = self.service.backend(self.hardware_config.backend_name)
        self._connected = True
        
        print(f" Connected to IBM Quantum")
        print(f"   Backend: {self.backend.name}")
        print(f"   Qubits: {self.backend.num_qubits}")
        print(f"   Status: {self.backend.status().status_msg}")
    
    def get_probabilities_simulator(self, params: np.ndarray) -> np.ndarray:
        """
        Get probabilities using local Qiskit simulator.
        
        Parameters
        ----------
        params : ndarray
            Circuit parameters.
        
        Returns
        -------
        ndarray
            Probability distribution.
        """
        from qiskit_aer import AerSimulator
        
        # Bind parameters
        bound_circuit = self.get_bound_circuit(params)
        
        # Run on simulator
        simulator = AerSimulator()
        transpiled = transpile(bound_circuit, simulator)
        
        job = simulator.run(transpiled, shots=self.hardware_config.shots)
        result = job.result()
        counts = result.get_counts()
        
        # Convert counts to probabilities
        probs = np.zeros(self.n_states)
        for bitstring, count in counts.items():
            # Qiskit uses little-endian, so reverse
            idx = int(bitstring[::-1], 2)
            probs[idx] = count
        
        return probs / probs.sum()
    
    def get_probabilities(self, params: np.ndarray) -> np.ndarray:
        """
        Get probabilities (uses simulator by default).
        
        For hardware execution, use run_on_hardware().
        """
        return self.get_probabilities_simulator(params)
    
    def run_on_hardware(
        self, 
        params: np.ndarray,
        error_mitigation: bool = True
    ) -> np.ndarray:
        """
        Run circuit on real IBM Quantum hardware.
        
        Parameters
        ----------
        params : ndarray
            Circuit parameters.
        error_mitigation : bool
            Whether to use error mitigation.
        
        Returns
        -------
        ndarray
            Measured probability distribution.
        
        Note
        ----
        This queues a job on real hardware, which may take
        minutes to hours depending on queue length.
        """
        if not self._connected:
            raise RuntimeError(
                "Not connected to IBM Quantum. Call connect_to_ibm() first."
            )
        
        # Bind parameters and transpile for hardware
        bound_circuit = self.get_bound_circuit(params)
        transpiled = transpile(
            bound_circuit, 
            self.backend,
            optimization_level=self.hardware_config.optimization_level
        )
        
        print(f" Submitting job to {self.backend.name}...")
        print(f"   Circuit depth: {transpiled.depth()}")
        print(f"   Gate count: {transpiled.count_ops()}")
        
        # Run with SamplerV2 primitive (Qiskit 1.0+ API)
        from qiskit_ibm_runtime import SamplerV2 as Sampler
        
        sampler = Sampler(mode=self.backend)
        
        # Submit job
        job = sampler.run([transpiled], shots=self.hardware_config.shots)
        
        print(f"   Job ID: {job.job_id()}")
        print("   Waiting for results...")
        
        result = job.result()
        
        # Extract probabilities from counts (SamplerV2 returns DataBin)
        pub_result = result[0]
        counts = pub_result.data.meas.get_counts()
        
        probs = np.zeros(self.n_states)
        total_shots = sum(counts.values())
        for bitstring, count in counts.items():
            # Convert bitstring to integer index
            idx = int(bitstring, 2)
            probs[idx] = count / total_shots
        
        print(" Hardware execution complete!")
        
        return probs
    
    def sample(
        self, 
        params: np.ndarray, 
        n_samples: int,
        use_hardware: bool = False
    ) -> np.ndarray:
        """
        Sample from the distribution.
        
        Parameters
        ----------
        params : ndarray
            Circuit parameters.
        n_samples : int
            Number of samples.
        use_hardware : bool
            Whether to use real hardware.
        
        Returns
        -------
        ndarray
            Sampled state indices.
        """
        if use_hardware:
            probs = self.run_on_hardware(params)
        else:
            probs = self.get_probabilities(params)
        
        return np.random.choice(self.n_states, size=n_samples, p=probs)
    
    def get_initial_params(self, seed: Optional[int] = None) -> np.ndarray:
        """Get random initial parameters."""
        if seed is not None:
            np.random.seed(seed)
        return np.random.uniform(0, 2 * np.pi, self.n_params)
    
    def draw_circuit(self, params: Optional[np.ndarray] = None) -> str:
        """
        Get text representation of the circuit.
        
        Parameters
        ----------
        params : ndarray, optional
            If provided, shows bound circuit.
        
        Returns
        -------
        str
            Circuit diagram.
        """
        if params is not None:
            circuit = self.get_bound_circuit(params)
        else:
            circuit = self.circuit
        
        return circuit.draw(output='text')
    
    def get_hardware_info(self) -> Dict[str, Any]:
        """Get information about the connected hardware."""
        if not self._connected:
            return {"error": "Not connected"}
        
        return {
            "name": self.backend.name,
            "num_qubits": self.backend.num_qubits,
            "basis_gates": self.backend.operation_names,
            "status": self.backend.status().status_msg,
            "pending_jobs": self.backend.status().pending_jobs,
        }
    
    def estimate_cost(self, n_circuits: int = 1) -> Dict[str, Any]:
        """
        Estimate the runtime cost.
        
        Parameters
        ----------
        n_circuits : int
            Number of circuits to run.
        
        Returns
        -------
        dict
            Cost estimation.
        """
        shots = self.hardware_config.shots
        return {
            "total_shots": shots * n_circuits,
            "estimated_time": f"{n_circuits * 2} - {n_circuits * 10} minutes",
            "note": "Actual time depends on queue length"
        }


def compare_simulator_vs_hardware(
    qcbm: QiskitQCBM,
    params: np.ndarray,
    target_dist: np.ndarray
) -> Dict[str, float]:
    """
    Compare QCBM performance on simulator vs hardware.
    
    Parameters
    ----------
    qcbm : QiskitQCBM
        Qiskit-based QCBM instance.
    params : ndarray
        Trained parameters.
    target_dist : ndarray
        Target distribution.
    
    Returns
    -------
    dict
        Comparison metrics.
    """
    from src.training.loss_functions import mmd_loss, fidelity
    
    # Simulator results
    sim_probs = qcbm.get_probabilities_simulator(params)
    
    results = {
        "simulator_fidelity": fidelity(sim_probs, target_dist),
        "simulator_mmd": mmd_loss(sim_probs, target_dist),
    }
    
    # Hardware results (if connected)
    if qcbm._connected:
        hw_probs = qcbm.run_on_hardware(params)
        results["hardware_fidelity"] = fidelity(hw_probs, target_dist)
        results["hardware_mmd"] = mmd_loss(hw_probs, target_dist)
        results["sim_hw_fidelity"] = fidelity(sim_probs, hw_probs)
    
    return results


# Utility function to save IBM credentials
def save_ibm_credentials(token: str, overwrite: bool = False):
    """
    Save IBM Quantum credentials for future use.
    
    Parameters
    ----------
    token : str
        IBM Quantum API token.
    overwrite : bool
        Whether to overwrite existing credentials.
    """
    if not QISKIT_AVAILABLE:
        raise ImportError("Qiskit not installed")
    
    QiskitRuntimeService.save_account(
        channel="ibm_quantum_platform",
        token=token,
        overwrite=overwrite
    )
    print(" IBM Quantum credentials saved!")
