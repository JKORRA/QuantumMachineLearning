"""Quantum model modules for QCBM."""

from .qcbm import QCBM
from .ansatz import HardwareEfficientAnsatz, create_ansatz
from .noise import NoiseModel, DepolarizingNoise
from .temporal import MarkovQCBM, SecondOrderMarkovQCBM

# IBM Backend is optional (requires qiskit)
try:
    from .ibm_backend import QiskitQCBM, HardwareConfig, save_ibm_credentials
    IBM_AVAILABLE = True
except ImportError:
    IBM_AVAILABLE = False

__all__ = [
    'QCBM',
    'HardwareEfficientAnsatz',
    'create_ansatz',
    'NoiseModel',
    'DepolarizingNoise',
    'MarkovQCBM',
    'SecondOrderMarkovQCBM',
]

if IBM_AVAILABLE:
    __all__.extend(['QiskitQCBM', 'HardwareConfig', 'save_ibm_credentials'])
