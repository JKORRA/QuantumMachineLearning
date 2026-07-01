"""
Trainer Module
==============

Training loop for QCBM optimization.
"""

import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Dict, Any, Literal
from time import time
from tqdm import tqdm

import sys
sys.path.insert(0, str(__file__).rsplit('/', 3)[0])

from src.models.qcbm import QCBM
from src.training.loss_functions import get_loss_function, mmd_loss, fidelity


@dataclass
class TrainingResult:
    """Container for training results."""
    final_params: np.ndarray
    final_loss: float
    loss_history: List[float]
    fidelity_history: List[float]
    time_per_step: List[float]
    total_time: float
    n_iterations: int
    optimizer: str
    loss_function: str
    converged: bool
    
    # Additional metrics
    final_fidelity: float = 0.0
    best_loss: float = float('inf')
    best_params: Optional[np.ndarray] = None
    
    def __post_init__(self):
        if self.fidelity_history:
            self.final_fidelity = self.fidelity_history[-1]
        self.best_loss = min(self.loss_history) if self.loss_history else float('inf')
    
    def summary(self) -> str:
        """Get a summary string of training results."""
        return (
            f"Training Summary:\n"
            f"  Optimizer: {self.optimizer}\n"
            f"  Loss Function: {self.loss_function}\n"
            f"  Iterations: {self.n_iterations}\n"
            f"  Final Loss: {self.final_loss:.6f}\n"
            f"  Best Loss: {self.best_loss:.6f}\n"
            f"  Final Fidelity: {self.final_fidelity:.4f}\n"
            f"  Total Time: {self.total_time:.2f}s\n"
            f"  Time/Step: {np.mean(self.time_per_step):.4f}s\n"
            f"  Converged: {self.converged}"
        )


OptimizerType = Literal['Powell', 'COBYLA', 'SLSQP', 'L-BFGS-B', 'Nelder-Mead']


class Trainer:
    """
    Trainer for QCBM models.
    
    Uses scipy.optimize for classical optimization of quantum circuit parameters.
    
    Parameters
    ----------
    qcbm : QCBM
        The quantum model to train.
    target_distribution : ndarray
        Target probability distribution to learn.
    loss_function : str
        Loss function name: 'mmd', 'kl', 'tv', 'hellinger'.
    optimizer : str
        Scipy optimizer: 'Powell', 'COBYLA', 'SLSQP', 'L-BFGS-B', 'Nelder-Mead'.
    
    Example
    -------
    >>> qcbm = QCBM(n_qubits=4, n_layers=3)
    >>> trainer = Trainer(qcbm, target_dist, loss_function='mmd', optimizer='Powell')
    >>> result = trainer.train(n_iterations=200)
    >>> print(result.summary())
    """
    
    def __init__(
        self,
        qcbm: QCBM,
        target_distribution: np.ndarray,
        loss_function: str = 'mmd',
        optimizer: OptimizerType = 'Powell',
        loss_kwargs: Optional[Dict[str, Any]] = None
    ):
        self.qcbm = qcbm
        self.target = target_distribution
        self.loss_function_name = loss_function
        self.optimizer_name = optimizer
        
        # Get loss function
        loss_kwargs = loss_kwargs or {}
        self.loss_fn = get_loss_function(loss_function, **loss_kwargs)
        
        # Training state
        self.loss_history: List[float] = []
        self.fidelity_history: List[float] = []
        self.time_per_step: List[float] = []
        self.current_params: Optional[np.ndarray] = None
        self.best_params: Optional[np.ndarray] = None
        self.best_loss: float = float('inf')
        
        # Callback counter
        self._iteration = 0
        self._pbar: Optional[tqdm] = None
    
    def _objective(self, params: np.ndarray) -> float:
        """Objective function for optimization."""
        step_start = time()
        
        # Get model distribution
        model_dist = self.qcbm.get_probabilities(params)
        
        # Compute loss
        loss = self.loss_fn(model_dist, self.target)
        
        # Compute fidelity for monitoring
        fid = fidelity(model_dist, self.target)
        
        # Record history
        self.loss_history.append(loss)
        self.fidelity_history.append(fid)
        self.time_per_step.append(time() - step_start)
        
        # Track best
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_params = params.copy()
        
        # Update progress bar
        self._iteration += 1
        if self._pbar is not None:
            self._pbar.update(1)
            self._pbar.set_postfix({
                'loss': f'{loss:.4f}',
                'fidelity': f'{fid:.4f}'
            })
        
        return loss
    
    def train(
        self,
        n_iterations: int = 200,
        initial_params: Optional[np.ndarray] = None,
        verbose: bool = True,
        seed: Optional[int] = None
    ) -> TrainingResult:
        """
        Train the QCBM.
        
        Parameters
        ----------
        n_iterations : int
            Maximum number of iterations.
        initial_params : ndarray, optional
            Initial parameters. If None, random initialization.
        verbose : bool
            Whether to show progress bar.
        seed : int, optional
            Random seed for reproducibility.
        
        Returns
        -------
        TrainingResult
            Training results including loss history and final parameters.
        """
        # Reset state
        self.loss_history = []
        self.fidelity_history = []
        self.time_per_step = []
        self._iteration = 0
        self.best_loss = float('inf')
        
        # Initialize parameters
        if initial_params is None:
            initial_params = self.qcbm.get_initial_params(strategy='random', seed=seed)
        
        self.current_params = initial_params.copy()
        
        # Setup optimizer options
        optimizer_options = {
            'maxiter': n_iterations,
            'maxfev': n_iterations * 10,  # Max function evaluations
        }
        
        # Progress bar
        if verbose:
            self._pbar = tqdm(total=n_iterations, desc='Training QCBM')
        
        # Run optimization
        start_time = time()
        
        try:
            result = minimize(
                self._objective,
                initial_params,
                method=self.optimizer_name,
                options=optimizer_options
            )
            converged = result.success
            final_params = result.x
        except Exception as e:
            print(f"Optimization error: {e}")
            converged = False
            final_params = self.best_params if self.best_params is not None else initial_params
        
        total_time = time() - start_time
        
        # Close progress bar
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None
        
        # Create result
        return TrainingResult(
            final_params=final_params,
            final_loss=self.loss_history[-1] if self.loss_history else float('inf'),
            loss_history=self.loss_history,
            fidelity_history=self.fidelity_history,
            time_per_step=self.time_per_step,
            total_time=total_time,
            n_iterations=len(self.loss_history),
            optimizer=self.optimizer_name,
            loss_function=self.loss_function_name,
            converged=converged,
            best_params=self.best_params
        )
    
    def train_with_restarts(
        self,
        n_iterations: int = 200,
        n_restarts: int = 3,
        verbose: bool = True,
        seed: Optional[int] = None
    ) -> TrainingResult:
        """
        Train with multiple random restarts and return best result.
        
        Parameters
        ----------
        n_iterations : int
            Iterations per restart.
        n_restarts : int
            Number of random restarts.
        verbose : bool
            Whether to show progress.
        seed : int, optional
            Base random seed.
        
        Returns
        -------
        TrainingResult
            Best result across all restarts.
        """
        best_result = None
        best_final_loss = float('inf')
        
        for restart in range(n_restarts):
            if verbose:
                print(f"\n--- Restart {restart + 1}/{n_restarts} ---")
            
            restart_seed = seed + restart if seed is not None else None
            result = self.train(
                n_iterations=n_iterations,
                verbose=verbose,
                seed=restart_seed
            )
            
            if result.final_loss < best_final_loss:
                best_final_loss = result.final_loss
                best_result = result
        
        return best_result


def train_qcbm(
    qcbm: QCBM,
    target_distribution: np.ndarray,
    n_iterations: int = 200,
    loss_function: str = 'mmd',
    optimizer: str = 'Powell',
    verbose: bool = True,
    seed: Optional[int] = None
) -> TrainingResult:
    """
    Convenience function to train a QCBM.
    
    Parameters
    ----------
    qcbm : QCBM
        Model to train.
    target_distribution : ndarray
        Target distribution.
    n_iterations : int
        Number of iterations.
    loss_function : str
        Loss function name.
    optimizer : str
        Optimizer name.
    verbose : bool
        Show progress.
    seed : int, optional
        Random seed.
    
    Returns
    -------
    TrainingResult
        Training results.
    """
    trainer = Trainer(
        qcbm=qcbm,
        target_distribution=target_distribution,
        loss_function=loss_function,
        optimizer=optimizer
    )
    
    return trainer.train(
        n_iterations=n_iterations,
        verbose=verbose,
        seed=seed
    )
