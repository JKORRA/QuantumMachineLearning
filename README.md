# Quantum Circuit Born Machines for Music Generation

## Master's Thesis Project - Quantum Machine Learning

**Author:** Jacopo  
**University:** University of Trento  
**Date:** January 2026

---

## Abstract

This thesis investigates whether **Quantum Machine Learning (QML)** can generate symbolic music (MIDI) better or differently than classical methods. Specifically, we test the hypothesis that **Quantum Entanglement** is a necessary physical resource for modeling complex, non-local musical correlations.

We implement a **Quantum Circuit Born Machine (QCBM)** that uses the Born Rule to sample music directly from a quantum wavefunction, acting as a native probability distribution sampler.

---

## Project Structure

```text
QuantumMachineLearning/
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── config.py                   # Global configuration
├── .gitignore                  # Git ignore rules
│
├── src/                        # Source code modules
│   ├── data/                   # Data processing (MIDI parsing, encoding, datasets)
│   ├── models/                 # Quantum models (QCBM, hardware ansatz, noise models)
│   ├── training/               # Training utilities (Loss functions, trainers)
│   └── utils/                  # Utilities (Visualization, metrics, figure savers)
│
├── notebooks/                  # Experiment notebooks (11 total)
│   ├── 01_data_exploration.ipynb
│   ├── 02_qcbm_basics.ipynb
│   ├── 03_baseline_simple.ipynb     # Entanglement advantage
│   ├── 04_scalability.ipynb
│   ├── 05_noise_robustness.ipynb
│   ├── 06_topology_battle.ipynb     # Optimal topology exploration
│   ├── 07_optimizer_battle.ipynb
│   ├── 08_loss_function_battle.ipynb
│   ├── 09_final_validation.ipynb
│   ├── 10_music_generation.ipynb
│   └── 11_advanced_features.ipynb   # IBM hardware validation
│
├── results/                    # Experiment results
│   ├── figures/                # Generated plots and visualizations
│   ├── logs/                   # Training logs
│   ├── models/                 # Saved model parameters
│   └── music_generation/       # Generated MIDI files
│
├── data/                       # Data files
│   └── midi/                   # Raw MIDI files used for training
│
└── paper/                      # Documentation and reports
    ├── QuantumMusic.pdf        # Final report
    ├── QuantumMusic.tex        # LaTeX source of the report
    ├── Spiegazione_Report_IT.md  # Italian explanation
    └── references.bib          # Bibliography
```

---

## Research Questions

1. **RQ1:** Is quantum entanglement necessary for modeling simple musical distributions?
2. **RQ2:** How does model scalability (4 vs 8 qubits) affect learning?
3. **RQ3:** What is the noise threshold for NISQ-compatible music generation?
4. **RQ4:** Which entanglement topology is optimal for correlated data?
5. **RQ5:** Which optimizer performs best on the QCBM loss landscape?
6. **RQ6:** Is MMD superior to KL Divergence for discrete quantum distributions?

---

## Experiments & Key Results

Through a systematic series of experiments in our `notebooks/`, we explored the feasibility and advantages of utilizing QCBMs for music generation.

### Phase 1: Baselines & Exploration
- **Entanglement Advantage:** Entangled circuits demonstrate a significant performance boost over classical separable distributions, achieving **82.6% fidelity** compared to 63.3% (a +19.3 percentage point improvement). 
- **Scalability & Noise Robustness:** We tested 4-qubit vs 8-qubit scales and introduced depolarizing noise (0% to 15%). The model tolerates up to **5% gate error** while maintaining graceful degradation, making it suitable for NISQ (Noisy Intermediate-Scale Quantum) devices.

### Phase 2: Optimization (The Battles)
- **Topology:** The linear topology emerged as optimal, balancing expressivity and trainability with only 33 parameters.
- **Optimizers & Loss:** The **SLSQP** optimizer paired with the Maximum Mean Discrepancy (**MMD**) loss function provided the most stable gradients and superior training fidelity.

### Phase 3: Hardware Validation
- **Real Quantum Hardware Testing:** We validated our QCBM on the real IBM `ibm_torino` quantum computer. It achieved an impressive **82.8% fidelity**, showing only a ~14% degradation from the ideal simulator results.

### Phase 4: Music Generation
- The fully trained and optimized model was deployed to generate novel symbolic music. The generated `.mid` files successfully capture melodic and temporal structures and can be found in `results/music_generation/`.

---

## Quick Start

### Setup Environment

To run the experiments or generate music, set up the environment using Python:

```bash
# 1. Clone the repository and move to the directory
cd QuantumMachineLearning

# 2. Setup virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Reviewing the Experiments

The results of all experiments are saved and can be reviewed directly by opening the Jupyter Notebooks:

```bash
jupyter notebook notebooks/
```

Key notebooks to check out:
- `03_baseline_simple.ipynb`: Demonstrates the entanglement advantage.
- `06_topology_battle.ipynb`: Highlights the search for optimal circuit architecture.
- `11_advanced_features.ipynb`: Details the real hardware execution via IBM Quantum.

---

## Technology Stack

- **Quantum Framework:** PennyLane 0.34+ (primary), Qiskit 1.0+ (hardware interface)
- **Classical ML:** NumPy, SciPy (optimizers)
- **Music Processing:** Mido (MIDI I/O)
- **Visualization:** Matplotlib, Seaborn
- **Hardware:** IBM Quantum (`ibm_torino`, 133-qubit Eagle r3)

---

## License

This project is for academic purposes only (University of Trento - Quantum Machine Learning Course).

## Acknowledgments

- **IBM Quantum** for providing access to real quantum hardware.
- **PennyLane** team for the comprehensive QML framework.
- **Nintendo (Super Mario Bros)** for the iconic melody dataset used in early training stages.
