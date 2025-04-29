# Hasegawa-Wakatani-FNO

This Repository contains the code to my master's project "Application of Fourier Neural Operator to 2D plasma turbulence model (2025)".

# Quick Summary

Implementations of Fourier Neural Operator (FNO, [Z. Li, N. Kovachki (2020)](https://arxiv.org/abs/2010.08895)) and U-Net enhanced Fourier Neural Operator (U-FNO), to extrapolate 2D plasma tubulent flow based on the Hasegawa-Wakatani model for 2D collisional drift waves ([M. Wakatani, A. Hasegawa (1984)](https://pubs.aip.org/aip/pfl/article/27/3/611/824784/A-collisional-drift-wave-description-of-plasma)).

The Neural Networks are trained to iteratively predict snapshots at the next time step from previous time steps over short sequences of the fluid flow. A curriculum learning approach is employed to transition linearly over the epochs from ground truth input (teacher forcing) to solely rely on previous predictions (free running) over.


![Sample Image](https://raw.githubusercontent.com/LionelSafar/Hasegawa-Wakatani-FNO/main/Data/sample_images/n_snaps_-1.png)
*Figure 1: Overview of flow extrapolation of each model type. FNO-based models maintain what visually resembles a quasi-equilibrium state of the HW system in the isotropic turbulent regime.*

# Scripts

The following scripts can be found in this repository:
- ~/data_generation/run_simulations.py
  - Runs Direct Numerical Simulations of the HW2D system based on an open source Implementation ([Greif, R. (2023)](https://github.com/the-rccg/hw2d))
- ~/model_training/training_run.py
  - Directly preprocesses the simulation data and trains the selected model with an iterative curriculum learning approach.
- ~/model_optimization/optimization_run.py
  - Optimizes a selection of hyperparameters for a given model, based on Bayesian Optimisation (Parzen Tree-Estimator Algorithm)
- ~/model_evaluation/evaluation_run.py
  - Evaluate a single model by comparing to the DNS or compare multiple trained models with each other, based on raw performance (error accumulation) and physical behaviours:
      - Autocorrelation
      - Spectral energy spectrum
      - Proper Orthogonal Decomposition (POD) spectrum
      - Quasi-equilibria quantities of Energy $E$, Enstrophy $U$, particle influx $\Gamma_n$, resistive dissipative flux $\Gamma_c$
      - Vorticity and strain-dominated regions and distribution based on Q-criterion / Okubo-Weiss parameter

For all options for each script, select the `--help` flag when calling the script
