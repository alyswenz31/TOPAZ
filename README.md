# TOPAZ: Topologically-based Parameter Inference for Agent-Based Model Optimization

This repository contains code and documentation for the paper:

**"Topologically-based parameter inference for agent-based model selection from spatiotemporal cellular data"**  
**Authors**: Alyssa R. Wenzel, Patrick M. Haughey, Kyle C. Nguyen, John T. Nardini, Jason M. Haugh, Kevin B. Flores  

---

## 🔍 Overview

This repository includes:
- The full code used to run all simulations and analyses in the manuscript.
- Two simplified Jupyter notebook tutorials for easier understanding and use.

The TOPAZ pipeline consists of four major stages:
1. **Simulation / Topological Data Analysis (TDA)**
2. **Approximate Bayesian Computation (ABC) (with the option of adding Approximate Approximate Bayesian Computation (AABC))**
3. **Statistical verification**
4. **Bayesian Information Criterion (BIC)**

Each stage has been modularized and can be run independently.

Most scripts in the `Full_Project_Code` folder are optimized for **High-Performance Computing (HPC)**. Even on HPC, Steps 1–2 of ABC and AABC_after_ABC can each take 1–2 days. All other scripts typically run in under an hour, many in just minutes.

---

## Model format and user prerequisites

To utilize this code, it is expected that the user has a familiarity with Python, specifically with understanding code, running scripts, and modifying code as needed. The Full Project Code is intended to reproduce the results of the paper whereas the tutorial and insert-your-own ABM examples are intended more as an example of how to use the TOPAZ pipeline to do model selection on other ABMs. The tutorial included should be ready-to-use whereas the insert-your-own ABM may require more technical skills to verify your data is in the correct formatting. 

The agent-based models (ABMs) used in this repository are provided as serialized Python objects stored in `.pkl` files. Each `.pkl` file contains a fully specified model configuration, including parameter values and simulation settings, and is intended to be loaded and executed within the provided Python analysis pipeline. More details about the data needed and possible changes needed to the code can be found in the insert-your-own ABM notebook. 

All scripts assume a standard scientific Python environment and are documented to indicate required inputs, outputs, and dependencies.

---

## 📁 Directory: `Full_Project_Code\ABC`

### Main Subfolders

- `Full_CLW_Simulation/`: Generates full simulations of all C, L, and W combinations; creates 2D/3D t-SNE plots.
- `ChangeW_00/`: W = 0 (no alignment); ABC simulations allow W to vary.
- `ChangeW_05/`: W = 0.05 (with alignment); ABC simulations allow W to vary.
- `NoChangeW_00/`: W = 0; ABC simulations fix W = 0 (D’Orsogna model).
- `NoChangeW_05/`: W = 0.05; ABC simulations fix W = 0 (D’Orsogna model).

### Supporting Scripts

- `copy_crocker_plots.sh`: Extracts Crocker plots from each simulation type.
- `copy_folders.sh`: Extracts selected parameter combinations for comparison.
- `get_results.py`: Aggregates simulation results into a CSV file.
- `submit_results.sh`: HPC launcher for `get_results.py`.

---

## 🧬 ABC Simulation Pipeline (Each "Change" Folder)

1. `S01_simulate_grid.py` – Runs ground-truth simulations (via `submitS01.sh`).
2. `S02_crockerplot_save.py` – Calculates Crocker plots (via `submitS02.sh`).
3. `ABC_S01_samples.py` – Runs 10,000 ABC simulations (via `submit1.sh`).
4. `ABC_S02_crockerplot_save.py` – Crocker plots for ABC samples (`submit2.sh`).
5. `ABC_S03_crockerplot_distance.py` – Calculates distances to 10 ground-truth sims (`submit3.sh`).
6. `ABC_S04_p01_tolerance.py` – Posterior inference and plots (`submit4.sh`).
7. `ABC_S05_process_sim.py` – Runs simulation at ABC-estimated medians (`submit5.sh`).
8. `ABC_S06_process_crocker.py` – Crocker plots for estimated medians (`submit6.sh`).
9. `bic.py` – Calculates BIC scores (`submitBIC.sh`).
10. `ic_vec.npy` – Initial conditions file for ABM simulations.

---

## ⚙️ Folder: `Scripts`

- `Alignment.py`: Adds alignment to the D’Orsogna model.
- `arrow_head_marker.py`: Used in arrow plots in `Full_CLW_Simulation`.
- `DorsognaNondim_Align.py`: D’Orsogna model with arbitrary W.
- `DorsognaNondim_Align_w00.py`: W fixed at 0.
- `DorsognaNondim_Align_w05.py`: W fixed at 0.05.
- `filtering_df.py`: Filters simulations before TDA.
- `crocker.py`: Generates Crocker matrices and plots.

**Legacy files (included but not required):**  
`Dorsogna_fluidization.py`, `Dorsogna.py`, `DorsognaNondim.py`, `Helper_Practice.py`, `parallel_ABM_simulate.py`, `parallel_crocker.py`, `parallel_parameter_estimation.py`, `simplex_PE.py`

---

## 📁 Folder: `Full_CLW_Simulation`

- `last_frame_params.csv`: Parameter sets used for arrow plots.
- `S01_simulate_grid_allW.py`: Simulates across full C, L, W grid.
- `S02_crockerplot_save.py`: Crocker plots from full simulations.
- `S03_crockerplot_tsneplot_gray_and_color.py`: 2D/3D t-SNE plots.
- `ProcessResults_allFrames.ipynb`: Arrow plots across time.
- `ProcessResults_lastFrames.ipynb`: Arrow plots at final time step.
- `Scripts/`: Same as above.
- **Not Used**: Alternate PCA/t-SNE scripts (included for completeness).

---

## 📁 Directory: `Full_Project_Code\AABC_after_ABC`

### Main Subfolders

- `Model_AL/`: AABC simulations for Model_AL for both W=0 and W=0.05 (Alignment model).
- `Model_DO/`: AABC simulations for Model_DO for both W=0 and W=0.05 (D’Orsogna model).

### Supporting Scripts

- `get_pars_n_crockers.py`: Concatenates and flattens crocker plots and parameters to be used in   `aabc_run_tree.py`.
- `get_results_aabc.py`: Aggregates simulation results into a CSV file.
- `submit_pars_n_crockers.sh`: HPC launcher for `get_pars_n_crockers.py`.
- `submit_results_aabc.sh`: HPC launcher for `get_results_aabc.py`.
- `view_crocker.py`: Generates Crocker pdf from chosen `crocker_angles.py` files.

---

## 🧬 AABC Simulation Pipeline (in Model_AL and Model_DO)

One Time Through: 
1. `ABC_S01_samples.py` – Runs X Number of ABC simulations on smaller grid (via `submit1.sh`).
2. `ABC_S02_crockerplot_save.py` – Crocker plots for ABC samples (`submit2.sh`).
3. `ABC_S03_crockerplot_distance.py` – Calculates distances to 10 ground-truth sims (`submit3.sh`).
4. `ABC_S04_p01_tolerance.py` – Posterior inference and plots (`submit4.sh`).
5. `ABC_S05_process_sim.py` – Runs simulation at ABC-estimated medians (`submit5.sh`).
6. `ABC_S06_process_crocker.py` – Crocker plots for estimated medians (`submit6.sh`).
7. `bic.py` – Calculates BIC scores (`submitBIC.sh`).
8. Run `get_pars_n_crockers.py` in `AABC_after_ABC` (`submit_pars_n_crockers.sh`).

Repeatedly until Convergence: 
1. `aabc_run_tree.py` - Generates X Number of AABC samples based on above ABC simulations (submitAABC_tree.sh).
2. `ABC_S03_crockerplot_distance_aabc_multiple.py` - Calculates distances using chosen ABC and AABC samples (`submit3_aabc.sh` or `submitAABC_pipeline.sh`).
3. Run Steps 4-7 from above (`submitAABC_pipeline.sh` will run the previous step and this step).
4. `bic_plot_results` - Generates plot of BIC values for each grouping of ABC and AABC samples (run locally). 

Extra:
- `ic_vec.npy` – Initial conditions file for ABM simulations.

---

## 📁 Directory: `Full_Project_Code\Statistical_Verification`

### Files

- `run_stats_tests_all_500.py`: Runs the statistical verification tests using PERMANOVA, Energy Distance, and MMD tests using 500 samples of Model_AL and Model_DO when W=0 and W=0.05.
- `submit_stats_tests.sh`: HPC launcher for `run_stats_tests_all_500.py`.

---

## 📓 Notebook_Tutorials

### Ready_To_Use_Example

- `TOPAZ_CLW.ipynb`: Minimal example (1 simulation + 30 ABC samples + 1 BIC).
- `sample_30/`: Precomputed Crocker matrices for 30 ABC samples.
- `Modules_CLW/`: Core modules adapted for the small-scale example.
- `Scripts/`: Shared utility scripts.
- `Example_created_code/`: Example outputs from the pipeline.
- `ic_vec.npy`: Shared initial conditions file.
- `mega_w_pic_dark`: Figure of the pipeline used in the notebook.

### Insert_Own_ABM_Example

- `TOPAZ_General.ipynb`: Template for inserting your own ABM simulation results.
- `Modules_General/`: General pipeline modules (some placeholders for user ABM).
- `mega_w_pic_dark`: Figure of the pipeline used in the notebook.

- Will be updated to include AABC and Statistical Verification.

---

## 📝 License

This project is released under the [MIT License](LICENSE).

---

## 📚 Citation

If you use this code or framework in your research, please cite:

> Wenzel, A.R., Haughey, P.M., Nguyen, K.C., Nardini, J.T., Haugh, J.M., & Flores, K.B.  
> _Topologically-based parameter inference for agent-based model selection from spatiotemporal cellular data_.  
> (Submitted to *PLOS Computational Biology*, 2025).

---

## 📬 Contact

For questions, bug reports, or contributions, please contact:  
Kevin B. Flores – [kbflores@ncsu.edu](mailto:kbflores@ncsu.edu)

