# Remaining Useful Life Estimation for Aircraft Engines with Risk-Aware Prediction Intervals via Conformalized Quantile Regression
This repository contains the code for the IJPHM paper submission: Remaining Useful Life Estimation for Aircraft Engines with Risk-Aware Prediction Intervals via Conformalized Quantile Regression, written by Colby Don Robinson.

This repository implements a remaining useful life (RUL) prediction framework that combines:
- **Point Estimation**
- **Uncertainty Quantification** via asymmetric conformalized quantile regression (CQR)
- **Risk-aware interval construction** aligned with aerospace risk preferences

The framework is evaluated on all four subsets of NASA's C-MAPSS benchmark dataset (FD001-FD004).
  
Below is a flowchart that outlines the structure of the proposed framework:



<img alt="Process flow (4)"
     src="https://github.com/user-attachments/assets/fddb24fb-617f-40e4-8fb7-3d4e4a8ab6ee"
     style="max-width: 100%; height: auto;" />
---

## Repository Structure

```text
.
├── data/
│   ├── README.md
│   └── processed/
│       ├── train_FD00X_cleaned.csv
│       ├── test_FD00X_cleaned.csv
│       └── rul_FD00X.csv
├── results/
│   ├── fd001/
│   ├── fd002/
│   ├── fd003/
│   └── fd004/
├── scripts/
│   ├── run_fd001_fd003_experiment.py
│   └── run_fd002_fd004_experiment.py
├── .gitignore
├── environment.yml
├── requirements.txt
└── README.md
```


## Data

See data/README.md for dataset information and the steps to recreate processed files. 

- Processed files required by the scripts are located in:

```
data/processed/
```


## Setup

1. Clone the repository
2. Create a new virtual environment and install the requirements or environment

```bash
conda env create -f environment.yml
conda activate rul_cmapss
```
or
```bash
pip install -r requirements.txt
```

3. Open the virtual environment and run:

For FD001 & FD003:
```   
python .\scripts\run_fd001_fd003_experiment.py --subset FD001

python .\scripts\run_fd001_fd003_experiment.py --subset FD003
```

For FD002 & FD004:
```   
python .\scripts\run_fd002_fd004_experiment.py --subset FD002

python .\scripts\run_fd002_fd004_experiment.py --subset FD004
```
#### Notes:
- The experiments run the pipeline detailed in the paper using gradient boosting regressors for:
     - the point-prediction model
     - the asymmetric CQR module

- This release reproduces numeric results and saved artifacts; plotting scripts will be added later.
- Random seed is set to 42 (see code for details).


## Results & Outputs

Each run writes outputs to:
```
results/<subset>/
```

Files produced include:
- Evaluation metrics: ``` results/<subset>/<subset>_eval_metrics.csv ```
- Predictions table (per engine): ``` results/<subset>/<subset>_rul_predictions_<timestamp>.csv ```
- Saved artifacts: .pkl files for point model, interval model, pca, and k-means (for regime-based normalization in FD002/FD004)

The results from the finalized runs used in the manuscript are currently available in the results folder.

## Citation

```
@unpublished{robinson2025rulcqr,
      title = {Remaining Useful Life Estimation for Aircraft Engines with Risk-Aware Prediction Intervals via Conformalized Quantile Regression}, 
      author = {Robinson, Colby Don},
      journal = {Manuscript submitted to the International Journal of Prognostics and Health Management (IJPHM)}
      year = {2025}
}
```

---
