# RUL-Prediction-Framework-with-Risk-Aware-Prediction-Intervals-via-Conformalized-Quantile-Regression
This repository contains the code for the IJPHM paper submission: Remaining Useful Life Estimation for Aircraft Engines with Risk-Aware Prediction Intervals via Conformalized Quantile Regression, written by Colby Don Robinson.

The paper introduces a remaining useful life prediction framework for aircraft engines that jointly addresses point estimation, uncertainty quantification, and the integration of aerospace risk preferences to advance decision support for safety-critical maintenance. The framework is evaluated on all four subsets of NASA's C-MAPSS benchmark dataset.


<img width="2368" height="724" alt="Framework Flowchart" src="https://github.com/user-attachments/assets/0f9256d2-5c6f-459c-9b1d-6024b22baec3" />


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
 python .\scripts\run_fd001_fd003_experiment.py --subset FD00X
```

   For FD002 & FD004:
```   
 python .\scripts\run_fd002_fd004_experiment.py --subset FD00X
```
These two lines of code will perform the experiments detailed in the paper for each of the specified subsets in the C-MAPSS dataset. The experiment's use the proposed RUL prediction framework that uses gradient boosting regressors for the two modeling components: the point-prediction model and the asymmetric CQR module. The random seed is set to 42.
