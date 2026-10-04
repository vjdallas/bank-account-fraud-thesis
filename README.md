# Machine Learning Methods for Bank Account Fraud Detection

Code and results for my MSc thesis at the University of the Aegean (2026). The project compares six machine-learning methods, in ten configurations, for detecting fraudulent bank account opening applications under severe class imbalance (1.11% fraud).

## Key findings

- **Gradient boosting wins.** XGBoost reached a test ROC-AUC of 0.893 and recovered 55% of fraud at a 5% false-positive rate.
- **A deep tabular model came close but did not beat it.** The FT-Transformer scored ROC-AUC 0.887, and a plain MLP matched logistic regression (0.876 vs 0.875).
- **The threshold matters as much as the model.** The same decision tree caught 6% of fraud at a 0.5 threshold and 69% at a threshold of 0.011.
- **Class weighting barely changed anything** for random forests or XGBoost.

## Results (test set)

| Model | Threshold | ROC-AUC | PR-AUC | TPR @ 5% FPR |
|---|---|---|---|---|
| Logistic regression | 0.011 | 0.8753 | 0.1467 | 0.5067 |
| Decision tree (default) | 0.5 | 0.8135 | 0.0848 | 0.4109 |
| Decision tree (threshold-adjusted) | 0.011 | 0.8135 | 0.0848 | 0.4109 |
| Decision tree (balanced priors) | 0.5 | 0.8187 | 0.0496 | 0.2808 |
| Random forest (default) | 0.011 | 0.8786 | 0.1485 | 0.5155 |
| Random forest (balanced) | 0.011 | 0.8786 | 0.1532 | 0.5164 |
| XGBoost (default) | 0.011 | 0.8925 | 0.1791 | 0.5515 |
| XGBoost (balanced) | 0.5 | 0.8927 | 0.1806 | 0.5473 |
| MLP | 0.5 | 0.8756 | 0.1437 | 0.5062 |
| FT-Transformer | 0.5 | 0.8872 | 0.1672 | 0.5356 |

Deep models are means over three seeds (42, 123, 2024). The notebook in `python/` additionally runs a timed 5-fold cross-validation of both deep models; its fold-averaged test scores (MLP 0.8750, FT-Transformer 0.8877 ROC-AUC) differ slightly from the three-seed figures above.

## Data

The Base variant of the [Bank Account Fraud (BAF) Suite](https://arxiv.org/abs/2211.13358) (Jesus et al., NeurIPS 2022): one million synthetic applications over eight months, 31 predictors. The data is **not** included in this repository. Download `Base.csv` from the dataset's [Kaggle page](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022) and place it in `data/`. Please check the dataset's licence before reuse.

After cleaning (dropping three high-missingness variables and one constant variable, then listwise deletion of 6,393 records) the analysis sample has 993,607 records, 10,995 of them fraudulent.

## Method in brief

- **Splits.** Classical models use a stratified 70/30 split (seed 123). Deep models use a stratified 70/15/15 split with early stopping on validation PR-AUC.
- **Tuning.** 5-fold cross-validation repeated 5 times with grid search for the classical models; hyperparameters were chosen by hand for the deep models.
- **Metrics.** ROC-AUC, PR-AUC and true-positive rate at 5% false-positive rate, which are threshold-free and informative at this imbalance. Accuracy is not used for comparison.

## Repository structure

```
R/                   classical models (logistic regression, trees, random forests, XGBoost)
python/              MLP and FT-Transformer (PyTorch)
outputs/             metrics tables and figures
data/                place Base.csv here (not included, see data/README.md)
install_packages.R   installs the R packages used by the scripts
requirements.txt     Python packages for the notebook
```
## Reproducing the results

### Software environment

**R** 4.5.3 (2026-03-11): caret 7.0.1, ranger 0.18.0, xgboost 1.7.11.1, PRROC 1.4, ggplot2 4.0.3, data.table 1.18.2.1, dplyr 1.2.1, tidyr 1.3.2, patchwork 1.3.2, rpart 4.1.27, pROC 1.19.0.1.

**Python** 3.13.15 (Google Colab, Tesla T4 GPU): install with `pip install -r requirements.txt`. R and Python each have their own xgboost installation, so the versions differ.

#### Run order

Run all R scripts from the project root (they use relative paths like `R/00_data.R`), with `Base.csv` in `data/`. Install the R packages first with `Rscript install_packages.R`. All scripts use 4 threads.

| Step | Script | Purpose | Output |
|---|---|---|---|
| 1 | `R/01_ch3_tables.R` | Chapter 3 tables and figures | `outputs/ch3/` |
| 2 | `R/02_tune_dt.R` | Decision tree tuning (cp grid, default and balanced priors) | `outputs/tuning/` |
| 3 | `R/02_tune_rf.R` | Random forest tree-count sweep (the final model uses 1000 trees) | `outputs/tuning/` |
| 4 | `R/02_tune_rf_balanced.R` | Class-weighted random forest grid | `outputs/tuning/` |
| 5 | `R/02_tune_xgb.R` | XGBoost grid, default | `outputs/tuning/` |
| 6 | `R/02_tune_xgb_balanced.R` | XGBoost grid, `scale_pos_weight` | `outputs/tuning/` |
| 7 | `R/04_final_models.R` | Refits the final models, metrics, figures, timings, `sessionInfo.txt` | `outputs/ch4/` |
| 8 | `R/05_cv_timing.R` | Wall-clock cost of the hyperparameter searches | `outputs/cv_timings.csv` |
| 9 | `python/deep_models.ipynb` | MLP and FT-Transformer, timed 5-fold CV, train/validation/test metrics (reads `Base.csv` from Google Drive) | Google Drive |

`R/00_data.R` is loaded by every other script via `source()` and is not run on its own. Logistic regression has no tuning script; it is fitted directly in `04_final_models.R`.

The tuning scripts (steps 2 to 6) are slow (the XGBoost searches take hours). `04_final_models.R` has the selected hyperparameters hard-coded, so run it directly to reproduce the final results.

## Limitations

- A single synthetic benchmark, so results may not transfer to live bank data.
- A static random split with no temporal evaluation.
- Classical and deep models are scored on different held-out sets (30% vs 15%).
- Deep model architectures were not tuned beyond a few manual trials, and fairness attributes were not analysed.

## Author

Vasileios-Ioannis Dallas · [LinkedIn](https://www.linkedin.com/in/vasileios-ioannis-dallas-3b1457209/)
