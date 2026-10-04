# Data

The dataset is not included in this repository.

1. Download **Bank Account Fraud Dataset Suite (NeurIPS 2022)** from Kaggle:
   https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022
2. Place the Base variant here as `data/Base.csv`.

| | |
|---|---|
| File | `Base.csv` (about 213 MB) |
| Rows | 1,000,000 |
| Columns | 32 (31 features and the target `fraud_bool`) |
| Fraud rate | about 1.1% |
| Rows after cleaning | 993,607 |

Cleaning is done in `R/00_data.R` and repeated in `python/deep_models.ipynb`:
four columns are dropped (`device_fraud_count`, `prev_address_months_count`,
`intended_balcon_amount`, `bank_months_count`), the value `-1` is treated as
missing in `current_address_months_count`, `session_length_in_minutes` and
`device_distinct_emails_8w`, and incomplete rows are removed.

The dataset is distributed by its authors under its own licence (see the Kaggle
page). Cite: Jesus et al., *Turning the Tables: Biased, Imbalanced, Dynamic
Tabular Datasets for ML Evaluation*, NeurIPS 2022.
