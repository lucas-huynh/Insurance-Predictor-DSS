# Clinical Decision Support System (DSS)

## Group Member

- Joanna Chang (joannac2)
- Lucas Huynh (lqh)
- Anshika Shukla (anshika2)

---

## Overview

This project implements a Clinical Decision Support System (DSS) for predicting healthcare insurance costs based on patient characteristics. It integrates machine learning, exploratory data analysis, and interpretability tools within a PostgreSQL-backed system. A Streamlit user interface allows for real-time patient management and cost prediction.

---

## Repository Contents

| File                          | Description                                                                 |
|-------------------------------|-----------------------------------------------------------------------------|
| `UI.py`                       | Streamlit web app interface for prediction and patient database interaction |
| `best_gradient_boosting_model.pkl` | Tuned Gradient Boosting model used in production                        |
| `clinical_dss_backup.dump`    | PostgreSQL database backup of current patient data                         |
| `csv_to_sql.ipynb`            | Converts `healthinsurancedatabase.csv` into a PostgreSQL table             |
| `eda_ml_final_format_v2.ipynb`| Full notebook with EDA, statistical tests, ML training, SHAP, and sensitivity analysis |
| `healthinsurance.csv`         | Original static dataset                                                    |
| `healthinsurancedatabase.csv` | Live patient dataset linked to Streamlit for add/edit/delete patient functionality |

---

## Setup Instructions

### Prerequisites

Before running the project, make sure you have the following installed:

- Python 3.8+
- PostgreSQL (v16 recommended)
- pgAdmin (optional for GUI access)
- Streamlit
- Anaconda (optional for environment management)

---

### Installation of Required Python Packages

Use pip to install all required packages:

```bash
pip install pandas seaborn matplotlib numpy scipy scikit-learn joblib shap streamlit sqlalchemy
```

**Note**: csv, time, math, os, uuid, and datetime are part of Python's standard library.

## Running the App

To launch the Streamlit interface for insurance prediction:

```bash
streamlit run UI.py
```

> If the above doesn't work, try:
```bash
python -m streamlit run UI.py
```

---

### Login Instructions

Upon launching, you'll be prompted to log in using one of the following credentials:

```python
USER_CREDENTIALS = {
    "lucas":   "his2025_lh",
    "joanna":  "his2025_jc",
    "anshika": "his2025_as",
    "aditya":  "his2025_ad",
    "xinyu":   "his2025_xy",
    "rema":    "his2025_rp"
}
```

---

### App Overview

Once logged in, you'll see three navigable pages in the sidebar:

#### 1. **Insurance Prediction Page**
- Enter patient details on the **left sidebar** (e.g., age, BMI, smoking status, region, gender, etc.).
- The system:
  - Predicts expected charges.
  - Displays SHAP values.
  - Provides an automatic, interpretable summary for healthcare providers.
- The patient is **automatically added** to the backend database upon prediction.

#### 2. **Descriptive Analytics Page**
- Explore visual insights that update dynamically as patients are added/removed.
- Filter by:
  - Smoking status
  - Geographic region
  - Age bins (e.g., <30, 30–50, >50)

#### 3. **Patient Database Page**
- View a full table of patient records.
- Add new patients with required details.
- Delete patients using their unique `Patient ID`.
- Each patient receives an **automatically generated UUID** to maintain confidentiality.

---

### Tips for Smooth Streamlit Execution

1. Ensure the following files are in the **same working directory**:
   - `UI.py`
   - `eda_ml_final_format_v2.ipynb`
   - `healthinsurance.csv`
   - `healthinsurancedatabase.csv`
   - `best_gradient_boosting_model.pkl`

2. Confirm that all required packages are installed (see [Installation of Required Python Packages](#installation-of-required-python-packages)).

3. Rerun the Jupyter notebook to:
   - Generate a **new or updated** model (`.pkl` file).
   - Rebuild any necessary supporting files.

4. You're ready to go! If `streamlit run` doesn't launch properly, fall back to:

```bash
python -m streamlit run UI.py
```

## PostgreSQL Database

### Restore from Backup

To restore the patient database using the provided `.dump` file, run the following command in your terminal:

```bash
pg_restore -U dadb -d clinical_dss -v clinical_dss_backup.dump
```
## Jupyter Notebook (`eda_ml_final_format_v2.ipynb`)

This notebook contains the full pipeline of data analysis, model building, interpretation, and validation.

---

### Exploratory Data Analysis (EDA)
- Distribution plots: **age**, **BMI**, **number of children**, **charges**
- **Correlation heatmaps** for numeric features
- Visual comparison: **smoker vs non-smoker** charges
- Regional analysis of charge differences across U.S. locations

---

### Statistical Testing
- **ANOVA**: testing charge differences across regions
- **Point-biserial correlation**: for binary vs continuous variables
- **Spearman correlation**: for monotonic relationships
- **T-tests**: comparing charges between smokers and non-smokers

---

### Machine Learning
#### Models Compared:
- Linear Regression
- Ridge Regression
- Lasso Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- K-Nearest Neighbors (KNN)

#### Evaluation Approach:
- **5-Fold Cross Validation**
- Metrics:
  - RMSE (Root Mean Squared Error)
  - MAE (Mean Absolute Error)
  - R² Score

---

### Model Interpretation (SHAP)
We used SHAP (SHapley Additive exPlanations) for interpretability:
- **Feature importance ranking**
- **Beeswarm plots**: showing direction and magnitude of feature impacts
- Highlighted high influence of **smoking**, **age**, and **BMI** on predictions

---

### Sensitivity Analysis
We tested model robustness under several data perturbations. Below are the results using R² score:

| Scenario                     | R² Score |
|-----------------------------|----------|
| Raw Charges (no transform)  | 0.850    |
| Add noise to BMI            | 0.822    |
| Remove smoker feature       | 0.223    |
| Only non-smokers            | 0.663    |
| 50% of data                 | 0.862    |

> **Baseline (log-transformed target):** R² = **0.855**

---

### Model Performance Summary (Raw vs Log-Transformed)

We evaluated each model on both raw charges and log-transformed charges:

| Model               | RMSE (Raw) | MAE (Raw) | R² (Raw) | RMSE (Log-Transformed) | MAE (Transformed) | R² (Transformed) |
|--------------------|------------|-----------|----------|-------------------------|--------------------|------------------|
| Gradient Boosting  | **4577.37** | 2542.06   | 0.852    | **4616.01**             | **2157.43**        | **0.855**        |
| Random Forest      | 4888.64    | 2716.66   | 0.833    | 4648.82                 | 2228.64            | 0.853            |
| Decision Tree      | 6617.55    | 3110.27   | 0.698    | 6540.33                 | 3048.75            | 0.708            |
| Ridge Regression   | 6077.27    | 4208.14   | 0.740    | 8389.37                 | 4259.63            | 0.520            |
| Lasso Regression   | 6077.23    | 4203.45   | 0.740    | 8394.74                 | 4649.06            | 0.519            |
| Linear Regression  | 6077.23    | 4203.41   | 0.740    | 8418.23                 | 4269.87            | 0.516            |
| KNN Regressor      | 10376.60   | 6849.19   | 0.260    | 11240.21                | 6597.01            | 0.138            |

#### Best Performers:
- **Gradient Boosting** had the lowest RMSE and highest R².
- **Random Forest** followed closely behind in accuracy.

#### Weakest Model:
- **KNN Regressor** struggled due to sensitivity to feature scaling and data variance.

---

### Hyperparameter Tuning

We performed **randomized grid search** using `RandomizedSearchCV` to optimize the **Gradient Boosting Regressor**:

#### Best Parameters:
```python
{
    'learning_rate': 0.0125,
    'max_depth': 3,
    'min_samples_leaf': 2,
    'min_samples_split': 3,
    'n_estimators': 876,
    'subsample': 0.508
}
```
#### Final Performance (on full cross-validation):
- **RMSE**: 4637.10  
- **MAE**: 2164.37  
- **R² Score**: 0.853

## CSV Files

### `healthinsurance.csv`
- Original dataset sourced in a Kaggle-style format
- Used for **initial exploratory data analysis (EDA)** and **model development**

### `healthinsurancedatabase.csv`
- Continuously updated dataset reflecting real-time changes
- Modified dynamically through the **Streamlit UI**
- Used to simulate a **real-time Clinical Decision Support System (CDSS)** environment



