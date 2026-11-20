# Loan Approval Prediction Model

A machine learning pipeline that predicts whether a loan application will be **Approved** or **Rejected** based on applicant financial and demographic features.

The project compares four classification models and achieves near-perfect performance with ensemble methods.

---

## Project Overview

Banks and financial institutions need reliable, data-driven systems to assess loan applications. This project builds an end-to-end predictive model that:

- Cleans and preprocesses real-world loan data
- Engineers meaningful financial ratios
- Trains and evaluates multiple algorithms
- Identifies the best-performing model for deployment

**Best model performance:** Random Forest — **99.88%** test accuracy

---

## Dataset

**Source:** `loan_approval_dataset.csv`  
**Size:** 4,269 loan applications  
**Target:** `loan_status` (Approved / Rejected)

### Original Features

| Feature                      | Description                              |
|-----------------------------|------------------------------------------|
| `no_of_dependents`          | Number of dependents                     |
| `education`                 | Graduate / Not Graduate                  |
| `self_employed`             | Yes / No                                 |
| `income_annum`              | Annual income                            |
| `loan_amount`               | Requested loan amount                    |
| `loan_term`                 | Loan tenure (years)                      |
| `cibil_score`               | Credit score (CIBIL)                     |
| `residential_assets_value`  | Value of residential assets              |
| `commercial_assets_value`   | Value of commercial assets               |
| `luxury_assets_value`       | Value of luxury assets                   |
| `bank_asset_value`          | Bank account / liquid assets             |

---

## Data Preprocessing & Feature Engineering

### Cleaning
- Removed leading/trailing whitespaces from column names and categorical values
- Converted categorical variables to binary:
  - `education`: Graduate → 1, Not Graduate → 0
  - `self_employed`: Yes → 1, No → 0
  - `loan_status`: Approved → 1, Rejected → 0
- Handled negative values in `residential_assets_value` (set to 0)

### Engineered Features
These ratios better capture repayment capacity than raw values:

- `total_assets` = sum of all asset columns
- `loan_to_income` = loan_amount / income_annum
- `loan_to_assets` = loan_amount / total_assets
- `assets_to_income` = total_assets / income_annum
- `liquid_to_assets` = bank_asset_value / total_assets
- `loan_per_year` = loan_amount / loan_term
- `loan_per_year_to_income` = loan_per_year / income_annum

---

## Models Evaluated

| Model                  | Test Accuracy |
|------------------------|---------------|
| Logistic Regression    | 93.21%        |
| Support Vector Machine | 95.32%        |
| **Random Forest**      | **99.88%**    |
| XGBoost                | 99.77%        |

Random Forest and XGBoost both achieved excellent results, with Random Forest slightly ahead on the held-out test set.

All models were trained after feature scaling and evaluated using:
- Accuracy
- Classification Report (Precision, Recall, F1-score)
- Confusion Matrix

---

## Project Structure
