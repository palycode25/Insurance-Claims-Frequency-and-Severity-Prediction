# Contribution of Automotive Telematics in Actuarial Modeling

Insurance modeling project — May 2026

## Summary

This project explores the contribution of automotive telematics data in two complementary actuarial problems:

- risk segmentation (classification of high-risk drivers),
- insurance pricing (pure premium estimation via frequency × severity modeling).

The dataset used is a synthetic dataset from So, Boucher & Valdez (2021), simulating an insurance portfolio based on 100,000 usage-based insurance policies.

The objective is to compare:
- traditional vs telematics-based variables
- linear vs non-linear models
- machine learning vs classical actuarial models

---

## Main Results

- Adding telematics variables: +0.11 AUC (0.68 → 0.79 with logistic regression)
- Switching to non-linear models (XGBoost): +0.08 additional AUC
- Total gain between baseline and best model: +0.19 AUC (~+29%)

### Risk Segmentation
- Best model: XGBoost + cost-sensitive learning
- Test AUC ≈ 0.89
- Strong ability to rank high-risk drivers

### Actuarial Pricing
- Frequency model: Poisson
- Severity model: Gamma
- Pure premium = frequency × severity
- Aggregate pricing error ≈ 1.9%
- Strong risk differentiation across deciles (~15× between extremes)

---

## Methodology

### Segmentation
- Logistic regression (baseline)
- XGBoost (non-linear model)
- Imbalance handling:
  - SMOTE
  - cost-sensitive learning
- Calibration: Platt scaling

### Pricing
- Frequency model: Poisson regression
- Severity model: Gamma regression
- Pure premium = frequency × severity

---

## Data

- Source: So, Boucher & Valdez (2021)
- 100,000 synthetic insurance policies
- 52 features:
  - 11 traditional variables
  - 39 telematics variables
  - 2 targets:
    - NB_Claim
    - AMT_Claim

Class imbalance:
- ~95% no claim
- ~5% at least one claim

---

## Tech Stack

- Python 3.12
- pandas, numpy
- scikit-learn
- xgboost
- statsmodels
- scipy
- matplotlib / plotly

---

## Project Structure

data/           raw data  
notebooks/      analysis and models  
app/            Streamlit application  
report/         PDF report  
presentation/   slides  

---

## Key Insights

- Telematics significantly improves risk prediction
- Non-linear models better capture driving behavior patterns
- SMOTE is less effective than cost-sensitive learning for tree-based models
- Machine learning and actuarial modeling are complementary:
  - ML → scoring and segmentation
  - Actuarial → robust pricing

---

## References

- So, Boucher & Valdez (2020, 2021)
- Denuit et al. (2007)
- Akerlof (1970)
- Chawla et al. (2002)
- Niculescu-Mizil & Caruana (2005)
- Stocksieker (2024)

---

## Author

Fatoumata Rami Eunice
