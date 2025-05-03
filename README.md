# CS4210Project
Machine Learning Project

Arthur Nguyen
Andy Nguyen
Kikie Liu
Nathan Zamora
---

Shelter Strain Prediction in Los Angeles County

Project Overview

This project aims to predict shelter strain across Los Angeles County by modeling the relationship between shelter capacity, socioeconomic indicators, and environmental factors. Shelter strain is defined as the ratio between the homeless population and available shelter beds.

Using publicly available datasets from 2023–2024, we trained multiple machine learning models to:

* Predict numerical strain scores (regression)
* Classify high-risk shelter regions (binary classification)

---

 Models Trained

| Model                     | Type       | Goal                           | Metric   | Result |
| ------------------------- | ---------- | ------------------------------ | -------- | ------ |
| `linreg_model.py`         | Regression | Predict exact strain score     | R²       | \~0.16 |
| `random_forest_model.py`  | Regression | Improved nonlinear prediction  | R²       | \~0.18 |
| `classify_high_strain.py` | Classifier | Flag high-risk shelter regions | Accuracy | \~69%  |

---

Key Features Used

* `SHELTER_CAPACITY`
* `EVICTION_COUNT`
* `AVG_UNEMPLOYMENT_RATE`
* `TOTAL_RAINFALL_IN`
* `AVG_TEMP_F`

---

Outputs

Each script:

* Trains a model on the final dataset
* Outputs key metrics (R², RMSE, accuracy)
* Displays charts (feature importance, actual vs predicted, confusion matrix)

---

What This Project Shows

* Integration of real-world, multi-source data
* Comparison of regression vs. classification approaches
* Honest evaluation of performance limitations
* Practical insights for shelter planning and policy
