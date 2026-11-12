# Lab 4 — Machine Learning: Data Leakage, Overfitting & Generalization

## Overview

This lab explores the challenges of building reliable machine learning models, with a particular focus on **data leakage, overfitting, evaluation strategies, and model generalization**.

Using a student performance survey dataset, the objective is to predict students' final CGPA based on behavioral, academic, and personal features. Different preprocessing strategies, data-splitting techniques, and regression models are investigated to understand their impact on predictive performance and generalization.

## Dataset

- **Dataset:** Hybrid Student Performance Survey
- **Number of observations:** 1,200 students
- **Target variable:** `final_cgpa`
- **Problem type:** Supervised Learning — Regression

The target variable is derived from `cgpa_category` by converting its categorical ranges into numerical midpoint values.

### Features

The dataset includes several categories of student-related information:

- **Study habits:** Daily study hours, revision frequency, and study consistency.
- **Academic behavior:** Attendance percentage, assignments and tasks completed on time.
- **Personal factors:** Age, sleep hours, stress level, and daily productivity.
- **Additional features:** Career goal clarity and a derived effort index.

## Lab Structure

### Part A — Data Inspection and Potential Leakage

- Inspect the dataset structure, dimensions, and feature types.
- Handle categorical variables using ordinal encoding.
- Construct derived features such as `effort_index`.
- Examine correlations, missing values, duplicates, and distributions.
- Identify potential sources of data leakage and overfitting.

**Key considerations:**
- Potential target leakage through `performance_risk_level`.
- Feature redundancy introduced by derived variables.
- Temporal leakage caused by inappropriate random splitting.
- Information loss from converting CGPA ranges into numerical midpoints.

### Part B — Impact of Data Leakage and Evaluation Strategy

Investigate how different evaluation strategies influence model performance.

Experiments include:

1. Comparing random and chronological 80/20 train-test splits.
2. Examining the effect of duplicate observations on evaluation.
3. Comparing preprocessing strategies, particularly fitting `StandardScaler` on the entire dataset versus training data only.

The experiments demonstrate why data preprocessing must be fitted exclusively on training data and why chronological splitting can provide a more realistic evaluation when temporal ordering matters.

### Part C — Understanding and Diagnosing Overfitting

Explore model complexity and its relationship with generalization.

- Experiment with different Decision Tree maximum depths.
- Compare training and testing RMSE.
- Calculate the generalization gap.
- Investigate polynomial regression with different degrees.
- Apply `TimeSeriesSplit` for chronological cross-validation.

The objective is to understand how increasing model complexity can improve training performance without necessarily improving predictions on unseen data.

### Part D — Improving Model Generalization

Evaluate different techniques for improving predictive performance:

- **Categorical encoding:** Rare-category grouping using One-Hot Encoding and Target Encoding.
- **Feature selection:** Compare models with and without cohort information.
- **Temporal features:** Investigate the contribution of submission day.
- **Regularization:** Compare Ridge and Lasso regression across different alpha values.
- **Early stopping:** Select the optimal number of Gradient Boosting estimators using validation performance.

## Models and Evaluation Metrics

### Models

- Linear Regression
- Ridge Regression
- Lasso Regression
- Decision Tree Regressor
- Gradient Boosting Regressor

### Evaluation Metrics

| Metric | Purpose |
|---|---|
| RMSE | Measures prediction error, penalizing larger errors. |
| MAE | Measures average absolute prediction error. |
| R² | Measures the proportion of target variance explained by the model. |
| Generalization Gap | Measures the difference between training and testing errors. |

## Final Results

The final models were evaluated using a chronological 80/20 train-test split.

| Model | Training RMSE | Testing RMSE | Testing R² |
|---|---:|---:|---:|
| Baseline | 1.2685 | 1.2362 | -0.0029 |
| Linear Regression | 1.1333 | 1.1142 | 0.1852 |
| Ridge (α = 0.01) | 1.1333 | **1.1142** | **0.1852** |
| Decision Tree (depth = 5) | 1.0846 | 1.1437 | 0.1415 |
| Gradient Boosting (48 estimators) | 1.0939 | 1.1194 | 0.1776 |

### Key Findings

- **Best generalization:** Ridge Regression (α = 0.01), achieving the lowest testing RMSE and highest R².
- **Largest generalization gap:** Decision Tree, showing a greater difference between training and testing errors than the regularized linear models.
- **Effective techniques:** Regularization, early stopping, and rare-category grouping helped control model complexity and improve validation performance.
- **Feature limitations:** The best R² of approximately 0.185 indicates that the available survey features explain only a modest proportion of CGPA variation.

## Technologies & Libraries

- Python
- Jupyter Notebook
- Pandas & NumPy
- Matplotlib & Seaborn
- Scikit-learn

## Conclusion

This lab highlights that achieving good training performance is not sufficient to build a reliable predictive model. Proper data splitting, leakage prevention, validation-based model selection, and regularization are essential for evaluating generalization.

Although Ridge Regression achieved the best results, its modest R² reveals the limitations of the available survey features and emphasizes the importance of data quality and feature relevance over simply increasing model complexity.
