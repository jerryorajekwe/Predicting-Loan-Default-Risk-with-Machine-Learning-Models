# Predicting Loan Default Risk with Machine Learning Models

## Overview

This project applies machine learning algorithms to predict loan default risk using a publicly available lending dataset. It explores models such as Logistic Regression, Random Forest, and XGBoost, combined with preprocessing steps like missing value imputation, scaling, and feature encoding. The pipeline includes hyperparameter optimization via RandomizedSearchCV, class imbalance handling using SMOTE, and model evaluation with classification metrics and visualizations.

## Files in This Repository

- `code.ipynb`: Complete Jupyter Notebook implementing the end-to-end pipeline.
- `loan_data.csv`: Raw dataset used for analysis.
- `loan_data_cleaned.csv`: Dataset after cleaning and feature engineering.
- `loan_data_final.csv`: Final dataset with SMOTE applied for balanced training.
- `visualizations`: Folder containing output plots (e.g., ROC curves, feature importance).
- `README.md`: Project documentation.

## Workflow

1. **Data Preprocessing**
   - Handle missing values, duplicates, and outliers.
   - Encode categorical variables using `OrdinalEncoder`.
   - Scale numerical features with `StandardScaler`.

2. **Handling Class Imbalance**
   - Apply SMOTE to address imbalance in the target variable.

3. **Modeling**
   - Train Logistic Regression, Random Forest, and XGBoost classifiers.
   - Tune hyperparameters using `RandomizedSearchCV`.

4. **Evaluation**
   - Assess models using accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrix.
   - Visualize results using precision-recall curves and feature importance rankings.

## Key Features

- **Hyperparameter Optimization**: Efficient tuning with `RandomizedSearchCV`.
- **Balanced Dataset**: SMOTE improves model fairness and performance.
- **Feature Insights**: Highlights most influential predictors of default.
- **Visual Reporting**: Clear plots for evaluation and interpretation.

## Dataset

The dataset used in this project is publicly available and can be accessed from [https://www.kaggle.com/datasets/adarshsng/lending-club-loan-data-csv?select=loan.csv]. It contains borrower information, loan attributes, and loan status indicators relevant to default risk prediction.

## Prerequisites

- Python 3.7+
- Jupyter Notebook
- Install dependencies:
  ```bash
  pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost
  ```

## How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/Loan-Default-Prediction.git
   ```
2. Navigate into the project folder:
   ```bash
   cd Loan-Default-Prediction
   ```
3. Open and run `code.ipynb` in Jupyter Notebook.

## Future Enhancements

- Extend dataset with macroeconomic indicators for better generalizability.
- Integrate SHAP for model interpretability and explainability.
- Experiment with deep learning models for improved prediction.

## Contact

For feedback or questions, reach out via GitHub Issues or email: **jerryorajekwe@gmail.com**
