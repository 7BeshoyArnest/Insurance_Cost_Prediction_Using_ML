📌 Overview
Accurate cost estimation in health insurance is vital for risk assessment and policy design. This project analyzes a real-world dataset to 
build regression models that predictthe medical charges incurred by insurance applicants, using variables such as age, BMI, smoking status, and number of dependents.

📊 Dataset
Source: https://www.kaggle.com/datasets/mirichoi0218/insurance
Features:

age – Age of the primary beneficiary

sex – Gender of the person

bmi – Body mass index

children – Number of dependents

smoker – Smoking status

region – Residential area

charges – Medical costs billed by health insurance (target)

🎯 Objectives
Explore and visualize relationships between features and insurance cost

Handle feature encoding and preprocessing

Train and compare regression models

Evaluate model performance and optimize results

⚙️ Models Used
Linear Regression

Lasso and Ridge Regression

Decision Tree Regressor

Random Forest Regressor

XGBoost Regressor

📈 Evaluation Metrics
Mean Absolute Error (MAE)

Mean Squared Error (MSE)

Root Mean Squared Error (RMSE)

R² Score

📌 Key Insights
Smoking status and BMI are strong predictors of insurance costs.

Regularization techniques improved model performance and avoided overfitting.

Random Forest and XGBoost provided the most accurate predictions.

