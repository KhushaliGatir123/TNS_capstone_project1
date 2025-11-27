📘 Manufacturing Equipment Output Prediction — Capstone Project (TNSIF)

This project predicts the hourly manufacturing output (Parts Per Hour) for injection-molding machines using machine learning regression. It supports production optimization, output forecasting, and machine performance analysis.

📌 Objective

Predict Parts_Per_Hour based on machine and material parameters

Improve production efficiency

Identify underperforming machines

Support data-driven decisions in manufacturing


📊 Dataset Overview

Records: 1000

Target Column: Parts_Per_Hour

Total Features: 17+


🧩 Features Included

Injection Temperature

Injection Pressure

Cycle Time

Cooling Time

Material Viscosity

Ambient Temperature

Machine Age

Operator Experience

Maintenance Hours

Temperature_Pressure_Ratio

Total_Cycle_Time

Efficiency Score

Machine Utilization


❌ Categorical Columns (Dropped for Modeling)

Dropped non-numeric columns for modeling:

Machine_Type

Material_Grade

Day_of_Week

Shift

Timestamp


🛠 Data Preprocessing

Filled missing numerical values using mean imputation

Removed categorical columns entirely

Removed timestamp column

Kept only numerical columns for modeling

Standardized numerical features using StandardScaler

Separated X and y for regression

Performed train-test split (80-20)


🤖 Model Used

Linear Regression


📈 Evaluation Metrics

RMSE

MSE

MAE

R² Score

Regression Accuracy (100 − MAPE)


📦 Saved Artifacts

linear_regression_model.pkl

scaler.pkl

feature_columns.json

Used for deployment and prediction.


🖥 Deployment

A Streamlit web application is included.

Run the App
pip install -r requirements.txt
streamlit run app.py


📁 Project Structure
Manufacturing_LR/
│── app.py
│── main.ipynb
│── linear_regression_model.pkl
│── scaler.pkl
│── feature_columns.json
│── manufacturing_dataset_1000_samples.csv
│── requirements.txt
│── README.md


🧰 Technologies Used

Python

Pandas, NumPy

Scikit-learn

Streamlit

Matplotlib, Seaborn
