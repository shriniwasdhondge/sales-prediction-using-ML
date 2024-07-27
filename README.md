# sales-prediction-using-ML
Project Overview
This project aims to develop a machine learning model to predict car sales based on customer demographic data.
The dataset includes customer name, customer email, country, gender, age, annual salary, credit card debt, and net worth. 
The goal is to use this information to predict sales trends and provide actionable insights to inform business strategies.
Features
Customer Name: Name of the customer (not used in prediction).
Customer Email: Email of the customer (not used in prediction).
Country: Country of the customer.
Gender: Gender of the customer.
Age: Age of the customer.
Annual Salary: Annual salary of the customer.
Credit Card Debt: Amount of credit card debt the customer has.
Net Worth: Net worth of the customer.

Objectives
Data Preprocessing: Handle missing values, encode categorical features, and scale numerical features.
Model Development: Train and evaluate multiple models including Linear Regression, Decision Trees, and XGBoost.
Hyperparameter Tuning: Optimize model performance using Grid Search for hyperparameter tuning.
Prediction and Aggregation: Use the trained model to predict sales and aggregate predictions based on different criteria.
Visualization and Insights: Visualize the predictions and derive actionable business insights.

Steps to Run the Project:

1.Install Dependencies:
2.Load and Preprocess Data
3.Train Models
4.Hyperparameter Tuning
5.Make Predictions

Results and Insights
Top Performing Models: The XGBoost model outperformed other models with optimized hyperparameters.
Sales Predictions by Country: The model highlighted countries with the highest predicted net worth, which can inform targeted marketing strategies.
Actionable Insights: Insights derived from gender and age group analysis can help in developing targeted products and marketing campaigns.
Conclusion
This project demonstrates how machine learning can be used to predict car sales and derive actionable business insights from customer demographic data.
By understanding the factors that influence sales, businesses can make informed decisions to optimize their strategies and improve overall performance.

#note:if you are unable to read the csv file ,then add this (encoding='latin1') at the end of path that you have copied
