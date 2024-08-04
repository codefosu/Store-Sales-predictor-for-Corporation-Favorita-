# Store-Sales-ML-Predictor-for-Corporation-Favorita
A machine Learning Regression Analysis project to predict store sales for Corporation Favorita 
Introduction
Sales forecasting is a critical aspect of retail operations, enabling businesses to manage inventory, optimize supply chains, and improve customer satisfaction. At Corporation Favorita, a large Ecuadorian-based grocery retailer, we embarked on a project to build a series of machine learning models aimed at predicting the unit sales for thousands of items across various stores. This article details the aim of the project and the comprehensive process we followed, highlighting key code snippets and methodologies.
Aim of the Project
The primary aim of this project was to develop accurate sales forecasting models to ensure optimal stock levels at Corporation Favorita's stores. By predicting future sales, we aimed to reduce stockouts and overstock situations, ultimately enhancing customer satisfaction and operational efficiency.
Project Process
We followed the CRISP-DM (Cross-Industry Standard Process for Data Mining) framework, which provided a structured approach to our data science project. The key steps included:
Business Understanding
Data Understanding
Data Preparation
Modeling
Evaluation
Deployment

1. Business Understanding
Understanding the business context and objectives was crucial. We collaborated with the marketing and sales teams to gather requirements and identify key factors influencing sales, such as promotions, holidays, and store locations.
2. Data Understanding
We retrieved relevant data from various sources, including historical sales data, store information, transactions, holidays, and oil prices. Here are the queries and data loading processes
# Retrieving data from SQL database
import pandas as pd
import pyodbc
connection = pyodbc.connect('Driver={SQL Server};Server=your_server;Database=your_db;Trusted_Connection=yes;')
query = "SELECT * FROM dbo.oil"
oil = pd.read_sql(query, connection)
query = "SELECT * FROM dbo.holidays_events"
holidays = pd.read_sql(query, connection)
query = "SELECT * FROM dbo.stores"
stores = pd.read_sql(query, connection)
# Loading CSV files
train = pd.read_csv('train.csv')
transactions = pd.read_csv('transactions.csv')
3. Data Preparation
Data preparation involved cleaning and merging datasets, handling missing values, and creating new features. Here's a snippet showcasing some of these steps:
# Merging datasets
merged = train.merge(stores, on='store_nbr', how='left')
merged = merged.merge(transactions, on=['store_nbr', 'date'], how='left')
merged = merged.merge(oil, on='date', how='left')
merged = merged.merge(holidays, on='date', how='left')
# Creating new features
merged['day'] = merged['date'].dt.day
merged['month'] = merged['date'].dt.month
4. Modeling
We trained multiple models to predict sales, including Linear Regression, Random Forest Regressor, Support Vector Regressor (SVR), and Decision Tree Regressor. Here are code snippets for training these models:
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
# Splitting data
X = merged[['day', 'month', 'transactions', 'dcoilwtico']]
y = merged['sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Training models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "SVR": SVR(),
    "Decision Tree": DecisionTreeRegressor(random_state=42)
}
results = {"Model": [], "MSE": [], "RMSE": [], "R²": []}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    results["Model"].append(name)
    results["MSE"].append(mse)
    results["RMSE"].append(rmse)
    results["R²"].append(r2)
# Displaying results
results_df = pd.DataFrame(results)
print(results_df)
5. Evaluation
We evaluated the models based on Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² score. The table below summarizes the performance of each model:
6. Deployment
The final step involved deploying the best-performing model to a production environment, allowing Corporation Favorita to make data-driven decisions in real-time.
Conclusion
This project demonstrated the importance of a structured approach in building effective sales forecasting models. By leveraging machine learning techniques, we significantly improved the accuracy of sales predictions, contributing to better inventory management and customer satisfaction at Corporation Favorita.
Call to Action
If you found this article helpful or have any questions, feel free to leave a comment below. For more insights and detailed tutorials, follow me on Medium and stay tuned for more articles on data science and machine learning projects.