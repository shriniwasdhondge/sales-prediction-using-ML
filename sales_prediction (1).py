#!/usr/bin/env python
# coding: utf-8

# In[38]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
matplotlib.rcParams["figure.figsize"]=(20,10)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# In[8]:


df1=pd.read_csv(r"C:\Users\shrut\Downloads\car_purchasing.csv", encoding='latin1')
df1.head()


# In[9]:


df1.shape


# In[10]:


df1['country'].unique()


# In[11]:


df1['country'].value_counts()


# In[12]:


df2 = df1.drop(['customer e-mail','customer name'],axis='columns')
df2.shape


# In[13]:


df2.head()


# In[14]:


df2.isnull().sum()


# In[15]:


len(df2.country.unique())


# In[16]:


df2.country = df2.country.apply(lambda x: x.strip())
country_stats = df2['country'].value_counts(ascending=False)
country_stats


# In[17]:


label_encoders = {}
for column in ['country', 'gender']:
    le = LabelEncoder()
    df2[column] = le.fit_transform(df2[column])
    label_encoders[column] = le


# In[18]:


# Fill missing values if any
df2.fillna(df2.mean(), inplace=True)


# In[19]:


df2['Sales'] = np.random.randint(0, 2, df2.shape[0])  # Replace with actual sales data

X = df2.drop(columns=['Sales'])
y = df2['Sales']


# In[39]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[40]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[41]:


# Function to evaluate model performance
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return rmse, mae, r2


# In[33]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Train the model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predict
y_pred_lr = lr_model.predict(X_test)

# Evaluate
print("Linear Regression RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lr)))
print("Linear Regression MAE:", mean_absolute_error(y_test, y_pred_lr))
print("Linear Regression R2:", r2_score(y_test, y_pred_lr))


# In[34]:


from sklearn.tree import DecisionTreeRegressor

# Train the model
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)

# Predict
y_pred_dt = dt_model.predict(X_test)

# Evaluate
print("Decision Tree RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_dt)))
print("Decision Tree MAE:", mean_absolute_error(y_test, y_pred_dt))
print("Decision Tree R2:", r2_score(y_test, y_pred_dt))


# In[23]:


pip install xgboost


# In[35]:


from xgboost import XGBRegressor

# Train the model
xgb_model = XGBRegressor(random_state=42)
xgb_model.fit(X_train, y_train)

# Predict
y_pred_xgb = xgb_model.predict(X_test)

# Evaluate
print("XGBoost RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_xgb)))
print("XGBoost MAE:", mean_absolute_error(y_test, y_pred_xgb))
print("XGBoost R2:", r2_score(y_test, y_pred_xgb))


# In[25]:


pip install tensorflow


# In[36]:


# Function to evaluate model performance
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return rmse, mae, r2

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_rmse, lr_mae, lr_r2 = evaluate_model(lr, X_test, y_test)

# Decision Tree Regressor
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
dt_rmse, dt_mae, dt_r2 = evaluate_model(dt, X_test, y_test)

# XGBoost Regressor
xgb = XGBRegressor(random_state=42)
xgb.fit(X_train, y_train)
xgb_rmse, xgb_mae, xgb_r2 = evaluate_model(xgb, X_test, y_test)

# Print evaluation metrics
print(f"Linear Regression - RMSE: {lr_rmse}, MAE: {lr_mae}, R^2: {lr_r2}")
print(f"Decision Tree - RMSE: {dt_rmse}, MAE: {dt_mae}, R^2: {dt_r2}")
print(f"XGBoost - RMSE: {xgb_rmse}, MAE: {xgb_mae}, R^2: {xgb_r2}")


# In[27]:


pip install pandas numpy scikit-learn xgboost matplotlib seaborn


# In[28]:


from sklearn.model_selection import GridSearchCV


# In[37]:


# Hyperparameter tuning for Decision Tree
dt_param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
dt_grid_search = GridSearchCV(estimator=dt, param_grid=dt_param_grid, cv=3, scoring='r2')
dt_grid_search.fit(X_train, y_train)
best_dt = dt_grid_search.best_estimator_

# Hyperparameter tuning for XGBoost
xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}
xgb_grid_search = GridSearchCV(estimator=xgb, param_grid=xgb_param_grid, cv=3, scoring='r2')
xgb_grid_search.fit(X_train, y_train)
best_xgb = xgb_grid_search.best_estimator_

# Evaluate tuned models
best_dt_rmse, best_dt_mae, best_dt_r2 = evaluate_model(best_dt, X_test, y_test)
best_xgb_rmse, best_xgb_mae, best_xgb_r2 = evaluate_model(best_xgb, X_test, y_test)

print(f"Tuned Decision Tree - RMSE: {best_dt_rmse}, MAE: {best_dt_mae}, R^2: {best_dt_r2}")
print(f"Tuned XGBoost - RMSE: {best_xgb_rmse}, MAE: {best_xgb_mae}, R^2: {best_xgb_r2}")


# In[42]:


# Load new data for prediction (replace 'path_to_new_data.csv' with your actual file)
new_data = pd.read_csv(r"C:\Users\shrut\Downloads\car_purchasing.csv", encoding='latin1')

# Preprocess new data similarly to the training data
new_data = new_data.drop(['customer name', 'customer e-mail'], axis=1)

# Encode categorical variables using the same encoders used during training
for column in ['country', 'gender']:
    new_data[column] = label_encoders[column].transform(new_data[column])

# Standardize features using the same scaler used during training
new_data_scaled = scaler.transform(new_data)

# Use the best performing model for prediction
predictions = best_xgb.predict(new_data_scaled)

# Print predictions
print("Predictions for new data:")
print(predictions)


# In[44]:


new_data['predicted_net_worth'] = predictions


# In[46]:


# Group by gender and sum predicted net worth
#we are predicted the sales by gender
gender_sales = new_data.groupby('gender').sum()['predicted_net_worth']
print("Sales Predictions by Gender:")
print(gender_sales)

# Visualize sales predictions by gender
gender_sales.plot(kind='bar', figsize=(10, 6), title='Sales Predictions by Gender')
plt.xlabel('Gender')
plt.ylabel('Predicted Net Worth')
plt.show()


# In[47]:


# Define age groups
#here we are finding how sales goin happen based on age group
bins = [0, 18, 30, 45, 60, 100]
labels = ['0-18', '19-30', '31-45', '46-60', '61-100']
new_data['age_group'] = pd.cut(new_data['age'], bins=bins, labels=labels)

# Group by age group and sum predicted net worth
age_group_sales = new_data.groupby('age_group').sum()['predicted_net_worth']
print("Sales Predictions by Age Group:")
print(age_group_sales)

# Visualize sales predictions by age group
age_group_sales.plot(kind='bar', figsize=(10, 6), title='Sales Predictions by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Predicted Net Worth')
plt.show()

