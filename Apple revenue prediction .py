#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

# Create pandas DataFrame with quarterly sales data
data = pd.DataFrame({'Quarter': ['2018 Q2', '2018 Q3', '2018 Q4', '2019 Q1', '2019 Q2', '2019 Q3', '2019 Q4', '2020 Q1', '2020 Q2', '2020 Q3', '2020 Q4', '2021 Q1', '2021 Q2', '2021 Q3', '2021 Q4'],
                     'Revenue (in billions USD)': [53.265, 62.9, 84.3, 58.02, 53.81, 64.04, 91.82, 58.3, 59.69, 64.7, 111.44, 89.58, 81.43, 83.44, 119.04]})
data.set_index('Quarter', inplace=True)

# Plot the quarterly sales data
plt.figure(figsize=(12,6))
plt.title('Quarterly Sales Data for Apple Inc.')
plt.xlabel('Quarter')
plt.ylabel('Revenue (in billions USD)')
plt.plot(data)
plt.show()

# Simple Linear Regression model
X = np.arange(len(data)).reshape(-1, 1)
y = data.values.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
simple_linear_reg = LinearRegression()
simple_linear_reg.fit(X_train, y_train)
X_future = np.array([15, 16, 17, 18]).reshape(-1, 1)
simple_linear_reg_pred = simple_linear_reg.predict(X_future)
plt.figure(figsize=(12,6))
plt.title('Quarterly Sales Predictions using Simple Linear Regression model for Apple Inc.')
plt.xlabel('Quarter')
plt.ylabel('Revenue (in billions USD)')
plt.plot(X, y, label='Actual')
plt.plot(X_future, simple_linear_reg_pred, label='Predicted')
plt.legend()
plt.show()


# In[22]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

# Create pandas DataFrame with quarterly sales data
data = pd.DataFrame({'Quarter': ['2018 Q2', '2018 Q3', '2018 Q4', '2019 Q1', '2019 Q2', '2019 Q3', '2019 Q4', '2020 Q1', '2020 Q2', '2020 Q3', '2020 Q4', '2021 Q1', '2021 Q2', '2021 Q3', '2021 Q4'],
                     'Revenue (in billions USD)': [53.265, 62.9, 84.3, 58.02, 53.81, 64.04, 91.82, 58.3, 59.69, 64.7, 111.44, 89.58, 81.43, 83.44, 119.04]})
data.set_index('Quarter', inplace=True)

# Remove NaN values
data.dropna(inplace=True)

# Print the quarterly sales data
print(data)

# Polynomial Regression model
X = np.arange(len(data)).reshape(-1,1)
y = data['Revenue (in billions USD)'].values.reshape(-1,1)
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=0)
polynomial_reg = LinearRegression()
polynomial_reg.fit(X_train, y_train)
polynomial_reg_pred = polynomial_reg.predict(X_test)

# Print the model coefficients and intercept
print('Coefficients: ', polynomial_reg.coef_)
print('Intercept: ', polynomial_reg.intercept_)

# Plot the predicted values against the actual values
plt.figure(figsize=(12,6))
plt.title('Quarterly Sales Predictions using Polynomial Regression model for Apple Inc.')
plt.xlabel('Quarter')
plt.ylabel('Revenue (in billions USD)')
plt.plot(y_test, label='Actual')
plt.plot(polynomial_reg_pred, label='Predicted')
plt.legend()
plt.show()


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Create pandas DataFrame with quarterly sales data
data = pd.DataFrame({'Quarter': ['2018 Q2', '2018 Q3', '2018 Q4', '2019 Q1', '2019 Q2', '2019 Q3', '2019 Q4', '2020 Q1', '2020 Q2', '2020 Q3', '2020 Q4', '2021 Q1', '2021 Q2', '2021 Q3', '2021 Q4'],
                     'Revenue (in billions USD)': [53.265, 62.9, 84.3, 58.02, 53.81, 64.04, 91.82, 58.3, 59.69, 64.7, 111.44, 89.58, 81.43, 83.44, 119.04]})
data.set_index('Quarter', inplace=True)

# Multiple Linear Regression model
X = pd.DataFrame({'Quarter': np.arange(len(data)), 'Previous Quarter Revenue': data.shift(1)['Revenue (in billions USD)']})
y = data['Revenue (in billions USD)'].values.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
multiple_linear_reg = LinearRegression()
multiple_linear_reg.fit(X_train, y_train)
multiple_linear_reg_pred = multiple_linear_reg.predict(X_test)

# Print the model coefficients and intercept
print('Coefficients: ', multiple_linear_reg.coef_)
print('Intercept: ', multiple_linear_reg.intercept_)

# Plot the predicted values against the actual values
plt.figure(figsize=(12,6))
plt.title('Quarterly Sales Predictions using Multiple Linear Regression model for Apple Inc.')
plt.xlabel('Quarter')
plt.ylabel('Revenue (in billions USD)')
plt.plot(y_test, label='Actual')
plt.plot(multiple_linear_reg_pred, label='Predicted')
plt.legend()
plt.show()

