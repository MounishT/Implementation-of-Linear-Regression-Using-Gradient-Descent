# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Initialize weights randomly.

2.Compute predicted values.

3.Compute gradient of loss function.

4.Update weights using gradient descent.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: T MOUNISH
RegisterNumber:  212223240098
*/
```
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1, y, learning_rate=0.01, num_iters=1000):
    # Add a column of ones to X for the intercept term 
  X = np.c_[np.ones(len(X1)), X1]
  # Initialize theta with zeros
  theta = np.zeros(X.shape[1]).reshape(-1,1)
  for _ in range(num_iters):
    predictions = (X).dot(theta).reshape(-1, 1)
    errors = (predictions - y).reshape(-1,1)
    theta -= learning_rate* (1 / len(X1)) * X.T.dot(errors)
  return theta

data = pd.read_csv('50_Startups.csv', header=None)
print(data.head())
# Assuming the last column is your target variable 'y' and the preceding column 
X = (data.iloc[1:, :-2].values)
print(X)
X1=X.astype(float)
scaler = StandardScaler()
y = (data.iloc[1:,-1].values).reshape(-1,1)
print(y)
X1_Scaled = scaler.fit_transform(X1)
Y1_Scaled = scaler.fit_transform(y)
print(X1_Scaled)
print(Y1_Scaled)
theta = linear_regression(X1_Scaled, Y1_Scaled)

# Predict target value for a new data point
new_data = np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled = scaler.fit_transform(new_data)
prediction = np.dot(np.append(1, new_Scaled), theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")
```

## Output:
![image](https://github.com/MounishT/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/138955798/f36d843c-dec8-4e20-beff-bae78abd3026)
![image](https://github.com/MounishT/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/138955798/c947ebac-fe7d-4d8f-b960-26b96e5722a7)
![image](https://github.com/MounishT/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/138955798/540ebf93-da30-453a-855c-9dddf6296792)
![image](https://github.com/MounishT/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/138955798/7922637c-10c6-4cba-8e37-fc3769bde43e)
![image](https://github.com/MounishT/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/138955798/8ec4facd-070d-4dcb-9bd2-776d878ddf00)
![image](https://github.com/MounishT/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/138955798/5aa48b00-0395-484a-a721-aba37b9ed344)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming
