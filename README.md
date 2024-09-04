# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
Step-1: Start
Step-2: Intialize weights randomly. 
Step-3: Compute predicted. 
Step-4: Compute gradient of loss function.
Step-5: Update weights using gradient descent.
Step-6: End
```
## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: T MOUNISH
RegisterNumber: 212223240098
*/

import numpy as np

import pandas as pd 
from sklearn.preprocessing import StandardScaler 
def linear_regression (X1,y, learning_rate=0.1,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]

    theta=np.zeros(X.shape[1]).reshape(-1,1)

    for _ in range(num_iters):

        #Calculate predictions 
        predictions=(X).dot(theta).reshape(-1,1)
        #calculate errors
        errors= (predictions-y).reshape(-1,1) #Update theta using gradient descent
        theta-=learning_rate*(1/len (X1))*X.T.dot(errors)

    return theta

data=pd.read_csv("C:/Users/admin/Desktop/50_Startups.csv",header=None)
data.head

X=(data.iloc[1:,:-2].values)
X1=X.astype(float)

scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)
print(X1_Scaled)

#Learn model Parameters
theta=linear_regression(X1_Scaled, Y1_Scaled)
#Predict target value for a new data point
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1, new_Scaled), theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: (pre)")
```

## Output:
![image](https://github.com/user-attachments/assets/79fd109e-b98a-40c5-9952-6abb8e6d8ca2)

![image](https://github.com/user-attachments/assets/f284c504-1d58-41e9-a51a-e683a925deae)

![image](https://github.com/user-attachments/assets/42c318bb-fc6d-4098-a2d9-561a21e2f011)

![image](https://github.com/user-attachments/assets/c6e5e7ef-2111-4f52-af63-6470b5c533f8)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
