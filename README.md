# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import required Python libraries and create the datasets with study hours and marks.
2.Divide the datasets into training and testing sets. 
3. Create a simple Linear Regression model and train it using the training data.
4. Use the trained model to predict marks on the testing data and display the predicted output.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: H MOHAMMED IRFAN
RegisterNumber: 212225230179 
*/
```
```
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("50_Startups.csv")

x = data['R&D Spend'].values
y = data['Profit'].values

import numpy as np
import matplotlib.pyplot as plt

w = 0.0
b = 0.0
alpha = 0.0000000001
epochs = 100
n = len(x)

losses = []


for _ in range(epochs):
    y_hat = w * x + b
    loss = np.mean((y_hat - y) ** 2)
    losses.append(loss)

    dw = (2/n) * np.sum((y_hat - y) * x)
    db = (2/n) * np.sum(y_hat - y)

    w -= alpha * dw
    b -= alpha * db

plt.figure(figsize=(15,5))
plt.subplot(1, 2, 1)
plt.plot(losses, color="blue")
plt.xlabel("No of Iterations")
plt.ylabel("Loss")
plt.title("LOSS VS ITERATIONS")

plt.figure(figsize=(11, 5))
plt.subplot(1, 2, 2)
plt.scatter(x, y, color="red", label="Data")
plt.plot(x, w * x + b, color="green", label="Regression Line")
plt.xlabel("R&D Spend")
plt.ylabel("Profit")
plt.title("PROFIT VS R&D SPEND")
plt.legend()

plt.tight_layout()
plt.show()

print("Final weight (w):", w)
print("Final bias (b):", b)
```

## Output:
![alt text](<Screenshot 2026-01-30 130220.png>)
![alt text](<Screenshot 2026-01-30 130229.png>)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
