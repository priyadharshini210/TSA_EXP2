# Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION
## Date:26/08/2025
## Register No:212223240128
### AIM:
To Implement Linear and Polynomial Trend Estiamtion Using Python.

### ALGORITHM:
Import necessary libraries (NumPy, Matplotlib)

Load the dataset

Calculate the linear trend values using least square method

Calculate the polynomial trend values using least square method

End the program
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("dc (2).csv", parse_dates=['date'], index_col='date')
data.head()

resampled_data = data['volume'].resample('Y').sum().to_frame()
resampled_data.head()
resampled_data.index = resampled_data.index.year

resampled_data.reset_index(inplace=True)
resampled_data.rename(columns={'date': 'Year'}, inplace=True)
resampled_data.head()

years = resampled_data['Year'].tolist()
volumes = resampled_data['volume'].tolist()

X = [i - years[len(years) // 2] for i in years]
x2 = [i ** 2 for i in X]
xy = [i * j for i, j in zip(X, volumes)]

n = len(years)
b = (n * sum(xy) - sum(volumes) * sum(X)) / (n * sum(x2) - (sum(X) ** 2))
a = (sum(volumes) - b * sum(X)) / n
linear_trend = [a + b * X[i] for i in range(n)]

x3 = [i ** 3 for i in X]
x4 = [i ** 4 for i in X]
x2y = [i * j for i, j in zip(x2, volumes)]
coeff = [[len(X), sum(X), sum(x2)],
         [sum(X), sum(x2), sum(x3)],
         [sum(x2), sum(x3), sum(x4)]]

Y = [sum(volumes), sum(xy), sum(x2y)]
A = np.array(coeff)
B = np.array(Y)
solution = np.linalg.solve(A, B)
a_poly, b_poly, c_poly = solution
poly_trend = [a_poly + b_poly * X[i] + c_poly * (X[i] ** 2) for i in range(n)]

print(f"Linear Trend: y={a:.2f} + {b:.2f}x")
print(f"\nPolynomial Trend: y={a_poly:.2f} + {b_poly:.2f}x + {c_poly:.2f}x²")

resampled_data['Linear Trend'] = linear_trend
resampled_data['Polynomial Trend'] = poly_trend
resampled_data.set_index('Year', inplace=True)

resampled_data['volume'].plot(kind='line',color='blue',marker='o') #alpha=0.3 makes
resampled_data['Linear Trend'].plot(kind='line',color='black',linestyle='--')

resampled_data['volume'].plot(kind='line',color='blue',marker='o')
resampled_data['Polynomial Trend'].plot(kind='line',color='black',linestyle='--')
```
### OUTPUT
A - LINEAR TREND ESTIMATION

<img width="549" height="448" alt="download" src="https://github.com/user-attachments/assets/d215f095-f996-44e9-936b-12c43d26482f" />

B- POLYNOMIAL TREND ESTIMATION

<img width="549" height="448" alt="download" src="https://github.com/user-attachments/assets/6926cd69-24c7-4ef9-a2ef-6e3a08656e3d" />

### RESULT:
Thus the python program for linear and Polynomial Trend Estiamtion has been executed successfully.
