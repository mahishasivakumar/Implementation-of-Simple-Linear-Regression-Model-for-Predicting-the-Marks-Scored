# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and read the dataframe.
2. Assign hours to X and scores to Y.
3. Implement training set and test set of the dataframe
4. Plot the required graph both for test data and training data.
5. Find the values of MSE , MAE and RMSE.

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Mahisha S 
RegisterNumber: 212222040095

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)


#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()


#Graph plot for test data
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)

```

## Output:
![image](https://github.com/mahishasivakumar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119559812/1c4dd7e1-2231-46cb-9db2-1db9aa643b2c)

![image](https://github.com/mahishasivakumar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119559812/282a1eef-2b8e-4e60-8d54-27ef7c3104d6)

![image](https://github.com/mahishasivakumar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119559812/2ad3ca6c-7316-47f1-a155-26d65bf2c2b1)

![image](https://github.com/mahishasivakumar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119559812/2636c616-58ef-4ae3-831a-d3363e557952)

![image](https://github.com/mahishasivakumar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119559812/e417bf1e-0edd-4664-a365-4cddfdba1502)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
