# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```

Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Naveen Kumar.B
RegisterNumber: 212222230091 

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
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
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

### Dataset
![image](https://github.com/mrnaviz/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/123350791/aaaf1474-85e0-4bae-b4fb-1dfc67dc66ec)

### Head Values
![image](https://github.com/mrnaviz/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/123350791/aa98aa16-d01f-4eac-9786-bb1c96de725a)

### Tail Values
![image](https://github.com/mrnaviz/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/123350791/181f1ece-5bd1-4a71-a6a3-38d093816930)

### X and Y Values
![image](https://github.com/mrnaviz/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/123350791/c903e193-131b-4f76-84fa-a199fb7b6392)

### Prediction Values of X and Y
![image](https://github.com/mrnaviz/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/123350791/b01f1d7d-9ae0-47ae-945e-c102f001fe28)

### MSE,MAE and RMSE
![image](https://github.com/mrnaviz/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/123350791/ce7ccf31-5b34-493f-bc9a-2546c44702e8)

### Training Set
![image](https://github.com/mrnaviz/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/123350791/fde9bb61-d59b-4d00-b2be-f57f108e076e)

### Testing Set
![image](https://github.com/mrnaviz/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/123350791/ba79f2ee-b461-43fa-93a6-cdc3fee46c9e)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
