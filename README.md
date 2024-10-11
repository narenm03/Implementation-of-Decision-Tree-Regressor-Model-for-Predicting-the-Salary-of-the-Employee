# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
STEP 1.START

STEP 2.Calculate the null values present in the dataset and apply label encoder.

STEP 3.Determine test and training data set and apply decison tree regression in dataset.

STEP 4.calculate Mean square error,data prediction and r2.

STEP 5.END
## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: NARENDHARAN.M
RegisterNumber:  212223230134
*/
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()
data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()

x= data[["Position","Level"]]
y=data["Salary"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 2)

from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)

from sklearn import metrics
mse = metrics.mean_squared_error(y_test,y_pred)
mse

r2= metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])

```

## Output:
![Screenshot 2024-09-20 111418](https://github.com/user-attachments/assets/5f0d43a4-4540-4e1b-bf6e-2dd88faa990f)


![Screenshot 2024-09-20 111424](https://github.com/user-attachments/assets/fa4fcbc7-f003-4a44-8425-11fb9ed74c16)


![Screenshot 2024-09-20 111430](https://github.com/user-attachments/assets/71debd02-5ac6-4bbf-9688-ca28ea5b9a6b)


![Screenshot 2024-09-20 111436](https://github.com/user-attachments/assets/a49fbf25-a629-4724-ab47-9191fe043592)


![Screenshot 2024-09-20 111731](https://github.com/user-attachments/assets/7ddac803-9890-4155-94d9-de0681defb98)



![Screenshot 2024-09-20 111440](https://github.com/user-attachments/assets/c8720cd5-c35e-403c-a62e-04e36e95cbea)


![Screenshot 2024-09-20 111528](https://github.com/user-attachments/assets/ac8431f2-b45d-4766-8d75-5c586768359b)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
