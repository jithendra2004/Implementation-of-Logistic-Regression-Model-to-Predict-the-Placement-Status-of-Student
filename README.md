# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by:V A Jithendra 
RegisterNumber:212221230043  
*/
import pandas as pd
data = pd.read_csv("Placement_Data.csv")
print(data.head())
data1 = data.copy()
data1= data1.drop(["sl_no","salary"],axis=1)
print(data1.head())
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
lc = LabelEncoder()
data1["gender"] = lc.fit_transform(data1["gender"])
data1["ssc_b"] = lc.fit_transform(data1["ssc_b"])
data1["hsc_b"] = lc.fit_transform(data1["hsc_b"])
data1["hsc_s"] = lc.fit_transform(data1["hsc_s"])
data1["degree_t"]=lc.fit_transform(data["degree_t"])
data1["workex"] = lc.fit_transform(data1["workex"])
data1["specialisation"] = lc.fit_transform(data1["specialisation"])
data1["status"]=lc.fit_transform(data1["status"])
print(data1)
x = data1.iloc[:,:-1]
print(x)
y = data1["status"]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver="liblinear")
print(lr.fit(x_train,y_train))
y_pred = lr.predict(x_test)
print(y_pred)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
print(accuracy)
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
print(confusion)
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
print(lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]]))
```

## Output:
![1](https://user-images.githubusercontent.com/94226297/162771233-1b4259fc-56ec-411b-9c56-da590be9cbe8.png)

![2](https://user-images.githubusercontent.com/94226297/162771160-4803ea5d-6c1a-4b46-81e8-029dc80322b4.png)

![3](https://user-images.githubusercontent.com/94226297/162771043-db558f91-b479-4f02-99a4-8550070e5e60.png)

![4](https://user-images.githubusercontent.com/94226297/162771089-818e2518-91db-40f5-b1fb-79f9951a0c5d.png)

![5](https://user-images.githubusercontent.com/94226297/162771318-d1da76c4-cda2-4979-9b36-513eb2bd24a0.png)

![6](https://user-images.githubusercontent.com/94226297/162771366-b1e1fb95-37ea-4bd5-a859-300e067ade8a.png)

![7](https://user-images.githubusercontent.com/94226297/162771402-762c3a63-4df3-4f56-a9cf-e269aac2a930.png)

![8](https://user-images.githubusercontent.com/94226297/162771470-f655b215-d12f-4933-98ea-e0a5bdedfb38.png)





## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
