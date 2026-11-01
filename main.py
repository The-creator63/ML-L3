#Logistic regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("/Users/muhammadtaqiriaz/Desktop/Machine Learning/L3/titanic2.csv")
print(data.head())
print(data.info())

data["Age"] = data["Age"].fillna(data["Age"].median(skipna = True),inplace = True)
data["Embarked"] = data["Embarked"].fillna(data["Embarked"].value_counts().idxmax(),inplace = True)
print(data.info())

data = data.drop("Cabin",axis = 1)
data = data.drop("PassengerId",axis = 1)
data = data.drop("Name",axis = 1)
data = data.drop("Ticket",axis = 1)
data = data.drop("SibSp",axis = 1)
data = data.drop("Parch",axis = 1)
print(data.head())

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
data["Sex"] = encoder.fit_transform(data["Sex"])
data["Embarked"] = encoder.fit_transform(data["Embarked"])
print(data.head())

x = data[["Pclass","Sex","Age","Fare","Embarked"]]
y = data["Survived"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 2)

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(x_train,y_train)
y_predict = log_reg.predict(x_test)

from sklearn.metrics import confusion_matrix,accuracy_score
accuracy = accuracy_score(y_test,y_predict)
accuracy = round(accuracy*100,2)
print("Accuracy = ",accuracy,"%")

import seaborn as sns
matrix = confusion_matrix(y_test,y_predict)
sns.heatmap(matrix,annot = True,fmt = "d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()