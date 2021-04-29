import pandas as pd
iris = pd.read_csv('iris.csv.csv')
print(iris.head())
# Data Pre-Processing !
iris = iris[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species']]
iris.columns=['sepallength','sepalwidth','petallength','petalwidth','speciea']
print(iris.head())

# Split the model into training & testing !
x = iris[['sepalwidth']]
y = iris[['sepallength']]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)
print(x_train.head())
print(x_test.head())
print(y_train.head())
print(y_test.head())

# Support Vector Machine !
from sklearn.svm import SVR
svm = SVR()
svm.fit(x_train,y_train)
print(y_test.head())
print(svm.predict(x_test)[0:5])
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test,x_test)

