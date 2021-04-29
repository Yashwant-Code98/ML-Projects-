import pandas as pd

boston = pd.read_csv('Boston.csv')

print(boston.head())

x = boston[['crim']]
y = boston[['medv']]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)
print(x_train.head())
print(x_test.head())
print(y_train.head())
print(y_test.head())

from sklearn.tree import DecisionTreeRegressor

dtr = DecisionTreeRegressor()
dtr.fit(x_train,y_train)
print(dtr.predict(x_test)[0:5])
print(y_test.head())

