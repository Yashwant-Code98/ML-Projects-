import pandas as pd
import seaborn as sns

iris = pd.read_csv('Iris.csv')
print(iris.head())

# Data Pre-Processing
iris = iris[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species']]
iris.columns=['sepallength','sepalwidth','petallength','petalwidth','species']

print(iris.head())

# Make a Scatter plot
sns.scatterplot(x = 'sepallength', y = 'petallength', data = iris, hue='species')
plt.show()

# Split The Data Into Training & Testing !

x = iris[['sepalwidth']]
y = iris[['sepallength']]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)

print(x_train.head())
print(x_test.head())
print(y_train.head())
print(y_test.head())

# Linear Regression !

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
print(lr.fit(x_train,y_train))
y_pred = lr.predict(x_test)

print(y_test.head())

from sklearn.metrics import mean_squared_error

mean_squared_error(y_pred,y_test)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             mean_squared_error(y_test,y_pred)


