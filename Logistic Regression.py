import pandas as pd
customer = pd.read_csv('customer.csv')
print(customer.head())

# Split the data into training and testing 
x = customer[['tenure']]
y = customer[['Churn']]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)
print(x_train.head())
print(x_test.head())
print(y_train.head())
print(y_test.head())

# Logistic Regression !
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
print(lr.fit(x_train,y_train))
# predicted values
print(lr.predict(x_test)[0:5])
# Actual values
print(y_test.head())

# finding the residual !
from sklearn.metrics import mean_squared_error
mean_squared_error(print(y_test),(x_test))
