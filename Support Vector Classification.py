import pandas as pd
customer = pd.read_csv('customer.csv')
print(customer.head())

x = customer[['tenure']]
y = customer[['Churn']]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)
print(x_train.head())
print(x_test.head())
print(y_train.head())
print(y_test.head())

from sklearn.svm import SVC
svm = SVC()
svm.fit(x_train,y_train)
y_pred = (print(svm.predict(x_test)[0:5]))
y_test = (print(y_test.head()))

