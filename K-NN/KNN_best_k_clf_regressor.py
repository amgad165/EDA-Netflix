import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
import pandas as pd
my_data=pd.read_csv('purchase4.csv')


print("Classification")
X=my_data[['Age', 'Income','Year-of-Education']]
y=my_data['Loyalty']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=2)
for k in range(1,10,1):
    clf = neighbors.KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train, y_train)
    print("K=%f score=%f" %(k,clf.score(X_test, y_test)))     


clf = neighbors.KNeighborsClassifier(n_neighbors=6)
clf.fit(X_train, y_train)
print(np.r_[y_test])      
print(clf.predict(X_test))               

print("Regression")
X=my_data[['Age', 'Income','Year-of-Education']]
y=my_data['Purchase-Amount']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=2)
for k in range(1,10,1):
    clf = neighbors.KNeighborsRegressor(n_neighbors=k)
    clf.fit(X_train, y_train)
    print("K=%f score=%f" %(k,clf.score(X_test, y_test)))     

clf = neighbors.KNeighborsRegressor(n_neighbors=4)
clf.fit(X_train, y_train)
print(np.r_[y_test])      
print(clf.predict(X_test).astype(int))       

