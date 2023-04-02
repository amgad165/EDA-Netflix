import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import pandas as pd
my_data=pd.read_csv('purchase4.csv')


print("Classification")
X=my_data[['Age', 'Income','Year-of-Education']]
y=my_data['Loyalty']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=2)
for my_C in [0.1,1,5,10,20,100,1000]:
    clf = svm.SVC(kernel='rbf', C=my_C)
    clf.fit(X_train, y_train)
    print("C=%f score=%f" %(my_C,clf.score(X_test, y_test)))     

clf = svm.SVC(kernel='rbf', C=10)
clf.fit(X_train, y_train)
print(np.r_[y_test])      
print(clf.predict(X_test))               

print("Regression")
X=my_data[['Age', 'Income','Year-of-Education']]
y=my_data['Purchase-Amount']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=2)
for my_C in [0.1,1,5,10,20,100,1000]:
    clf = svm.SVR(kernel='rbf', C=my_C)
    clf.fit(X_train, y_train)
    print("C=%f score=%f" %(my_C,clf.score(X_test, y_test))) 
    
    
clf = svm.SVR(kernel='rbf', C=20)
clf.fit(X_train, y_train)    
print(np.r_[y_test])      
print (clf.predict(X_test).astype(int))