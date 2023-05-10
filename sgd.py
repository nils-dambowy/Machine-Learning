from sklearn.datasets import fetch_california_housing
from sklearn import preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

X = fetch_california_housing().data
Y = fetch_california_housing().target

# split the data set into train and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = pd.DataFrame(data = X_train, columns=fetch_california_housing().feature_names)
X_train['Price'] = list(y_train)  
X_test = pd.DataFrame(data = X_test, columns=fetch_california_housing().feature_names)
X_test['Price'] = list(y_test)

print(X_train.head())

def sgd_regressor(X, y, learning_rate=0.1, n_epochs=1000, k=20):

    w = np.random.randn(1,8)  # Randomly initializing weights
    b = np.random.randn(1,1)  # Random intercept value 
    epoch=1
    
    while epoch <= n_epochs:
        
        # create batch
        temp = X.sample(k)

        # first 13 rows = features
        X_tr = temp.iloc[:,0:8].values
        
        # last row = price
        y_tr = temp.iloc[:,-1].values

        Lw = w
        Lb = b
        
        loss = 0
        y_pred = []
        
        for i in range(k):
            #y_pred = w*X + b
            Lw = (-2/k * X_tr[i]) * (y_tr[i] - np.dot(X_tr[i],w.T) - b)
            Lb = (-2/k) * (y_tr[i] - np.dot(X_tr[i],w.T) - b)

            w = w - learning_rate * Lw
            b = b - learning_rate * Lb
            
            y_predicted = np.dot(X_tr[i],w.T)
            y_pred.append(y_predicted)
        
        loss = mean_squared_error(y_pred, y_tr)
            
        print("Epoch: %d, Loss: %.3f" %(epoch, loss))
        epoch+=1
        learning_rate = learning_rate/1.02
        
    return w,b

def predict(x,w,b):
    y_pred=[]
    for i in range(len(x)):
        temp_ = x
        X_test = temp_.iloc[:,0:8].values
        y = np.ndarray.item((np.dot(w,X_test[i])+b))
        y_pred.append(y)
    return np.array(y_pred)


w,b = sgd_regressor(X_train,y_train)
y_pred_customsgd = predict(X_test,w,b)

plt.figure(figsize=(25,6))
plt.plot(y_test, label='Actual')
plt.plot(y_pred_customsgd, label='Predicted')
plt.legend(prop={'size': 16})
plt.show()
print('Mean Squared Error :',mean_squared_error(y_test, y_pred_customsgd))