import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Binarizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import  cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

header = ['CRIM', 'ZN', 'INDIS','CHAS', 'NOX', 'RM', 'AGE', 'DIS',
          'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
data = pd.read_csv("./data/3.housing.csv",
                   delim_whitespace=True, names=header)
array = data.values
X = array[:,0:13]
Y = array[:,13]
scaler = MinMaxScaler(feature_range=(0,1))
rescaled_X = scaler.fit_transform(X)

(X_train, X_text,
 Y_train, Y_test) = train_test_split(X, Y, test_size=0.2)
#print(X_train.shape, X_text.shape, Y_train.shape, Y_test.shape)
model = LinearRegression()
model.fit(X_train, Y_train)
y_pred = model.predict(X_text)

plt.scatter(range(len(X_text[:15])), y_pred[:15], color = 'blue')
plt.scatter(range(len(X_text[:15])), y_pred[:15], color = 'red', marker='*')
plt.xlabel("Index")
plt.ylabel("MEDV($1,000)")
plt.show()
mse = mean_squared_error(Y_test, y_pred)
print(mse)


kfold = KFold(n_splits = 10, shuffle = True, random_state = 40)
acc = cross_val_score(model, rescaled_X, Y, cv=kfold, scoring = 'neg_mean_squared_error')
print(acc)

plt.show()



