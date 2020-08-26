import pandas as pd
import numpy as np
from pandas import Series, DataFrame

df = pd.read_csv('Stage_C_quiz_dataset.csv')
print(df)
Data = df.drop(columns= 'stab')

X = Data.iloc[:, :-1].values
Y = Data.iloc[:, 12].values

# Encoding dependent variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Y = le.fit_transform(Y)


# Split dataset into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.20,random_state=1)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
check=RandomForestClassifier(n_estimators=100)
#Train the model
check.fit(X_train,Y_train)
y_pred=check.predict(X_test)

from sklearn.metrics import  accuracy_score
#ACCURACY
#accuracy = accuracy_score( Y_test, y_pred)
#print('Accuracy: {}'.format(round(accuracy*100)))


import xgboost as xgb
from sklearn.metrics import mean_squared_error
boost = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 100)
boost.fit(X_train,Y_train)
preds = boost.predict(X_test)
accuracy = accuracy_score( Y_test, y_pred)
print('Accuracy: {}'.format(round(accuracy*100)))



  