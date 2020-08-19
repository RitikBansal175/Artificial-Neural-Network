# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 23:34:58 2020

@author: devil may cry
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing the dataset
dataset =pd.read_csv("")
X = dataset.iloc[:,[2,3]].values
y= dataset.iloc[:,4].values

#Encoding Catagorical Data
#Encoding independent variables
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
labelencoder_X=LabelEncoder()
X[:,0]=labelencoder_X.fit_transform(X[:,0])

#Using OneHotEncoder 
"""
onehotencoder = OneHotEncoder(categorical_features=[0])
X=onehotencoder.fit_transform(X).toarray()
"""

#Using ColumnTransformer for categorical data
from sklearn.compose import ColumnTransformer
transformer=ColumnTransformer([('encoder' , OneHotEncoder(), [1])] ,remainder= 'passthrough')
X=np.array(transformer.fit_transform(X) , dtype=np.float)

#Avoiding the dummy variable trap
X= X[:, 1:]
#Encoding dependent Variables
labelencoder_y= LabelEncoder()
y= labelencoder_y.fit_transform(y)

#splitting the data into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.25,random_state=0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#Create Your Classifier here
#classifier = 

#Pridicting the test set results
y_pred= classifier.predict(X_test)

#Creating confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)




