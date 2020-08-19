# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 23:53:19 2020

@author: devil may cry
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 23:34:58 2020

@author: devil may cry
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Part 1 Data Preprocessing
# importing the dataset
dataset =pd.read_csv("Churn_Modelling.csv")
X = dataset.iloc[:, 3:-1].values
y= dataset.iloc[:,-1].values

#Encoding independent variables
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
country=LabelEncoder()
gender=LabelEncoder()
X[:,1]=country.fit_transform(X[:,1])
X[:,2]=gender.fit_transform(X[:,2])

from sklearn.compose import ColumnTransformer
transformer=ColumnTransformer([('encoder' , OneHotEncoder(), [1])] ,remainder= 'passthrough')
X=np.array(transformer.fit_transform(X) , dtype=np.float)
X=X[:,1:]
#splitting the data into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#Part 2 Building ANN
import keras
from keras.layers import Dense
from keras.models import Sequential

#Initializing a ANN
classifier = Sequential()

#Adding the input and hidden layers
classifier.add(Dense(output_dim=9, init= 'uniform' , activation = 'relu' , input_dim = 11))

classifier.add(Dense(output_dim=6, init= 'uniform' , activation = 'relu'))

classifier.add(Dense(output_dim=1, init= 'uniform' , activation = 'sigmoid'))

#Compiling ANN
classifier.compile(optimizer = 'adam' , loss = 'binary_crossentropy', metrics= ['accuracy'])

#Fitting the training set in ANN
classifier.fit(X_train, y_train, batch_size=5, nb_epoch=100)

#Pridicting the test set results
y_pred= classifier.predict(X_test)
y_pred =(y_pred > 0.5)

#Part 3 Make the predictions and evaluate the network
#Creating confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)




