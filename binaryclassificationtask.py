#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 15:08:50 2019

@author: esraaadel
"""

import pandas as pd
train = pd.read_csv('train.csv')
validation = pd.read_csv('validate.csv')


train = train.drop( ['variable8' , 'variable9' , 'variable13','variable15' ] , axis=1)
validation = validation.drop( ['variable8' , 'variable9' , 'variable13','variable15' ] , axis=1)

#split data into dependent variable and target variable
x= train.iloc[:,:-1]  #independent variables for train dataset
y = train.iloc[:,-1]    #dependent variable (label ) for train dataset
x1 = validation.iloc[:,:-1]    #independent variables for validation dataset
y1 = validation.iloc[:,-1]      #dependent variable (label ) for validation dataset


# Encoding categorical data
# Encoding the independent and target variables
# x & y for train data
# x1 & y1 for validation data
x = pd.get_dummies(x)
x1 = pd.get_dummies(x1)
y = pd.get_dummies(y)
y1 = pd.get_dummies(y1) 

#to avoid dummy variable trap in labels 
Label_in_validation = y1.drop(['yes.'],axis = 1)  #means 1 is No & 0 is Yes
                                                  # labels of validation
Y = y.drop(['yes.'],axis = 1)  #means 1 is No & 0 is Yes   
                                     #labels of train 


#to avoid dummy variable trap in independent variables
x = x.drop(['variable1_b','variable6_u','variable6_y','variable7_gg','variable7_p',
                  'variable12_t','variable16_p','variable16_s','variable20_t'], axis=1)
x1 = x1.drop(['variable1_b','variable6_y','variable7_p',
                  'variable12_t','variable16_p','variable16_s','variable20_t'], axis=1) 


#MISSING DATA for train
X= x.iloc[:,:].values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 0:11])
X[:, 0:11] = imputer.transform(X[:, 0:11])

#MISSING DATA for validation

validate_x= x1.iloc[:,:].values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(validate_x[:, 0:11])
validate_x[:, 0:11] = imputer.transform(validate_x[:, 0:11])


# Splitting the dataset into the Training set and Test set
# X is data of train without labels
#Y is train labels 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X , Y, test_size = 1/3.0, random_state = 0)


#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


#creating classifier
from sklearn.linear_model import LogisticRegression
classifier= LogisticRegression(random_state = 0)
classifier.fit( X_train , y_train  )

#predict the test set result 
Y_pred = classifier.predict(X_test)
X_test = validate_x
Y_pred1 = classifier.predict(X_test)

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
accuracy_score(y_test, Y_pred)
print (classification_report(y_test , Y_pred))
print (classification_report(Label_in_validation , Y_pred1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, Y_pred)
print(cm)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Label_in_validation , Y_pred1)
print(cm)


