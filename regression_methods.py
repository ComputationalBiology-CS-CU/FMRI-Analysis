from sklearn.linear_model import SGDRegressor, LinearRegression, Lasso, LassoLars
from sklearn import svm, tree
import numpy as np
import scipy
from sklearn import cross_validation, metrics
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree


def Linear_Regression(kf,data,label,k):
	val=0
	for train, test in kf:
		X_train, X_test, y_train, y_test = data[train,:], data[test,:], label[train], label[test]
		log =  LinearRegression()
		logit = log.fit(X_train,y_train)
		y_pred =  logit.predict(X_test)
		val+= metrics.mean_squared_error(y_test, y_pred)  
		print y_pred, y_test
	return val/3
	# print "Linear_Regression, Mean Squared Error ", "{0:.4f}".format(val/3)

def Lars_Lasso(kf,data,label,k):
	val=0
	for train, test in kf:
		X_train, X_test, y_train, y_test = data[train,:], data[test,:], label[train], label[test]
		log =  LassoLars(alpha=.1)
		logit = log.fit(X_train,y_train)
		y_pred =  logit.predict(X_test)
		val+= metrics.mean_squared_error(y_test, y_pred)  
	return val/3
	# print "Lasso_Regression, Mean Squared Error ", "{0:.4f}".format(val/3)

def Lasso_Regression(kf,data,label,k):
	val=0
	for train, test in kf:
		X_train, X_test, y_train, y_test = data[train,:], data[test,:], label[train], label[test]
		log =  Lasso(alpha=0.1)
		logit = log.fit(X_train,y_train)
		y_pred =  logit.predict(X_test)
		val+= metrics.mean_squared_error(y_test, y_pred)  
	return val/3
	# print "Lasso_Regression, Mean Squared Error ", "{0:.4f}".format(val/3)

def SVM_Regression(kf,data,label,k):
	val=0

	for train, test in kf:
		X_train, X_test, y_train, y_test = data[train,:], data[test,:], label[train], label[test]
		log =  svm.SVR()
		logit = log.fit(X_train,y_train)
		y_pred =  logit.predict(X_test)
		val+= metrics.mean_squared_error(y_test, y_pred)  
	return val/3
	# print "SVM_Regression, Mean Squared Error ", "{0:.4f}".format(val/3)

def SGD_Regression(kf,data,label,k):
	val=0
	for train, test in kf:
		X_train, X_test, y_train, y_test = data[train,:], data[test,:], label[train], label[test]
		log =  SGDRegressor(loss='squared_loss', penalty='l2', alpha=0.0001, l1_ratio=0.15,n_iter=5)
		logit = log.fit(X_train,y_train)
		y_pred =  logit.predict(X_test)
		val += metrics.mean_squared_error(y_test, y_pred) 
	return val/3 
	# print "SGD_Regression, Mean Squared Error ", "{0:.4f}".format(val/3)