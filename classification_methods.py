from sklearn.linear_model import Perceptron
from sklearn import svm, tree
import numpy as np
import scipy
from sklearn import cross_validation
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree


def RFC(kf,data,label,k,col):
	for train, test in kf:
		X_train, X_test, y_train, y_test = data[train,:], data[test,:], label[train], label[test]
		log = RandomForestClassifier(oob_score=False)
		logit = log.fit(X_train,y_train)
		y_pred =  logit.predict(X_test)
	scores = cross_validation.cross_val_score(log, data, label, cv=k)
	return scores.mean()

def perceptron(kf,data,label,k):
	for train, test in kf:
		X_train, X_test, y_train, y_test = data[train,:], data[test,:], label[train], label[test]
		log = Perceptron(penalty="l2", alpha=0.003)
		logit = log.fit(X_train,y_train)
		y_pred =  logit.predict(X_test)
	scores = cross_validation.cross_val_score(log, data, label, cv=k)
	return scores.mean()

def decision_tree(kf,data,label,k):
	for train, test in kf:
		X_train, X_test, y_train, y_test = data[train,:], data[test,:], label[train], label[test]
		log = tree.DecisionTreeClassifier()
		logit = log.fit(X_train,y_train)
		y_pred =  logit.predict(X_test)
	scores = cross_validation.cross_val_score(log, data, label, cv=k)
	return scores.mean()
	
def SVM(kf,data,label,k):
	C = 1.2
	for train, test in kf:
		X_train, X_test, y_train, y_test = data[train,:], data[test,:], label[train], label[test]
		log = svm.SVC(kernel='rbf', gamma=0.9, C=C)
		logit = log.fit(X_train,y_train)
		y_pred =  logit.predict(X_test)
	scores = cross_validation.cross_val_score(log, data, label, cv=k)
	return scores.mean()


