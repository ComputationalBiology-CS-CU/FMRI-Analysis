import csv
import numpy as np
from classification_methods import *
from regression_methods import *

from sklearn.metrics import zero_one_loss
from sklearn.cross_validation import KFold
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, chi2,SelectPercentile,f_classif, f_regression

input_data = np.zeros((41,91))
label= np.zeros((41,))
label1= np.zeros((41,))
i=0
col=[]

with open('input.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
    	if i==0:
    		col = row[7:98]
    		i+=1
    		continue
        label[i-1] = row[4]
        # label1[i-1] = row[5] 
        input_data[i-1,:] = row[7:98]
        i=i+1


data = preprocessing.normalize(input_data)

choice = raw_input("Enter the choice \n 1. Logistic Regression \n 2. Random Forest Classification \n 3. Perceptron \n 4. SVM \n 5. Stochastic Gradient Descent \n")
k=15
mean1=[]

kf = KFold(41, n_folds=k, shuffle=True)


# if choice=="1" or choice=="0":
# 	mean1.append(logisticRegression(kf,data,label,k))
if choice=="2" or choice=="0":
	mean1.append(RFC(kf,data,label,k,col))
if choice=="3" or choice=="0":
	mean1.append(perceptron(kf,data,label,k))
if choice=="4" or choice=="0":
	mean1.append(SVM(kf,data,label,k))
if choice=="5" or choice=="0":
	mean1.append(decision_tree(kf,data,label,k))


method=['RFC\t', 'Perceptron', 'SVM\t', 'Decision Tree']

print 
print
print
print "Scores Table"
print "Method\t\t\t Accuracy"
print

for me in mean1:
	print "\t"+str("{0:.4f}".format(me)),

	print

j=0

x = np.arange(1)

y=[mean1[0]]
z=[mean1[1]]
k=[mean1[2]]
l=[mean1[3]]
ax = plt.subplot(111)
w = 0.1
recs1 = ax.bar(x-0.20, y,width=w,color='b',align='center')
recs2 = ax.bar(x-0.10, z ,width=w,color='g',align='center')
recs3 = ax.bar(x, k,width=w,color='r',align='center')
recs4 = ax.bar(x+0.1,l ,width=w,color='y',align='center')


ax.set_xticks(x+w)

ax.set_ylabel('Accuracy')
ax.set_xlabel('Features')
ax.set_title('Classification Accuracy')
ax.legend( (recs1[0], recs2[0], recs3[0],recs4[0]), ('RFC', 'Perceptron', 'SVM', 'Decision Tree') )

plt.show()








    
