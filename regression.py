import csv
import numpy as np
from classification_methods import *
from regression_methods import *

from sklearn.metrics import zero_one_loss
from sklearn.cross_validation import KFold
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, chi2,SelectPercentile,f_classif, f_regression

input_data_zero = np.zeros((17,90))
input_data_one = np.zeros((24,90))
label= np.zeros((17,))
label1= np.ones((24,))
i=0
j=0
col=[]

with open('input.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
    	if i==0:
    		col = row[7:98]
    		i+=1
    		j+=1
    		continue
        
        # label1[i-1] = row[5] 
        if row[4]==str(0):
        	input_data_zero[j-1,:] = row[7:97]
        	label[j-1] = row[97]
        	j+=1
        else:
        	input_data_one[i-1,:] = row[7:97]
        	label1[i-1] = row[97]
        	i=i+1


data = preprocessing.normalize(input_data_zero)
# label = label1


choice = raw_input("Enter the choice \n 1. SVM Regression \n 2. SGD Regression \n 3. Linear Regression \n 4. LARS Lasso \n 5. Lasso Regression \n")
k=2
mean1=[]
kf = KFold(41, n_folds=k, shuffle=True)

if choice=="1" or choice=="0":
	mean1.append(SVM_Regression(kf,data,label,k))
if choice=="2" or choice=="0":
	mean1.append(SGD_Regression(kf,data,label,k))
if choice=="3" or choice=="0":
	mean1.append(Linear_Regression(kf, data, label, k))
if choice=="4" or choice=="0":
	mean1.append(Lars_Lasso(kf,data,label,k))
if choice=="5" or choice=="0":
	mean1.append(Lasso_Regression(kf,data,label,k))

print

method=['SVM Regression\t', 'SGD Regression\t', 'Linear Regression', 'LARS Lasso\t', 'Lasso Regression']

print mean1
print
print "MSR Table"
print
j=0
for m in method:
	print m+"\t"+str("{0:.4f}".format(mean1[j])),
	print
	j+=1

j=0
# classifiers = ["SVM Regression, 10%","RFC, 10%","Perceptron, 10%","SVM, 10%","Log Regression, 15%","RFC, 15%","Perceptron, 15%","SVM, 15%","Log Regression, 20%","RFC, 20%","Perceptron, 20%","SVM, 20%"]
x = np.arange(1)

y=[mean1[0]]
z=[mean1[1]]
k=[mean1[2]]
l=[mean1[3]]
j = [mean1[4]]
ax = plt.subplot(111)
w = 0.1
recs1 = ax.bar(x-0.20, y,width=w,color='b',align='center')
recs2 = ax.bar(x-0.10, z ,width=w,color='g',align='center')
recs3 = ax.bar(x, k,width=w,color='r',align='center')
recs4 = ax.bar(x+0.1,l ,width=w,color='y',align='center')
recs5 = ax.bar(x+0.2,j ,width=w,color='c',align='center')

ax.set_xticks(x+w)
# ax.set_xticklabels( ('10 Top', '15 Top', '20 Top') )
# ax.autoscale(tight=True)
# ax.set_ylim([0,1])
ax.set_ylabel('MRS')
ax.set_xlabel('Features')
ax.set_title('Mean Root Square')
ax.legend( (recs1[0], recs2[0], recs3[0],recs4[0], recs5[0]), ('SVM Regression', 'SGD Regression', 'Linear Regression', 'LARS Lasso', 'Lasso Regression') )

plt.show()








    
