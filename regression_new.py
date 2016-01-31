## func: using all other features to predict the left one feature, with different kinds of regression models
## two method will be used:
#	1. randomly select one feature as the output (repor that feature in plotting);
#	2. walk across all 91 features, and plot the distribution of all errors, for two classes and for different regression methods


import numpy as np
from sklearn.linear_model import SGDRegressor, LinearRegression, Lasso, LassoLars
from sklearn import svm, tree
from sklearn import cross_validation, metrics
from pylab import plot, show, savefig, xlim, figure, hold, ylim, legend, boxplot, setp, axes, ylabel, xlabel




# function for setting the colors of the box plots pairs
def setBoxColors(bp):
	setp(bp['boxes'][0], color='blue')
	setp(bp['caps'][0], color='blue')
	setp(bp['caps'][1], color='blue')
	setp(bp['whiskers'][0], color='blue')
	setp(bp['whiskers'][1], color='blue')
	setp(bp['fliers'][0], color='blue')
	setp(bp['medians'][0], color='blue')

	setp(bp['boxes'][1], color='red')
	setp(bp['caps'][2], color='red')
	setp(bp['caps'][3], color='red')
	setp(bp['whiskers'][2], color='red')
	setp(bp['whiskers'][3], color='red')
	setp(bp['fliers'][1], color='red')
	setp(bp['medians'][1], color='red')



## using all the models to predict, and return the errors (list) for different models
def prediction(kf, k, X, Y):

	error_list = []

	##==== linear regression
	error = 0
	for train_index, test_index in kf:
		X_train, X_test, Y_train, Y_test = X[train_index], X[test_index], Y[train_index], Y[test_index]
		log = LinearRegression()
		logit = log.fit(X_train, Y_train)
		Y_pred = logit.predict(X_test)
		error += metrics.mean_squared_error(Y_test, Y_pred)
	error = error / k
	error_list.append(error)


	##==== LARS Lasso regression
	error = 0
	for train_index, test_index in kf:
		X_train, X_test, Y_train, Y_test = X[train_index], X[test_index], Y[train_index], Y[test_index]
		log = LassoLars(alpha=0.1)
		logit = log.fit(X_train, Y_train)
		Y_pred = logit.predict(X_test)
		error += metrics.mean_squared_error(Y_test, Y_pred)
	error = error / k
	error_list.append(error)


	##==== Lasso regression
	error = 0
	for train_index, test_index in kf:
		X_train, X_test, Y_train, Y_test = X[train_index], X[test_index], Y[train_index], Y[test_index]
		log = Lasso(alpha=0.1)
		logit = log.fit(X_train, Y_train)
		Y_pred = logit.predict(X_test)
		error += metrics.mean_squared_error(Y_test, Y_pred)
	error = error / k
	error_list.append(error)


	##==== SVM regression
	error = 0
	for train_index, test_index in kf:
		X_train, X_test, Y_train, Y_test = X[train_index], X[test_index], Y[train_index], Y[test_index]
		log = svm.SVR()
		logit = log.fit(X_train, Y_train)
		Y_pred = logit.predict(X_test)
		error += metrics.mean_squared_error(Y_test, Y_pred)
	error = error / k
	error_list.append(error)


	##==== SGD regression
	error = 0
	for train_index, test_index in kf:
		X_train, X_test, Y_train, Y_test = X[train_index], X[test_index], Y[train_index], Y[test_index]
		log = SGDRegressor(loss='squared_loss', penalty='l2', alpha=0.0001, l1_ratio=0.15, n_iter=5)
		logit = log.fit(X_train, Y_train)
		Y_pred = logit.predict(X_test)
		error += metrics.mean_squared_error(Y_test, Y_pred)
	error = error / k
	error_list.append(error)

	return error_list





if __name__ == "__main__":


	matrix_pos = []
	matrix_neg = []


	file = open("input.csv" ,'r')
	line = file.readline()

	while 1:
		line = (file.readline()).strip()
		if not line:
			break

		line = line.split(',')
		response = line[4]
		corr = line[7]

		if response == '0':	## positive response
			feature_list = map(lambda x: float(x), line[7:])
			matrix_pos.append(feature_list)
		else:			## negative response
			feature_list = map(lambda x: float(x), line[7:])
			matrix_neg.append(feature_list)

	file.close()

	n_pos = len(matrix_pos)
	n_neg = len(matrix_neg)
	n_feature = len(matrix_pos[0])

	k = 4	## TODO: folder to be changed
	kf_pos = cross_validation.KFold(n_pos, n_folds=k, shuffle=False, random_state=None)
	kf_neg = cross_validation.KFold(n_neg, n_folds=k, shuffle=False, random_state=None)


	error_1 = [[], []]
	error_2 = [[], []]
	error_3 = [[], []]
	error_4 = [[], []]
	error_5 = [[], []]


	##=================================================================================
	# random generate output feature
	#index_output = np.random.random_integers(0, n_feature)


	##=================================================================================
	# walk through all the samples
	for index_output in range(n_feature):
		## the positive response matrix
		X = []
		Y = []
		for i in range(n_pos):
			X.append([])
			for j in range(n_feature):
				feature = matrix_pos[i][j]
				if j == index_output:
					Y.append(feature)
				else:
					X[i].append(feature)
		X = np.array(X)
		Y = np.array(Y)
		error_list_pos = prediction(kf_pos, k, X, Y)
		error_1[0].append(error_list_pos[0])
		error_2[0].append(error_list_pos[1])
		error_3[0].append(error_list_pos[2])
		error_4[0].append(error_list_pos[3])
		error_5[0].append(error_list_pos[4])

		# get the outlier
		#for error in error_list_pos:
		#	if error > 0.08:
		#		print index_output

		## the negative response matrix
		X = []
		Y = []
		for i in range(n_neg):
			X.append([])
			for j in range(n_feature):
				feature = matrix_neg[i][j]
				if j == index_output:
					Y.append(feature)
				else:
					X[i].append(feature)
		X = np.array(X)
		Y = np.array(Y)
		error_list_neg = prediction(kf_neg, k, X, Y)
		error_1[1].append(error_list_neg[0])
		error_2[1].append(error_list_neg[1])
		error_3[1].append(error_list_neg[2])
		error_4[1].append(error_list_neg[3])
		error_5[1].append(error_list_neg[4])

		# get the outlier
		#for error in error_list_neg:
		#	if error > 0.08:
		#		print index_output


	##=================================================================================
	# plotting: error_x, x=1, 2, 3, 4, 5
	fig = figure()
	ax = axes()
	hold(True)

	# first boxplot pair
	bp = boxplot(error_1, positions = [1, 2], widths = 0.6)
	setBoxColors(bp)

	# second boxplot pair
	bp = boxplot(error_2, positions = [4, 5], widths = 0.6)
	setBoxColors(bp)

	# thrid boxplot pair
	bp = boxplot(error_3, positions = [7, 8], widths = 0.6)
	setBoxColors(bp)

	# 4 boxplot pair
	bp = boxplot(error_4, positions = [10, 11], widths = 0.6)
	setBoxColors(bp)

	# 5 boxplot pair
	bp = boxplot(error_5, positions = [13, 14], widths = 0.6)
	setBoxColors(bp)


	# set axes limits and labels
	xlim(0, 15)
	ylim(0, 0.15)
	ax.set_xticklabels(['linear', 'LARS Lasso', 'Lasso', 'SVM', 'SGD'])
	ax.set_xticks([1.5, 4.5, 7.5, 10.5, 13.5])

	# draw temporary red and blue lines and use them to create a legend
	hB, = plot([1,1],'b-')
	hR, = plot([1,1],'r-')
	legend((hB, hR),('positive response class', 'negative response class'))
	hB.set_visible(False)
	hR.set_visible(False)
	ylabel('prediction errors (after cross validation)')

	savefig('boxcompare.png')
	show()


