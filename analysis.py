import sys
import numpy as np 
import nibabel as nib
import matplotlib

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from scipy import stats
import scipy
from PIL import Image
from scipy.optimize import curve_fit
from statsmodels.tsa.stattools import ccf

import pandas as pd
import time
import math
from decimal import *
import nitime

voxels=[]
# seed_regions = ['Ventral rostral putamen', 'Nucleus Accumbens', 'Dorsal Caudate', 'Dorsal Caudal Putamen','Dorsal Rostral putamen','Ventral Caudate']

# points = [ (28,33,9), (45,21,13),(30,28,9), (24,13,15), (37,13,16), (45,32,20), 
# 					(45,3,16), (49,28,14), (36,0,25), (36,17,15), (14,10,25), (27,0,24),
# 					(49,1,11), (24,33,9), (39,27,14), (30,28,14), (43,12,17), (29,0,18), (17,0,17), (37,29,28),
# 					(50,21,17), (16,24,13), (45,19,14), (13,20,16), (45,31,20),
# 					(16,26,13), (45,19,14), (27,33,10), (28,40,16), (52,6,14), (45,31,20),
# 					(37,28,11), (30,27,13), (43,23,5), (27,13,16), (43,15,18), (35,16,16), (27,0,18), (18,10,17)] 
				
def find_nearbypoints(distance):
	x,y,z = np.mgrid[0:64,0:64,0:40]
	data = zip(x.ravel(), y.ravel(), z.ravel())
	point_tree = spatial.cKDTree(data)

	indices = list(list(np.zeros((1,3))))
	# This finds the index of all points within distance 1 of [1.5,2.5].
	for point in points:
		indices.append(point_tree.data[point_tree.query_ball_point(point, distance)])
	return indices

def store_elements(filename):
	example_filename = filename
	img = nib.load(example_filename)
	hdr = img.header
	axes = hdr.get_xyzt_units()

	img_array = img.get_data()
	return img_array

def pre_processing(img_array, t1,t2,threshold):
	for i in range(0,64):
		for j in range(0,64):
			for k in range(0,40):
				std = np.std(img_array[i,j,k,t1:t2])
				if(i==43 and j==23 and k==5):
					print std, img_array[i,j,t1:t2]
				if(std<=threshold):
					voxels.append(str(i)+','+str(j)+','+str(k))
				else:
					continue

def in_voxel(x,y,z):
	if str(str(x)+','+str(y)+','+str(z)) in voxels:
		return True
	return False

def get_series(img_array,x,y,z, t1,t2):
	series = pd.Series(np.zeros(shape=(t2-t1,)))
	for t in range(t1,t2):
		series[t-t1,0] = img_array[x,y,z,t]
	return series

def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

def shuffle(img_array,x,y,z,t1,t2,x1,y1,z1, shuffle_num):
	print x,y,z,t1,t2

	series1 = img_array[x,y,z,t1:t2]
	series2 = img_array[x1,y1,z1,t1:t2]
	org_series1 = series1
	org_series2 = series2

	corr = np.zeros(shuffle_num,)
	for i in range(0, shuffle_num):
		random.shuffle(series1)
		random.shuffle(series2)
		corr[i]= pd.Series(series1).corr(pd.Series(series2))




	
	print('normality =', scipy.stats.normaltest(corr))
	from scipy.stats import norm
		# Fit a normal distribution to the data:
	mu, std = norm.fit(corr)

	# Plot the histogram.
	plt.hist(corr, bins=30, normed=True, alpha=0.6, color='g')

	# Plot the PDF.
	xmin, xmax = plt.xlim()
	x = np.linspace(xmin, xmax, 100)
	p = norm.pdf(x, mu, std)
	plt.plot(x, p, 'k', linewidth=2)
	title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
	plt.title(title)

	# plt.show()

	


	# x = range(0,shuffle_num)
	# n = len(x)                          #the number of data
	# mean = sum(x*corr)/n                   #note this correction
	# sigma = sum(corr*(x-mean)**2)/n       #note this correction

	# def gaus(x,a,x0,sigma):
	#     return a*exp(-(x-x0)**2/(2*sigma**2))

	# popt,pcov = curve_fit(gauss,x,corr,p0=[1,mean,sigma])

	# plt.plot(x,corr,'b+:',label='data')
	# plt.plot(x,gauss(x,*popt),'ro:',label='fit')
	# plt.legend()
	# plt.title('Fig. 3 - Fit for Time Constant')
	# plt.xlabel('(s)')
	# plt.ylabel('(V)')
	plt.show()
	return np.mean(corr)

def bootstrap(series1, boot_num):
	t = len(series1)
	output = np.zeros((t,1))
	if(boot_num < len(series1)):
		for i in range(0,boot_num):
			if i==0:
				output = np.random.choice(series1, (t,1), replace=True)
				continue
			temp = np.random.choice(series1, (t,1), replace=True)
			
			np.vstack((output,temp))
	output = np.mean(output, axis=0)

	return output


def find_average_intensity(filename, t1,t2,boot_num):
	img_array = store_elements(filename)
	img_array_new = np.zeros((64,64,40))
	for i in range(0,64):
		for j in range(0,64):
			for k in range(0,40):
				img_array_new[i,j,k] = bootstrap(img_array[i,j,k,t1:t2], boot_num)

	plt.imshow(img_array_new[:,:,39])
	plt.show()
	return img_array_new

	# from mayavi import mlab
	# s = 64
	# t=40
	# x,y,z = np.ogrid[0:64,0:64,0:40]

	# data = img_array_new

	# grid = mlab.pipeline.scalar_field(data)
	# grid.spacing = [1.0, 1.0, 1.0]

	# contours = mlab.pipeline.contour_surface(grid,contours=[5,15,25], transparent=True)
	# mlab.show()


def find_pair_wise_correlation(img_array,t1,t2,i,j,k):
	series1 = pd.Series(img_array[i,j,k,t1:t2])
	corr = np.zeros((64,64,40))
	corr_p = np.zeros((64,64,40))

	print "here"
	for i1 in range(0,64):
		for j1 in range(0,64):
			for k1 in range(0,40):
				if in_voxel(i,j,k) or in_voxel(i1,j1,k1):
					corr[i1,j1,k1]= -5
				else:
					series2 = pd.Series(img_array[i1,j1,k1,t1:t2])
					corr[i1,j1,k1],corr_p[i1,j1,k1] = stats.pearsonr(img_array[i,j,k,t1:t2], img_array[i1,j1,k1,t1:t2])

	corr1 = dict()
	corr = {"correlation":corr, "p-value":corr_p}

	return corr
	

def find_max_correlation(img_array,t1,t2,i,j,k,threshold):
	corr = find_pair_wise_correlation(img_array,t1,t2,i,j,k)	

	index = np.where(corr["correlation"]!=-5)
	print index,len(index[0])	
	temp = np.zeros(len(index[0]))
	temp_p = np.zeros(len(index[0]))

	temp = corr["correlation"][index]
	temp_p = corr["p-value"][index]

	print temp,temp_p
	# for i in temp:

	import matplotlib.pyplot as plt

	img_array_new = find_average_intensity("cmh_01_func.nii", 0,212,50)
	img_array_new1 = img_array_new
	print "shape ",img_array_new.shape
	plt.imshow(img_array_new[:,:,0])
	plt.show() 
	print img_array_new
	
	plt.imshow(img_array_new1[:,:,0], cmap='Greys', interpolation='nearest')
	

	# img_array_new[index] = temp
	mask = np.ones(a.shape,dtype=bool) #np.ones_like(a,dtype=bool)
	mask[indices] = False
	img_array_new[mask] = temp
	img_array_new[~mask] = 0
	plt.imshow(img_array_new[:,:,0])

	plt.show()

def find_min_correlation(img_array,t1,t2,i,j,k,threshold):
	corr = find_pair_wise_correlation(img_array,t1,t2,i,j,k)
	index = np.where(corr!=-5)
	print corr
	print
	print
	# print corr[index]
	return index

def find_correlation_option3(filename, x,y,z, t1, t2, threshold):
	img_array = store_elements(filename)
	pre_processing(img_array, 0, 212, 0)
	find_max_correlation(img_array, t1,t2,x,y,z,threshold)
	# val = find_max_correlation(img_array, t1,t2,x,y,z,threshold)
	# f = open('Max_Correlation','a')
	# print p-value
	# f.write(str(val))
	# f.close()

def find_correlation_option4(filename, x,y,z, t1, t2, threshold):
	img_array = store_elements(filename)
	val = find_min_correlation(img_array, t1,t2,x,y,z,threshold)
	f = open('Min_Correlation','a')
	print val
	f.write(val)
	f.close()



def correlation(img_array, x,y,z, x1,y1,z1, t1,t2):
	
	if(in_voxel(x,y,z) or in_voxel(x1,y1,z1)):
		return -5
	series1 = get_series(img_array,x,y,z,t1,t2)
	series2 = get_series(img_array,x1,y1,z1,t1,t2)
	# plt.plot(series1)
	# plt.plot(series2)
	return series1.corr(series2)

def find_correlation_option1(filename, x,y,z,x1,y1,z1, t1,t2):
	img_array = store_elements(filename)
	shuffle(img_array,x,y,z,t1,t2,x1,y1,z1,250)
	pre_processing(img_array, 0, 212, 0)
	corr = correlation(img_array,x,y,z,x1,y1,z1,t1,t2)

	# f = open('Results_correlation','a')
	# f.write("Correlation between ")
	# f.write(str(x)+" "+str(y)+" "+str(z))
	# f.write(" & ")
	# f.write(str(x1)+" "+str(y1)+" "+str(z1))
	# f.write(" is ")
	# f.write(str(corr))
	# f.write("\n")
	# f.close()

	print "Correlation between ",x,y,z," & ",x1,y1,z1," is ", corr
	series1 = get_series(img_array,x,y,z,t1,t2)
	series2 = get_series(img_array,x1,y1,z1, t1,t2)
	s1,= plt.plot(series1)
	s2,= plt.plot(series2)
	plt.legend([s1, s2], ['Voxel1', 'Voxel2'])
	plt.title('Signal')
	plt.xlabel('Time Period', fontsize=14, color='red')
	plt.ylabel('Intensity', fontsize=14, color='red')
	plt.show()


def find_corr(img_array,t1,t2,t3,t4,x,y,z,x1,y1,z1):
	threshold = shuffle(img_array,x,y,z,t1,t2,x1,y1,z1,500)
	series1 = img_array[x,y,z,t1:t2]
	series2 = img_array[x1,y1,z1,t3:t4]
	corr = ccf(series1,series2)
	print corr

	

	fig = plt.figure()
	x = np.arange(t2-t1)
	ax = fig.add_subplot(111)
	ax.set_ylim(-1,1)
	plt.plot(corr,marker='o', color='r')
	for i,j in zip(x,corr):
		ax.annotate(str("{0:.4f}".format(j)),xy=(i,j))

	plt.xlabel('Time Lag')
	plt.ylabel('Correlation')
	plt.show()

#Input of format - x,y,z,x1,y2,z2
choice = raw_input("Enter the choice ") or "1"

if choice=="1":
	filename = raw_input("Enter the filename ") or "cmh_01_func.nii"
	t1 = raw_input("Enter the start time") or "0"
	t2 = raw_input("Enter the end time") or "60"
	x,y,z = raw_input("Enter the voxel 1") or "43","23","5"
	x1,y1,z1 = raw_input("Enter the voxel 2") or "30","28","9"
	start_time = time.clock()
	find_correlation_option1(filename, int(x),int(y),int(z), int(x1),int(y1),int(z1),int(t1),int(t2))
	print("--- %s minutes ---" % str((time.time() - start_time)/60))

elif choice=="2":
	print "Find average intensity using bootstrap "
	filename = raw_input("Enter the filename ") or "cmh_01_func.nii"
	t1 = raw_input("Enter the start time") or "0"
	t2 = raw_input("Enter the end time") or "212"
	boot_num = raw_input("Enter the boostrap resampling ") or "50"
	start_time = time.clock()
	find_average_intensity(filename,int(t1), int(t2), int(boot_num))
	print("--- %s minutes ---" % str((time.time() - start_time)/60))
	
elif choice=="3":
	print "Maximum correlation"
	filename = raw_input("Enter the filename ") or "cmh_01_func.nii"
	t1 = raw_input("Enter the start time") or "0"
	t2 = raw_input("Enter the end time") or "212"
	x,y,z = raw_input("Enter the voxel 1") or "43","23","5"
	threshold = raw_input("Enter the threshold") or "0.2"
	start_time = time.clock()
	find_correlation_option3(filename, int(x),int(y),int(z), int(t1), int(t2), float(threshold))
	print("--- %s minutes ---" % str((time.time() - start_time)/60))

elif choice=="4":
	print "Least correlation"
	filename = raw_input("Enter the filename ") or "cmh_01_func.nii"
	t1 = raw_input("Enter the start time") or "0"
	t2 = raw_input("Enter the end time") or "212"
	x,y,z = raw_input("Enter the voxel 1") or "43","23","5"
	threshold = raw_input("Enter the threshold") or "0.2"
	start_time = time.clock()
	find_correlation_option4(filename, int(x),int(y),int(z), int(t1), int(t2), float(threshold))
	print("--- %s minutes ---" % str((time.time() - start_time)/60))

else:
	filename = raw_input("Enter the filename ") or "cmh_02_func.nii"
	img_array = store_elements(filename)
	t1 = raw_input("Enter the start time") or "0"
	t2 = raw_input("Enter the end time") or "60"
	t3 = raw_input("Enter the start time") or "0"
	t4 = raw_input("Enter the end time") or "60"
	x,y,z = raw_input("Enter the voxel 1") or "43","23","5"
	x1,y1,z1 = raw_input("Enter the voxel 2") or "30","28","9"
	# "43","23","6"

	find_corr(img_array,int(t1),int(t2),int(t3),int(t4),int(x),int(y),int(z),int(x1),int(y1),int(z1))

