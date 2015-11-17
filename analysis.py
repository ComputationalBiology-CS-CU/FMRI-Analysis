import os,sys
import numpy as np 
import nibabel as nib
from nibabel.testing import data_path
import matplotlib.pyplot as plt

import pandas as pd
import scipy.spatial as spatial
import time

seed_regions = ['Ventral rostral putamen', 'Nucleus Accumbens', 'Dorsal Caudate', 'Dorsal Caudal Putamen','Dorsal Rostral putamen','Ventral Caudate']

points = [ (28,33,9), (45,21,13),(30,28,9), (24,13,15), (37,13,16), (45,32,20), 
					(45,3,16), (49,28,14), (36,0,25), (36,17,15), (14,10,25), (27,0,24),
					(49,1,11), (24,33,9), (39,27,14), (30,28,14), (43,12,17), (29,0,18), (17,0,17), (37,29,28),
					(50,21,17), (16,24,13), (45,19,14), (13,20,16), (45,31,20),
					(16,26,13), (45,19,14), (27,33,10), (28,40,16), (52,6,14), (45,31,20),
					(37,28,11), (30,27,13), (43,23,5), (27,13,16), (43,15,18), (35,16,16), (27,0,18), (18,10,17)] 
				
				

def find_nearbypoints(distance):
	x,y,z = np.mgrid[0:64,0:64,0:40]
	data = zip(x.ravel(), y.ravel(), z.ravel())
	point_tree = spatial.cKDTree(data)

	indices = list(list(np.zeros((1,3))))
	# This finds the index of all points within distance 1 of [1.5,2.5].
	for point in points:
		indices.append(point_tree.data[point_tree.query_ball_point(point, distance)])

	return indices

def find_avg_intensity(indices, img_array, time):
	X=[]
	for t in range(time[0],time[1]):
		avg_intensity = list()
		
		for index_list in range(1,len(indices)):
			# print indices[index_list]
			avg = 0
			for index in indices[index_list]:
				[x,y,z] = index
				avg+=img_array[x,y,z,t]

			avg= avg/len(indices[index_list])
			
			avg_intensity.append(avg)
		X.append(avg_intensity)
	# print X
	return X

def get_series(img_array,voxel, t1,t2):
	x,y,z = voxel
	series = pd.Series(np.zeros(shape=(t2-t1,)))
	for t in range(t1,t2):
		series[t-t1,0] = img_array[x,y,z,t]
	return series

def store_elements(filename):
	example_filename = filename
	img = nib.load(example_filename)
	hdr = img.header
	axes = hdr.get_xyzt_units()

	img_array = img.get_data()
	return img_array

def correlation(img_array, voxel1, voxel2, t1,t2):
	series1 = get_series(img_array,voxel1,t1,t2)
	series2 = get_series(img_array,voxel2, t1,t2)
	# plt.plot(series1)
	# plt.plot(series2)

	
	return series1.corr(series2)


def find_correlation(filename, voxel1, voxel2, t1, t2, time_period):
	img_array = store_elements(filename)

	corr = correlation(img_array,voxel1,voxel2,t1,t2)
	f = open('Results_correlation','a')
	f.write("Correlation between ")
	f.write(str(voxel1))
	f.write(" & ")
	f.write(str(voxel2))
	f.write(" is ")
	f.write(str(corr))
	f.write("\n")
	f.close()

	print "Correlation between ",voxel1," & ",voxel2," is ", corr
	series1 = get_series(img_array,voxel1,t1,t2)
	series2 = get_series(img_array,voxel2, t1,t2)
	plt.plot(series1)
	plt.plot(series2)

	plt.show()

def find_significant_correlation(filename, voxel1, voxel2, t1, t2, time_period):
	img_array = store_elements(filename)
	t = t1
	corr=[]
	time=[]
	while(t<t2):
		corr.append(correlation(img_array,voxel1,voxel2,t,t+time_period))
		t+=time_period
		time.append(t)

	f = open('Results_significant_correlation','a')
	f.write("Correlation between ")
	f.write(str(voxel1))
	f.write(" & ")
	f.write(str(voxel2))
	f.write(" is ")
	f.write(str(corr))
	f.write(" for time periods ")
	f.write(str(t))
	f.write("\n")
	f.close()

	plt.scatter(time,corr)
	plt.show()

#Input of format - x,y,z,x1,y2,z2

start_time = time.clock()

filename = sys.argv[1]
inputVoxel = sys.argv[2].split('\'')[0].split(',')

choice = sys.argv[3]
t1 = int(sys.argv[4])
t2 = int(sys.argv[5])
time_period = int(sys.argv[6])
voxel1 = np.array([int(inputVoxel[0]), int(inputVoxel[1]), int(inputVoxel[2])])
voxel2 = np.array([int(inputVoxel[3]), int(inputVoxel[4]), int(inputVoxel[5])])

print "The given inputs: \n 1. Filename: ",filename,"\n2. Voxel1: ",voxel1,"\n3. Voxel2 : ",voxel2,"\n4. Starting Time : ",t1,"\n5. Ending Time: ",t2,"\n6. Time Period : ",time_period


if choice=="1":
	find_correlation(filename, voxel1, voxel2,t1, t2, time_period)
elif choice=="2":
	find_significant_correlation(filename, voxel1, voxel2, t1, t2, time_period)

print("--- %s seconds ---" % (time.time() - start_time))
