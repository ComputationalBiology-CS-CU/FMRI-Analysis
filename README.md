Author: Chithra Srinivasan

Introduction:
=============
To run the program follow the format:
python analysis.py

Prompts:

1. filename is the name of the nii file
2. x1,y1,z1 is the voxel of the seed 1
3. x2,y2,z2 is the voxel of the seed 2
4. choice is 1 or 2 - 1 is for finding correlation between two seeds - 2 is for finding significant correlation


Classification:
python classification.py
(Choose one of the techniques. "0" for all the techniques to run)

Regression:
python regression.py
(Choose one of the techniques. Eg., 1, 2 or 5. "0" for all the techniques to run)

Installation:
=============
You would need the following python packages installed:
numpy, scipy, pandas, nibabel, matplotlib

Input Files:
============
cmh_01_func.nii

cmh_02_func.nii

cmh_03_func.nii

mrc_01_func.nii

mrc_02_func.nii

mrc_03_func.nii

These are the various regions:
==============================
The coordinates are specified as values to the keys: Eg., [28,33,9], [30,28,9]
These coordinates are defined as 'points' in analysis.py

'''

{'ventral rostral putamen':
									{'orbital frontal cortex': np.array([28,33,9]),
									 'Insula': np.array([45,21,13]), 
									 'Subcollosal cortex': np.array([30,28,9]),
									 'Thalamus_+': np.array([24,13,15]),
									 'Thalamus_-': np.array([37,13,16]),
									 'Dorsolateral Prefrontal cortex': np.array([45,32,20])
									 },
							 
	'Nucleus Accumbens':
									{'Middle Temporal Gyrus': np.array([45,3,16]),
									'Inferios frontal gyrus': np.array([49,28,14]),
									'Superior parietal lobule': np.array([36,0,25]),
									'Thalamus': np.array([36,17,15]),
									'Supramarginal gyrus': np.array([14,10,25]),
									'Lateral occipital cortex': np.array([27,0,24])
									},

	'Dorsal Caudate':
								   {'inferior temporal gyrus': np.array([49,1,11]),
								   'frontal pole': np.array([24,33,9]),
								   'putamen': np.array([39,27,14]),
								   'accumbens': np.array([30,28,14]),
								   'planum temporale_-': np.array([43,12,17]),
								   'occipital cortex': np.array([29,0,18]),
								   'planum temporale_+': np.array([17,0,17]),
								   'superior frontal gyrus': np.array([37,29,28])
								   },

	'Dorsal caudal putamen':
									{'Insula_-': np.array([50,21,17]),
									'Insula_+': np.array([16,24,13]),
									'Insula_=': np.array([45,19,14]),
									'planum polare': np.array([13,20,16]),
									'dorsolateral prefrontal cortex': np.array([45,31,20])
								  	},
								
	'Dorsal rostral putamen':
									{'Frontal operculum cortex': np.array([16,26,13]),
									'Insula': np.array([45,19,14]),
									'Orbital frontal cortex': np.array([27,33,10]),
									'Anterior cingulate cortex': np.array([28,40,16]),
									'Middle temporal gyrus': np.array([52,6,14]),
									'Middle frontal gyrus': np.array([45,31,20])
									},

	'Ventral caudate':
									{'Accumbens, putamen': np.array([37,28,11]), 
									'Accumbens, caudate': np.array([30,27,13]),
									'middle frontal gyrus': np.array([43,23,5]),
									'Thalamus_-': np.array([27,13,16]),
									'Insula': np.array([43,15,18]),
									'Thalamus_+': np.array([35,16,16]),
									'Occipital cortex': np.array([27,0,18]),
									'supramargival gyrus': np.array([18,10,17])
									}
}

'''



# update:

The new regression code is added, to draw the prediction errors treating all features as output (one by one).
