import sys
import os
import numpy as np
import cv2
import SimpleITK as sitk
import h5py

### Helper functions
def load_nrrd(full_path_filename):
	
	# this function loads .nrrd files into a 3D matrix and outputs it
	# the input is the specified file path
	# the output is the N x A x B for N slices of sized A x B
	
	data = sitk.ReadImage(full_path_filename)						# read in image
	data = sitk.Cast(sitk.RescaleIntensity(data),sitk.sitkUInt8)	# convert to 8 bit (0-255)
	data = sitk.GetArrayFromImage(data)								# convert to numpy array
	
	return(data)

def create_folder(full_path_filename):
	
	# this function creates a folder if its not already existed
	if not os.path.exists(full_path_filename):
		os.makedirs(full_path_filename)

	return

### ---------- initialize image sizes/cropping ---------------------------------------------------------------------------------------------------

n1 = 112						# set input image and label size (cropped from image center). 400,320,240,160

N_train_patients = 5 #100		# number of patients to use from train set
N_test_patients  = 1 #54		# number of patients to use from the test set

### ----------------------------------------------------------------------------------------------------------------------------------------------

# create an output folder for the images if it wasnt already created
create_folder("UTAH Test set")
create_folder("UTAH Test set/log")

# list all the files in training and testing sets
train_files = os.listdir("Training Set")
test_files = os.listdir("Testing Set")

# counters for the number of correct data sets, timer, and initialise the image and label arrays
Image,Label = [],[]

# loop through all training patients
for i in range(N_train_patients):

	# read in the MRI scan
	patient_3DMRI_scan = load_nrrd(os.path.join("Training Set",train_files[i],'lgemri.nrrd'))
	
	# cavity labels (1 = positive, 0 = negative)
	patient_3DMRI_cavity = load_nrrd(os.path.join("Training Set",train_files[i],'laendo.nrrd'))//255

	# move dimension one to the end so that dimension = X by Y x Z directions
	patient_3DMRI_scan   = np.rollaxis(patient_3DMRI_scan,0,3)
	patient_3DMRI_cavity = np.rollaxis(patient_3DMRI_cavity,0,3)

	# based off the image size, and the specified input size, find coordinates to crop image
	midpoint = patient_3DMRI_cavity.shape[0]//2
	n11, n12 = midpoint - int(n1/2), midpoint + int(n1/2)
	
	# local image label for scan
	for n_slice in range(patient_3DMRI_scan.shape[2]):

		Image.append(patient_3DMRI_scan[n11:n12,n11:n12,n_slice])
		Label.append(patient_3DMRI_cavity[n11:n12,n11:n12,n_slice])

# creating test set by writing to output tiff
for i in range(N_test_patients):
	
	# read in the MRI scan
	patient_3DMRI_scan = load_nrrd(os.path.join("Testing Set",test_files[i],'lgemri.nrrd'))
	
	# cavity labels (1 = positive, 0 = negative)
	patient_3DMRI_cavity = load_nrrd(os.path.join("Testing Set",test_files[i],'laendo.nrrd'))//255

	# move dimension one to the end so that dimension = X by Y x Z directions
	patient_3DMRI_scan   = np.rollaxis(patient_3DMRI_scan,0,3)
	patient_3DMRI_cavity = np.rollaxis(patient_3DMRI_cavity,0,3)
	
	# create an output folder for the single patient
	create_folder(os.path.join("UTAH Test set",str(i+1)+" - "+test_files[i]))
	
	# create output folders for the scan and label
	create_folder(os.path.join("UTAH Test set",str(i+1)+" - "+test_files[i],))						# folder for patient single scan
	create_folder(os.path.join("UTAH Test set",str(i+1)+" - "+test_files[i],"data"))				# folder for data
	create_folder(os.path.join("UTAH Test set",str(i+1)+" - "+test_files[i],"cavity"))				# folder for CAVITY labels
	create_folder(os.path.join("UTAH Test set",str(i+1)+" - "+test_files[i],"auto segmentation"))	# folder for automatic segmentation
	
	for n_slice in range(patient_3DMRI_scan.shape[2]):
	
		# slice name with correct format for AMIRA
		output_filename = "slice"+"{0:03}".format(n_slice+1)+".tiff"
		
		# write to data and label file
		cv2.imwrite(os.path.join("UTAH Test set",str(i+1)+" - "+test_files[i],"data",output_filename),patient_3DMRI_scan[:,:,n_slice])
		cv2.imwrite(os.path.join("UTAH Test set",str(i+1)+" - "+test_files[i],"cavity",output_filename),patient_3DMRI_cavity[:,:,n_slice]*255)

Image,Label = np.array(Image),np.array(Label)

# encoding label to neural network output format
temp = np.empty(shape=[Label.shape[0],n1,n1,2])
temp[:,:,:,0] = 1-Label
temp[:,:,:,1] = Label

Image,Label = np.reshape(Image,newshape=[-1,n1,n1,1]),np.reshape(temp,newshape=[-1,n1,n1,2])

# calculate train mean and standard deviation
train_mean,train_sd = np.mean(Image),np.std(Image)

# normalise the training data
Image = (Image-train_mean)/train_sd

# create a HDF5 dataset
h5f = h5py.File('UTAH Test set/Utah_Training.h5','w')
h5f.create_dataset("image",			data=Image)
h5f.create_dataset("label",			data=Label)
h5f.create_dataset("train.mean",	data=train_mean)
h5f.create_dataset("train.sd",		data=train_sd)
h5f.close()