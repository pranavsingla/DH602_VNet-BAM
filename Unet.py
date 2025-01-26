from tflearn.layers.core import input_data,dropout
from tflearn.layers.conv import conv_2d,max_pool_2d,conv_2d_transpose
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression
from tflearn.models.dnn import DNN
from sklearn.metrics import f1_score
import h5py
import cv2
import numpy as np
import os


### Helper Functions
def predict(dir_path,CNN_model,mu=0,sd=1):

	# ********************************************************************************************************************
	n1 = 112
	# ********************************************************************************************************************
	
	# get all the files for testing
	files = os.listdir(dir_path)
	files.remove("log")
	files.remove("Utah_Training.h5")
		
	for i in range(len(files)):
	
		# get all the files for that single patient and remove the annoying file that MAC users have ffs

		print("Segmenting: "+os.path.join(files[i]))
		
		# get the shape of the image and number of slices
		temp = cv2.imread( os.path.join(dir_path,files[i],"data","slice001.tiff") ,cv2.IMREAD_GRAYSCALE)
		number_of_slices = len(os.listdir(os.path.join(dir_path,files[i],"data")))
		
		# based off the image size, and the specified input size, find coordinates to crop image
		midpoint = temp.shape[0]//2
		n11, n12 = midpoint - int(n1/2), midpoint + int(n1/2)
		
		# initialise temp array for prediction
		input1 = np.zeros(shape=[number_of_slices,n1,n1])
		
		#***  Loading data
		for n in range(number_of_slices):
		
			input_filename = "slice"+"{0:03}".format(n+1)+".tiff"
			
			ImageIn = cv2.imread(os.path.join(dir_path,files[i],"data",input_filename),cv2.IMREAD_GRAYSCALE)
			ImageIn = (ImageIn-mu)/sd
			
			input1[n,:,:] = ImageIn[n11:n12,n11:n12]
		
		#*** Making predictions
		output = np.zeros(shape=[number_of_slices,n1,n1,2])
		for n in range(number_of_slices):
			output[n,:,:,:] = CNN_model.predict([input1[n,:,:,None]])
		
		output = np.argmax(output,3)
		
		#*** Writing data to output
		for n in range(number_of_slices):
		
			Imout = np.zeros(shape=[temp.shape[0],temp.shape[1]])
			Imout[n11:n12,n11:n12] = output[n,:,:]

			output_filename = "slice"+"{0:03}".format(n+1)+".tiff"
			cv2.imwrite(os.path.join(dir_path,files[i],"auto segmentation",output_filename),255*Imout)

def Score(dir_path,log_path):

	# create a txt file to write the results to
	f = open(os.path.join(log_path,"log.txt"),"a")
	f1_scores = []
	
	# get all the files for testing
	files = os.listdir(dir_path)
	files.remove("log")
	files.remove("Utah_Training.h5")
	
	for i in range(len(files)):
			
		temp = cv2.imread(os.path.join(dir_path,files[i],"auto segmentation","slice001.tiff"),cv2.IMREAD_GRAYSCALE)
		pred = np.zeros([len(os.listdir(os.path.join(dir_path,files[i],"auto segmentation"))),temp.shape[0],temp.shape[1]])
		
		temp = cv2.imread(os.path.join(dir_path,files[i],"cavity","slice001.tiff"),cv2.IMREAD_GRAYSCALE)
		true = np.zeros([len(os.listdir(os.path.join(dir_path,files[i],"cavity"))),temp.shape[0],temp.shape[1]])
		
		for k in range(pred.shape[0]):
			
			input_filename = "slice"+"{0:03}".format(k+1)+".tiff"
			pred[k,:,:] = cv2.imread(os.path.join(dir_path,files[i],"auto segmentation",input_filename),cv2.IMREAD_GRAYSCALE)//255
			true[k,:,:] = cv2.imread(os.path.join(dir_path,files[i],"cavity",input_filename),cv2.IMREAD_GRAYSCALE)//255 
		
		# calculate f1 score
		pred_f1,true_f1 = pred.flatten(),true.flatten()
		f1 = f1_score(pred_f1,true_f1,average="binary")
		
		f.write(files[i]+" - F1 Score: "+str(round(f1,3))+"\n")
		f1_scores.append(f1)

	f.write("\nOVERALL F1 AVEARGE = "+str(np.mean(np.array(f1_scores))))
	f.write("\n\n" )
	f.close()

### Computation Graph
block1a = input_data(shape=[None,112,112,1])
block1a = conv_2d(block1a,64,3,activation='relu')
block1a = conv_2d(block1a,64,3,activation='relu')

block2a = max_pool_2d(block1a,2,2)

block2a = conv_2d(block2a,128,3,activation='relu')
block2a = conv_2d(block2a,128,3,activation='relu')

block3a = max_pool_2d(block2a,2,2)

block3a = conv_2d(block3a,256,3,activation='relu')
block3a = conv_2d(block3a,256,3,activation='relu')

block4a = max_pool_2d(block3a,2,2)

block4a = conv_2d(block4a,512,3,activation='relu')
block4a = conv_2d(block4a,512,3,activation='relu')
block4a = dropout(block4a,0.5)

block5 = max_pool_2d(block4a,2,2)

block5 = conv_2d(block5,1024,3,activation='relu')
block5 = conv_2d(block5,1024,3,activation='relu')
block5 = dropout(block5,0.5)

block4b = conv_2d_transpose(block5,512,3,[block5.shape[1].value*2,block5.shape[2].value*2,512],[1,2,2,1])
block4b = merge([block4a,block4b],'concat',axis=3)
block4b = conv_2d(block4b,512,3,activation='relu')
block4b = conv_2d(block4b,512,3,activation='relu')

block3b = conv_2d_transpose(block4b,256,3,[block4b.shape[1].value*2,block4b.shape[2].value*2,256],[1,2,2,1])
block3b = merge([block3a,block3b],'concat',axis=3)
block3b = conv_2d(block3b,256,3,activation='relu')
block3b = conv_2d(block3b,256,3,activation='relu')

block2b = conv_2d_transpose(block3b,128,3,[block3b.shape[1].value*2,block3b.shape[2].value*2,128],[1,2,2,1])
block2b = merge([block2a,block2b],'concat',axis=3)
block2b = conv_2d(block2b,128,3,activation='relu')
block2b = conv_2d(block2b,128,3,activation='relu')

block1b = conv_2d_transpose(block2b,64,3,[block2b.shape[1].value*2,block2b.shape[2].value*2,64],[1,2,2,1])
block1b = merge([block1a,block1b],'concat',axis=3)
block1b = conv_2d(block1b,64,3,activation='relu')
block1b = conv_2d(block1b,64,3,activation='relu')

Clf     = conv_2d(block1b,2,1,1,activation='softmax')
regress = regression(Clf,optimizer='adam',loss='categorical_crossentropy',learning_rate=0.0001)
model   = DNN(regress)

### Training
file_path = "UTAH Test set"
log_path  = "UTAH Test set/log"
data_path = "UTAH Test set/Utah_Training.h5"

f = open(os.path.join(log_path,"log.txt"),"w");f.close()

data = h5py.File(data_path,"r")

### Training Evaluation
for n in range(1,1000):
	
	f = open(os.path.join(log_path,"log.txt"),"a")
	f.write("-"*75+" Epoch "+str(n)+"\n")
	f.close()
	
	model.fit(data["image"],data["label"],n_epoch=1,show_metric=True,batch_size=4,shuffle=True)
	#model.save(os.path.join(log_path,"model"+str(n)))
	
	predict(file_path,model,data["train.mean"].value,data["train.sd"].value)
	Score(file_path,log_path)