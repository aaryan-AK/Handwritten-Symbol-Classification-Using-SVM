import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from skimage import filters
from skimage.feature import hog
import cv2
import pickle
import bz2
import _pickle as cPickle
import skimage.transform as st

#################################################
# Please enter the path to the Images and Labels
# you will be using for testing
test_img_array = './Images2.npy'
test_labels = './Labels2.npy'
#################################################

# Function Definitions #

# Load any compressed pickle file
def decompress_pickle(file):
	data = bz2.BZ2File(file, 'rb')
	data = cPickle.load(data)
	return data
	
def test_func(test_images):
	test_images=test_images.astype('uint8')
	n = len(test_images)
	features = np.zeros((n,1089))

	# HOG Settings
	winSize = (60,60)
	blockSize = (10,10)
	blockStride = (5,5)
	cellSize = (10,10)
	nbins = 9
	derivAperture = 1
	winSigma = -1.
	histogramNormType = 0
	L2HysThreshold = 0.2
	gammaCorrection = 1
	nlevels = 64
	signedGradients = True
 
	hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,
	cellSize,nbins,derivAperture,
	winSigma,histogramNormType,L2HysThreshold,
	gammaCorrection,nlevels, signedGradients)


	# Calculate HOG vector for each test image
	i=0
	while i<n:
	  img = cv2.resize(test_images[i,:,:], (60, 60))
	  image_resized = np.array(img,dtype=np.uint8)
	  features[i,:] = hog.compute(image_resized).flatten()
	  i=i+1

	# load compressed pickle file that contains 
	# our trained classifier
	svc = decompress_pickle('trained_svc.pbz2') 
	# use the pretrained classifier to predict 
	# the labels of our test images
	predicted_labels = svc.predict(features)
	return predicted_labels

#################################################


# Load the input npy files here
test_images = np.load(test_img_array,allow_pickle=True)
true_values = np.load(test_labels,allow_pickle=True)

# Now we will take in the input images
# and test our classifier model
predicted_labels = test_func(test_images)


# Compare the output of our model to the true labels
correct_count = 0
for i in range(len(true_values)):      
  if(predicted_labels[i]==true_values[i]):
     correct_count += 1   
accuracy = (correct_count / len(true_values))*100 

print("percent correct using pretrained Support Vector Classifier:")
print(accuracy)

# save all the labels we predicted into an npy output array
np.save('predicted_labels.npy',predicted_labels)


