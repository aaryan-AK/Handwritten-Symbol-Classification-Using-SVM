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

# Pickle a file and then compress it into a file with extension 
def compressed_pickle(title, data):
	with bz2.BZ2File(title + '.pbz2', 'w') as f: 
		cPickle.dump(data, f)


IMS = np.load('./Train_Images.npy')
LABS = np.load('./Train_Labels.npy')

n = len(IMS)

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


features = np.zeros((n,1089))
print("begin HOG calculations")
i=0
while i<n:
	# HOG
    img = cv2.resize(IMS[i,:,:], (60, 60))
    image_resized = np.array(img,dtype=np.uint8)
    features[i,:] = hog.compute(image_resized).flatten()
    i=i+1

svc = svm.SVC(gamma=0.001 , C = 100.) 
# gamma and C are hyperparameters
print("now training the SVC model")
svc.fit(features , LABS)

compressed_pickle('trained_svc', svc)


