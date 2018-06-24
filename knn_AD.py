import numpy as np
import os
import models.CAE as cae
import utils.ImagesProcessor as ip
import tensorflow as tf
from keras import metrics
import models.CAE as cae
import utils.ImagesProcessor as ip
from operator import itemgetter

TRAINING_RAW_PATH = "training/raw"
TRAINING_PATH = "training/processed"
INPUT_RAW_PATH = "input/raw"
INPUT_PATH = "input/processed"
OUTPUT_PATH = "output"

EXT_IMAGES = ".jpg"

TrainAutoencoder = False
GetFeatures = False
ratioTrainTest = 0.8
slicesSize = np.array([28, 28, 3])
overlap = 10
inputShape = np.array([300, 300, 3])

IP = ip.ImagesProcessor()
autoencoder = cae.CAE(slicesSize,nbNeuronsLayers=[16, 8, 8], nbConvFilters=(3,3), poolScale=(2, 2))

def processImage(filename):
	img = IP.readImage(filename)
	img = IP.resizeImage(img, inputShape[:-1]).astype(np.float32)/255.0
	img_slices = IP.sliceImage(img,(slicesSize[0], slicesSize[1]),overlap)
	return img_slices

def euclidienne_distance(data1, data2):    
    return np.sum(np.power(np.array(data1) - np.array(data2), 2))
	

def get_neighbors(training_set, 
              data_point, 
              k, 
              distance=euclidienne_distance):
	distances = []
	for indexTrain in range(len(training_set)):
	    dist = distance(data_point, training_set[indexTrain])
	    distances.append(dist)
	distances.sort()
	neighbors = distances[:k]
	mean_distance = np.mean(neighbors)

	return mean_distance

if TrainAutoencoder:
	print("Autoencoder training")
	directory = os.fsencode(TRAINING_RAW_PATH)
	trainImgs = []
	for file in os.listdir(directory):
		filename = os.fsdecode(file)
		if filename.lower().endswith(EXT_IMAGES): 
			img_slices = processImage(TRAINING_RAW_PATH + "/" + filename)
			trainImgs.append(img_slices)
	trainImgs = np.array(trainImgs).reshape(-1, 28, 28, 3)

	autoencoder.createModel()
	# Randomly split the dataset into training and test subsets
	#np.random.shuffle(trainImgs)
	x_train = trainImgs[:int(ratioTrainTest*len(trainImgs))]
	x_test = trainImgs[int(ratioTrainTest*len(trainImgs)):]

	# Train the autoencoder and save the weights
	autoencoder.train(x_train, x_test, epochs=300, batch_size=124)
	autoencoder.save('model_knn.h5')
else :
	autoencoder.load('model_knn.h5') 

if GetFeatures:
	print("Reference Features extraction")
	directory = os.fsencode(TRAINING_RAW_PATH)
	features_ref = []
	for file in os.listdir(directory):
		filename = os.fsdecode(file)
		if filename.lower().endswith(EXT_IMAGES): 
			img_slices = processImage(TRAINING_RAW_PATH + "/" + filename)

			for img in img_slices:
				feature = np.squeeze(autoencoder.extractFeatures(np.expand_dims(img, axis=0)))
				features_ref.append(feature)

	features_ref = np.array(features_ref)
	np.save("features_ref", features_ref)
else:
	features_ref = np.load("features_ref.npy")


#neighbors = get_neighbors(features_ref, features_test, 4, distance=euclidienne_distance)
print("Inputs Processing")
directory = os.fsencode(INPUT_RAW_PATH)
inputDistances = []
for file in os.listdir(directory):
	filename = os.fsdecode(file)
	if filename.lower().endswith(EXT_IMAGES): 
		img_slices = processImage(INPUT_RAW_PATH + "/" + filename)

		features_test = []
		for img in img_slices:
			feature = np.squeeze(autoencoder.extractFeatures(np.expand_dims(img, axis=0)))
			distance = get_neighbors(features, features_ref, 1)
			features_test.append(feature)
		features_test = np.array(features_test)
		max_distance = np.max(features_test)
		print(filename + " - " + str(max_distance))
		inputDistances.append((filename, max_distance))

print("Sorting results")
inputDistances.sort(key=itemgetter(1))
for result in inputDistances: 
	print(result[0], " - Max Distance = ", result[1])





#print(neighbors)
#distances = []
#for img in neighbors.array:
	#distances.append((img[0], testFilenames[img[1]]))

#distances.sort(key=itemgetter(0))
#for distance in distances: 
#	print(distance[1], " - distance = ", distance[0])


