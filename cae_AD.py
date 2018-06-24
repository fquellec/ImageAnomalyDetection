import numpy as np
import argparse
import os
import models.CAE as cae
import utils.ImagesProcessor as ip
import tensorflow as tf
from keras import metrics
from operator import itemgetter

TRAINING_RAW_PATH = "training/raw"
TRAINING_PATH = "training/processed"
INPUT_RAW_PATH = "input/raw"
INPUT_PATH = "input/processed"
OUTPUT_PATH = "output"

EXT_IMAGES = ".jpg"

processImages = True
trainModel = True
processInputs = True


ratioTrainTest = 0.8
InputShape = np.array([300, 300, 3])
IP = ip.ImagesProcessor()
autoencoder = cae.CAE(InputShape,nbNeuronsLayers=[128, 64, 32], nbConvFilters=(3,3), poolScale=(2, 2))


if processImages:
	directory = os.fsencode(TRAINING_RAW_PATH)
	imgs = []
	for file in os.listdir(directory):
		filename = os.fsdecode(file)
		if filename.lower().endswith(EXT_IMAGES): 
			img = IP.readImage(TRAINING_RAW_PATH + "/" + filename)
			img = IP.resizeImage(img, InputShape[:-1])
			img = IP.extractChromaticity(img)
			IP.writeImage(img, TRAINING_PATH + "/" + filename)


if(trainModel):
	directory = os.fsencode(TRAINING_PATH)
	imgs = []
	for file in os.listdir(directory):
		filename = os.fsdecode(file)
		if filename.lower().endswith(EXT_IMAGES): 
			img = IP.readImage(TRAINING_PATH + "/" + filename).astype(np.float32)
			# normalize
			img /= 255.0
			print(img)
			imgs.append(img)

	imgs = np.array(imgs)
	imgs = np.reshape(imgs, (-1, InputShape[0],InputShape[1],InputShape[2]))

	# Randomly split the dataset into training and test subsets
	np.random.shuffle(imgs)
	x_train = imgs[:int(ratioTrainTest*len(imgs))]
	x_test = imgs[int(ratioTrainTest*len(imgs)):]

	autoencoder.createModel()
	autoencoder.train(x_train, x_test, epochs=300, batch_size=1)
	autoencoder.save('model_autoencoder.h5')
else:
	autoencoder.load('model_autoencoder.h5') 


if processInputs:
	directory = os.fsencode(INPUT_RAW_PATH)
	for file in os.listdir(directory):
		filename = os.fsdecode(file)
		if filename.lower().endswith(EXT_IMAGES): 
			print("Process: ", filename)
			img = IP.readImage(INPUT_RAW_PATH + "/" + filename)
			img = IP.resizeImage(img, InputShape[:-1])
			img = IP.extractChromaticity(img)
			IP.writeImage(img, INPUT_PATH + "/" + filename)


# Read and predict inputs images
results = []
directory = os.fsencode(INPUT_PATH)
for file in os.listdir(directory):
	filename = os.fsdecode(file)
	if filename.lower().endswith(EXT_IMAGES): 
		img = IP.readImage(INPUT_PATH + "/" + filename).astype(np.float32)
		#normalize
		img /= 255.0
		predict = np.squeeze(autoencoder.predict(np.expand_dims(img, axis=0)))

		tf_session = tf.Session()
		tensor_pred = tf.convert_to_tensor(predict, np.float32)
		tensor_valid = tf.convert_to_tensor(img, np.float32)
		#ce = metrics.binary_crossentropy(tensor_valid, tensor_pred)
		mse = metrics.mean_squared_error(tensor_valid, tensor_pred)
		mse = np.mean(mse.eval(session=tf_session))

		#print(filename, " - MSE = ", np.mean(mse.eval(session=tf_session)))

		results.append((filename, mse))
		predict = (predict*255).astype(np.uint8)
		IP.writeImage(predict, OUTPUT_PATH + "/" + filename)


# Sort and disply results
results.sort(key=itemgetter(1))
for result in results: 
	print(result[0], " - MSE = ", result[1])



