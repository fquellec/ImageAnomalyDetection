import numpy as np
import argparse
import os
import models.CAE as cae
import utils.ImagesProcessor as ip

TRAINING_RAW_PATH = "training/raw"
TRAINING_PATH = "training/processed"
INPUT_RAW_PATH = "input/raw"
INPUT_PATH = "input/processed"
OUTPUT_PATH = "output"

EXT_IMAGES = ".jpg"

processImages = False
trainModel = True
processInputs = False

InputShape = np.array([300, 300, 3])
IP = ip.ImagesProcessor()
autoencoder = cae.CAE(InputShape)

if processImages:
	directory = os.fsencode(TRAINING_RAW_PATH)
	imgs = []
	for file in os.listdir(directory):
		filename = os.fsdecode(file)
		if filename.lower().endswith(EXT_IMAGES): 
			img = IP.readImage(TRAINING_RAW_PATH + "/" + filename)
			img = IP.resizeImage(img, InputShape[:-1])
			IP.writeImage(img, TRAINING_PATH + "/" + filename)


#train_x = IP.readImage("input/raw/test2.JPG")
#train_x = IP.resizeImage(train_x, InputShape[:-1])
#test_x = IP.readImage("training/raw/test2.JPG")
#test_x = IP.resizeImage(test_x, InputShape[:-1])

#autoencoder.createModel()
#autoencoder.train(np.expand_dims(train_x, axis=0), np.expand_dims(test_x, axis=0), epochs=300)
#autoencoder.save('my_model.h5')


if(trainModel):
	directory = os.fsencode(TRAINING_PATH)
	imgs = []
	for file in os.listdir(directory):
		filename = os.fsdecode(file)
		if filename.lower().endswith(EXT_IMAGES): 
			img = IP.readImage(TRAINING_PATH + "/" + filename)
			#img_slices = IP.sliceImage(img,(124,124),28)
			#imgs.append(img_slices)
			imgs.append(img)

	#normalize 
	imgs = np.array(imgs)/255
	imgs = np.reshape(imgs, (-1, InputShape[0],InputShape[1],InputShape[2]))

	autoencoder.createModel()
	autoencoder.train(imgs, imgs, epochs=1000)
	autoencoder.save('my_model.h5')
else:
	autoencoder.load('my_model.h5') 


if processInputs:
	directory = os.fsencode(INPUT_RAW_PATH)
	for file in os.listdir(directory):
		filename = os.fsdecode(file)
		if filename.lower().endswith(EXT_IMAGES): 
			img = IP.readImage(INPUT_RAW_PATH + "/" + filename)
			img = IP.resizeImage(img, InputShape[:-1])
			IP.writeImage(img, INPUT_PATH + "/" + filename)


# Read and predict inputs images
directory = os.fsencode(INPUT_PATH)
for file in os.listdir(directory):
	filename = os.fsdecode(file)
	if filename.lower().endswith(EXT_IMAGES): 
		img = IP.readImage(INPUT_PATH + "/" + filename)/255
		predict = autoencoder.predict(np.expand_dims(img, axis=0))
		mse = np.sum(np.power(np.subtract(img, predict), 2))/img.size
		print(filename, " - MSE = ", mse)
		IP.writeImage(np.squeeze(predict)*255, OUTPUT_PATH + "/" + filename)


