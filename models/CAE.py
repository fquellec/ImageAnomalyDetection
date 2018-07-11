from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras.callbacks import TensorBoard, EarlyStopping
from keras.models import load_model
import numpy as np
import tensorflow as tf
from keras import metrics

class CAE:
	def __init__(self, inputShapes, nbNeuronsLayers=[128, 64, 32], nbConvFilters=(3,3), poolScale=(2, 2)):
		# Model parameters
		self.inputShapes = inputShapes
		self.nbNeuronsLayers = nbNeuronsLayers
		self.nbConvFilters = nbConvFilters
		self.poolScale = poolScale

		# CAE objects
		self.autoencoder = None
		self.encoder = None

		# threshold for anomaly detection
		self.deltaError = 0

	def createModel(self):
		input_img = Input(shape=self.inputShapes)  # adapt this if using `channels_first` image data format

		x = Conv2D(self.nbNeuronsLayers[0], self.nbConvFilters, activation='sigmoid', padding='same')(input_img)
		x = MaxPooling2D(self.poolScale, padding='same')(x)
		x = Conv2D(self.nbNeuronsLayers[1], self.nbConvFilters, activation='sigmoid', padding='same')(x)
		x = MaxPooling2D(self.poolScale, padding='same')(x)
		x = Conv2D(self.nbNeuronsLayers[2], self.nbConvFilters, activation='sigmoid', padding='same')(x)
		encoded = MaxPooling2D(self.poolScale, padding='same')(x)

		# 

		x = Conv2D(self.nbNeuronsLayers[2], self.nbConvFilters, activation='sigmoid', padding='same')(encoded)
		x = UpSampling2D(self.poolScale)(x)
		x = Conv2D(self.nbNeuronsLayers[1], self.nbConvFilters, activation='sigmoid', padding='same')(x)
		x = UpSampling2D(self.poolScale)(x)
		x = Conv2D(self.nbNeuronsLayers[0], self.nbConvFilters, activation='sigmoid')(x)
		x = UpSampling2D(self.poolScale)(x)
		decoded = Conv2D(3, self.nbConvFilters, activation='sigmoid', padding='same')(x)

		self.autoencoder = Model(input_img, decoded)
		#print(self.autoencoder.summary())

		self.encoder = Model(input_img, encoded)
		#print(self.encoder.summary())

		#compile 
		self.autoencoder.compile(optimizer='adam', loss='mean_squared_error')

	def train(self, x_train, x_test, epochs=50, batch_size=128, shuffle=True):
		assert self.autoencoder is not None, "CAE not initiate, please call createModel() first"
		self.autoencoder.fit(x_train, x_train,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=shuffle,
                validation_data=(x_test, x_test),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder'), EarlyStopping(monitor='val_loss', min_delta=0, patience=5)])

		ref_dataset = np.append(x_train, x_test, axis=0)
		ref_predicts = self.autoencoder.predict(ref_dataset)
		errors = []
		for i in range(len(ref_predicts)):
			tf_session = tf.Session()
			tensor_pred = tf.convert_to_tensor(ref_predicts[i], np.float32)
			tensor_valid = tf.convert_to_tensor(ref_dataset[i], np.float32)
			mse = metrics.mean_squared_error(tensor_valid, tensor_pred)
			mse = np.mean(mse.eval(session=tf_session))
			errors.append(mse)
		errors = np.array(errors)
		print("errors_ref: ", np.sort(errors))
		self.deltaError = np.percentile(errors, 75)
		print("delta_error: ", self.deltaError)
	
	def predict(self, x):
		assert self.autoencoder is not None, "CAE not initiate, please call createModel() first"
		predicts =  self.autoencoder.predict(x)
	
		errors = []
		for i in range(len(predicts)):
			tf_session = tf.Session()
			tensor_pred = tf.convert_to_tensor(predicts[i], np.float32)
			tensor_valid = tf.convert_to_tensor(x[i], np.float32)
			mse = metrics.mean_squared_error(tensor_valid, tensor_pred)
			mse = np.mean(mse.eval(session=tf_session))
			if mse < self.deltaError:
				errors.append(1)
			else:
				errors.append(-1)
		errors = np.array(errors)

		return errors

	def extractFeatures(self, x):
		assert self.encoder is not None, "CAE not initiate, please call createModel() first"
		return self.encoder.predict(x)

	def save(self, filename):
		assert self.autoencoder is not None, "CAE not initiate, please call createModel() first"
		return self.autoencoder.save(filename), self.encoder.save("encoder_"+filename)

	def load(self, filename):
		self.autoencoder = load_model(filename)
		self.encoder = load_model("encoder_"+filename)

	



