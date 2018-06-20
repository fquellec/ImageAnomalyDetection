from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras.callbacks import TensorBoard, EarlyStopping
from keras.models import load_model

class CAE:
	def __init__(self, inputShapes, nbNeuronsLayers=[32, 32, 32], nbConvFilters=(3,3), poolScale=(2, 2)):
		# Model parameters
		self.inputShapes = inputShapes
		self.nbNeuronsLayers = nbNeuronsLayers
		self.nbConvFilters = nbConvFilters
		self.poolScale = poolScale

		# CAE objects
		self.autoencoder = None
		self.encoder = None

	def createModel(self):
		input_img = Input(shape=self.inputShapes)  # adapt this if using `channels_first` image data format

		x = Conv2D(self.nbNeuronsLayers[0], self.nbConvFilters, activation='relu', padding='same')(input_img)
		x = MaxPooling2D(self.poolScale, padding='same')(x)
		x = Conv2D(self.nbNeuronsLayers[1], self.nbConvFilters, activation='relu', padding='same')(x)
		x = MaxPooling2D(self.poolScale, padding='same')(x)
		x = Conv2D(self.nbNeuronsLayers[2], self.nbConvFilters, activation='relu', padding='same')(x)
		encoded = MaxPooling2D(self.poolScale, padding='same')(x)

		# 

		x = Conv2D(self.nbNeuronsLayers[2], self.nbConvFilters, activation='relu', padding='same')(encoded)
		x = UpSampling2D(self.poolScale)(x)
		x = Conv2D(self.nbNeuronsLayers[1], self.nbConvFilters, activation='relu', padding='same')(x)
		x = UpSampling2D(self.poolScale)(x)
		x = Conv2D(self.nbNeuronsLayers[0], self.nbConvFilters, activation='relu')(x)
		x = UpSampling2D(self.poolScale)(x)
		decoded = Conv2D(3, self.nbConvFilters, activation='sigmoid', padding='same')(x)

		self.autoencoder = Model(input_img, decoded)
		print(self.autoencoder.summary())

		self.encoder = Model(input_img, encoded)
		print(self.encoder.summary())

		#compile 
		self.autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

	def train(self, x_train, x_test, epochs=50, batch_size=256, shuffle=True):
		assert self.autoencoder is not None, "CAE not initiate, please call createModel() first"
		self.autoencoder.fit(x_train, x_train,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=shuffle,
                validation_data=(x_test, x_test),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder'), EarlyStopping(monitor='val_loss', min_delta=0, patience=10)])
	
	def predict(self, x):
		assert self.autoencoder is not None, "CAE not initiate, please call createModel() first"
		return self.autoencoder.predict(x)

	def extractFeatures(self, x):
		assert self.autoencoder is not None, "CAE not initiate, please call createModel() first"
		return self.encoder.predict(x)

	def save(self, filename):
		assert self.autoencoder is not None, "CAE not initiate, please call createModel() first"
		return self.autoencoder.save(filename)

	def load(self, filename):
		self.autoencoder = load_model(filename)

	



