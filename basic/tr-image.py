"""
Simple file for training image classification with Keras

Training images should be in a single directory
Sub directories constitue the labels

byarbrough
June 2019
"""
from tensorflow.keras import Sequential, callbacks
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from os import path
import sys

# Define defaults
EPOCHS = 11					# how many epochs
BATCH_SIZE = 32 			# size of a training batch
LEARN_RATE = 0.001 			# learning rate for optimizer
ROOT_PATH = ""				# modify path to data directory
IMG_DIM = 32				# dimension to resize image to
FNAME = 'model'				# filename to save outputs as
OPTIMIZER = True			# include the optimizer when saving


def load(tr_data_dir):
	"""
	Uses keras ImageDataGenerator to load the data from directories.
	Uses keras flow_from_directory to read the iamges from subdirectories
	Each subdirectroy is a label.

	Args
		tr_data_dir (str): The directory that contains subdirectories
	Returns
		tr_gen (DirectoryIterator): The loaded data and labels
	"""

	print("Loading training data from", tr_data_dir)
	# strucutre to load data
	tr_datagen = ImageDataGenerator(rescale=1./255, 
		shear_range=0.2,
		zoom_range=0.2,
		horizontal_flip=False) # helps with overfitting
	# load training data
	tr_gen = tr_datagen.flow_from_directory(tr_data_dir,
		target_size=(IMG_DIM, IMG_DIM),
		batch_size=BATCH_SIZE,
		class_mode='categorical')

	return tr_gen


def train(tr_gen):
	"""
	Constructs and trains a CNN
	Several Convolution2D and MaxPooling2D layers,
	then one hidden dense layer.
	The last layer does softmax to the number of classes.
	All layers but last use relu activation.
	Uses Adam optimzer.

	Args
		tr_gen (DirectoryIterator): The loaded data and labels
	Returns
		model (Sequential): A compiled and trained model
	"""

	# math some constants
	num_classes = len(tr_gen.class_indices)

	# build a neural network
	model = Sequential()
	# first layer
	model.add(Convolution2D(filters=16, kernel_size=(3,3),
		input_shape=(IMG_DIM, IMG_DIM, 3), activation='relu'))
	# pool to reduce number of features
	model.add(MaxPooling2D(pool_size=2))
	# additional layers
	model.add(Convolution2D(32, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=2))
	model.add(Convolution2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=2))
	# flatten into single vector
	model.add(Flatten())
	# a hidden dense layer
	model.add(Dense(512))
	# last layer
	model.add(Dense(num_classes, activation='softmax'))

	# compile
	model.compile(optimizer=Adam(lr=LEARN_RATE),
		loss='categorical_crossentropy',
		metrics=['accuracy'])

	# train.. lost of options to mess with in this function
	model.fit_generator(tr_gen, epochs=EPOCHS)

	return model


def save(model, fname):
	"""
	Save the full model as a .h5 file

	Args
		fname (str): filename to save file as
		model (Sequential): The compiled and trained model to save
	"""
	# print a summary
	print(model.summary())

	# make sure there is not a filetype included
	fname = fname.split('.', 1)[0]
	
	# save model
	if OPTIMIZER:
		model.save(fname+'.h5', include_optimizer=True)
	else:
		model.save(fname+'.h5', include_optimizer=False)

	# confirm
	print('Model saved as ' + fname +'.h5')


def main():
	"""
	Load training data
	Build and train a CNN
	Save the model
	"""
	
	# locate training data
	if (len(sys.argv) != 2):
		print("Argument required: training directory")
		quit()
	tr_data_dir = path.join(ROOT_PATH, sys.argv[1])

	# load data
	tr_gen = load(tr_data_dir)

	# fit the model
	model = train(tr_gen)

	# save the model
	save(model, FNAME)


if __name__ == '__main__':
	main()