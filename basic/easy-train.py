"""
Simple file for training image classification with Keras

Training images should be in a single directory
Sub directories constitue the labels

byarbrough
June 2019
"""
from keras import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from os import path
import sys

# Define defaults
EPOCHS = 16					# how many epochs
BATCH_SIZE = 32 			# size of a training batch
LEARN_RATE = 0.001 			# learning rate for optimizer
ROOT_PATH = ""				# modify path to data directory
IMG_DIM = 32				# dimension to resize image to
NUM_INTERNAL_LAYERS = 1		# number of coputational layers
INTERNAL_LAYER_SIZE = 128	# size of internal layer
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
		tr_set (DirectoryIterator): The loaded data and labels
	"""

	print("Loading training data from", tr_data_dir)
	# strucutre to load data
	tr_datagen = ImageDataGenerator(rescale=1./255, 
		shear_range=0.2, zoom_range=0.2, horizontal_flip=True) # helps with overfitting
	# load training data
	tr_set = tr_datagen.flow_from_directory(tr_data_dir,
		target_size=(IMG_DIM, IMG_DIM), batch_size=32, class_mode='categorical')

	return tr_set


def train(tr_set):
	"""
	Constructs and trains a CNN
	First layer takes in and resizes a color image
	Second layer does max pooling to reduce the number of features
	The next NUM_INTERNAL_LAYERS of size INTERNAL_LAYER_SIZE do the work
	The last layer does softmax to the number of classes.
	All layers but last use relu activation.
	Uses Adam optimzer.

	Args
		tr_set (DirectoryIterator): The loaded data and labels
	Returns
		model (Sequential): A compiled and trained model
	"""

	# math some constants
	num_classes = len(tr_set.class_indices)
	steps_per = tr_set.n // tr_set.batch_size

	# build a neural network
	model = Sequential()
	# first layer
	model.add(Convolution2D(32, 3, 3, input_shape=(IMG_DIM, IMG_DIM, 3), activation='relu'))
	# pool to reduce number of features
	model.add(MaxPooling2D(pool_size=(2,2)))
	# flatten into single vector
	model.add(Flatten())
	# internal layers
	[model.add(Dense(INTERNAL_LAYER_SIZE, activation='relu')) for i in range(NUM_INTERNAL_LAYERS)]
	# last layer
	model.add(Dense(num_classes, activation='softmax'))

	# compile
	model.compile(optimizer=Adam(lr=LEARN_RATE),
		loss='categorical_crossentropy',
		metrics=['accuracy'])
	
	# train.. lost of options to mess with in this function
	model.fit_generator(tr_set, steps_per_epoch=steps_per, epochs=EPOCHS)

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
	tr_set = load(tr_data_dir)

	# fit the model
	model = train(tr_set)

	# save the model
	save(model, FNAME)


if __name__ == '__main__':
	main()