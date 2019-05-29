#######################################################
## Simple file for training Tensor Flow Neural Net
## Uses Keras
##
## Training images should be in a single directory
## Sub directories constitue the labels
##
## byarbrough
## May 2019
########################################################
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, optimizers
from skimage import data, transform
from skimage.color import rgb2gray
from os import path, listdir
import sys
import numpy as np

# Define defaults
EPOCHS = 10					# how many times to train
BATCH_SIZE = 32 			# size of a training batch
LEARN_RATE = 0.001 			# learning rate for optimizer
ROOT_PATH = "/home/"	# modify path to data directory
IMG_DIM = 28				# DxD size of square image
NUM_INTERNAL_LAYERS = 1		# number of layers, excluding first and last
INT_LAYER_SIZE = -1			# size of internal layer, -1 for default

def load_data(data_directory):
	"""
	Load images from given directory

	Args:
		data_directory (str): directory that holds directories of images

	Returns:
		images (list): list of images
		lables (list); list of corresponding labesls
	"""
	# create list of all directories
	direct = [d for d in listdir(data_directory)
		if path.isdir(path.join(data_directory, d))]

	images = []
	labels = []

	# load from each directory
	for d in direct:
		label_direct = path.join(data_directory, d)
		file_names = [path.join(label_direct, f)
			for f in listdir(label_direct)
			if f.endswith(".ppm")] # this is pretty strict

		for f in file_names:
			images.append(data.imread(f))
			labels.append(int(d)) # this only works if the directory is a number

	return images, labels

def preprocess(images):
	"""
	Standardize images so they are ready for training

	Args
		images (list): images to be processed

	Returns
		p_images (list): processed images
	"""

	# resize the image
	p_images = [transform.resize(image, (IMG_DIM, IMG_DIM)) for image in images]

	# convert to grayscale and unravel
	p_images = [rgb2gray(np.array(image)).ravel() for image in p_images]

	# needs to be an array
	p_images = np.array(p_images)

	return p_images

def train(p_images, labels):
	"""
	Constructs and trains a dense neural network.
	First layer has as many nodes as processed image has pixels.
	The next NUM_INTERNAL_LAYERS default to the same number of nodes
	or are set to INT_LAYER_SIZE.
	The last layer has as many nodes as there are categories.
	Uses relu activation internally and softmax for last layer.
	Uses Adam optimzer.

	Args
		p_images (narray): images that have been preprocessed
		labels (categorical): labels of images in a one-hot matrix

	Returns
		model (keras.Sequential): a trained model

	"""

	pix = IMG_DIM * IMG_DIM
	int_size = INT_LAYER_SIZE
	if INT_LAYER_SIZE == -1:
		int_size = pix
	num_categories = len(labels[0])

	# first layer
	model = keras.Sequential()
	model.add(layers.Dense(pix, activation='relu'))
	# internal layers
	[model.add(layers.Dense(int_size, activation='reul')) for i in range(INT_LAYER_SIZE)]
	# last layer
	model.add(layers.Dense(num_categories, activation='softmax'))

	# configure
	model.compile(optimizer=keras.optimizers.Adam(lr=LEARN_RATE),
		loss='categorical_crossentropy',
		metrics=['accuracy'])
	
	# train
	model.fit(p_images, labels, batch_size=BATCH_SIZE, epochs=EPOCHS)

	return model

def save_model():
	print("Model saved: ")

def main():
	# locate training data
	if (len(sys.argv) != 2):
		print("Argument required: training directory")

	train_data_dir = path.join(ROOT_PATH, sys.argv[1])

	# load training data
	print("Loading training data from", train_data_dir)
	images, labels = load_data(train_data_dir)
	print(len(images), 'images loaded')

	# preprocess images
	p_images = preprocess(images)

	# tf keras categorization of labels
	labels = tf.keras.utils.to_categorical(np.array(labels))

	# train model
	print("Beginning training")
	model = train(p_images, labels)

	# save model


if __name__ == '__main__':
	main()