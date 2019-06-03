#######################################################
## Simple file for predicting with Tensor Flow Neural Net
## Uses Keras
##
## Testing images should be in a single directory
## Sub directories constitue the labels
##
## byarbrough
## May 2019
########################################################
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import model_from_yaml
from skimage import data, transform
from skimage.color import rgb2gray
from os import path, listdir
import sys
import numpy as np

ROOT_PATH = "/home/"		# modify path to data directory
IMG_DIM = 28				# DxD size of square image

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


def preprocess(images, labels):
	"""
	Standardize images so they are ready for training

	Args
		images (list): images to be processed

	Returns
		p_images (list): processed images
	"""

	# resize the image
	p_images = [transform.resize(image, (IMG_DIM, IMG_DIM)) for image in images]

	# needs to be an array
	p_images = np.array(p_images)

	# tf keras categorization of labels
	p_labels = tf.keras.utils.to_categorical(np.array(labels))

	return p_images, p_labels


def open_model(name):
	"""
	Opens and returns the previously trained model
	Require YAML and h5 files

	Args
		name (str): Name of model
	Return
		model (keras model): loaded neural network
	"""

	# structure of model
	model = keras.models.load_model(name)
	print(model.summary)
	return model


def test(images, labels, model):
	"""
	Tests the model against the testing set

	Args
		images (list): processed images
		labels (keras categorical): accompanying labels
		model (keras model): the loaded model
	Return
		result (list): loss and accuracy of evaluation
	"""
	result = model.evaluate(images, labels)
	return result


def main():
	"""
	A simple demo of computer vision with Tensor Flow Keras
	Loads images, preprocess images, opens a model
	Uses model to predict labels of images

	Args
	"""
	# locate testing data
	if len(sys.argv) == 1:
		print("Argument required: testing directory")
	if len(sys.argv) == 2:
		print("Argument required: model name")
	if (len(sys.argv) !=3):
		print("Wrong number of arguments")
		quit()
	test_data_dir = path.join(ROOT_PATH, sys.argv[1])

	# load testing data
	print("Loading testing data from", test_data_dir)
	images, labels = load_data(test_data_dir)
	print(len(images), 'images loaded')

	# preprocess images
	p_images, labels = preprocess(images, labels)

	# open model
	model = open_model(sys.argv[2])

	# test
	print('Evaluating model...')
	result = test(p_images, labels, model)
	print('test loss, test acc: ', result)


if __name__ == '__main__':
	main()