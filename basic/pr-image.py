#######################################################
## Simple file for predicting with Tensor Flow Neural Net
## Uses Keras
##
## Prediction images should be in a single directory
##
## byarbrough
## June 2019
########################################################
from tensorflow import keras
from os import path, listdir
from skimage import data, transform
import sys
import numpy as np

ROOT_PATH = ""		# modify path to data directory
IMG_DIM = 28				# DxD size of square image

def load_data(data_directory):
	"""
	Load images from given directory

	Args:
		data_directory (str): image or directory that holds images 

	Returns:
		images (list): list of images as data
		f_names (list): list of filenames
	"""
	# open directory if there is one
	images = []
	if path.isdir(data_directory):
		# open images
		f_names = listdir(data_directory)
		file_names = [path.join(data_directory, f)
			for f in f_names
			if f.endswith(".ppm")] # this is pretty strict
		for f in file_names:
			images.append(data.imread(f))
			
	elif data_directory.endswith(".ppm"): # single image
		f_names = data_directory
		images.append(data.imread(data_directory))

	return images, f_names


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

	# needs to be an array
	p_images = np.array(p_images)

	return p_images


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
	print('Opening', name)
	model = keras.models.load_model(name)
	print('Opened', name)
	print(model.summary)
	return model


def predict(p_images, f_names, model):
	"""
	Use the loaded model to predict images
	Returns index of highest likelihood category for each file
	Args:
		p_images (ndarray): processed images
		f_names (list): list of filenames
		model (keras model): loaded neural network
	Returns:
		results (2-D array): pairing of [filename, label_index] 
	"""
	predictions = model.predict(p_images)
	labels = predictions.argmax(axis=1)
	results = np.column_stack((f_names, labels))
	return results


def main():
	"""
	A simple demo of computer vision with Tensor Flow Keras
	Loads images, preprocess images, opens a model
	Uses model to predict labels of images

	Args
	"""
	# locate testing data
	if len(sys.argv) == 1:
		print("Argument required: prediction directory")
	if len(sys.argv) == 2:
		print("Argument required: model name")
	if (len(sys.argv) !=3):
		print("Wrong number of arguments")
		quit()
	pr_data_dir = path.join(ROOT_PATH, sys.argv[1])

	# load testing data
	print("Loading testing data from", pr_data_dir)
	images, f_names = load_data(pr_data_dir)
	print(len(images), 'images loaded')

	# preprocess images
	p_images  = preprocess(images)

	# open model
	model = open_model(sys.argv[2])

	# predict
	results = predict(p_images, f_names, model)
	print('Predictions:\n', results)


if __name__ == '__main__':
	main()