"""
Simple file for testing image classifcation with Keras

Testing images should be in a single directory
Sub directories constitue the labels

byarbrough
June 2019
"""
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from os import path
import sys

ROOT_PATH = ""		# modify path to data directory


def load_ts_data(ts_data_dir, img_dim):
	"""
	Uses keras ImageDataGenerator to load the data from directories.
	Uses keras flow_from_directory to read the iamges from subdirectories
	Uses numpy to convert to arrays.
	Each subdirectroy is a label.

	Args
		ts_data_dir (str): The directory that contains subdirectories
		img_dim (int): Width of square input image. Must match first model layer
	Returns
		ts_imgs (Numpy array): The loaded images
		ts_labels (Numpy array): The loaded labels
	"""

	print("Loading testing data from", ts_data_dir)
	# strucutre to load data
	ts_datagen = ImageDataGenerator(rescale=1./255)
	# load testing data
	ts_gen = ts_datagen.flow_from_directory(ts_data_dir,
		target_size=(img_dim, img_dim), class_mode='categorical')

	return ts_gen


def open_model(name):
	"""
	Opens and returns the previously trained model.
	Requires full h5 file.

	Args
		name (str): Name of model
	Returns
		model (Sequential): Loaded neural network
	"""
	# simple check for filetype
	if not name.lower().endswith('.h5'):
		print('Error, requires .h5 model')
		quit()

	# structure of model
	model = load_model(name)
	print(model.summary())
	return model


def main():
	"""
	A simple computer vision model evaluation with Keras.
	Loads images with flow_from_directory, opens a model,
	Uses model to predict and compare image classes.
	"""
	# check data and model args
	if len(sys.argv) == 1:
		print("Argument required: testing directory")
	elif len(sys.argv) == 2:
		print("Argument required: model name")
	if (len(sys.argv) !=3):
		print("Wrong number of arguments")
		quit()
	# set paths
	ts_data_dir = path.join(ROOT_PATH, sys.argv[1])
	model_location = path.join(ROOT_PATH, sys.argv[2])

	# open model
	model = open_model(model_location)
	img_dim = model.layers[0].input_shape[1]

	# load testing data
	ts_gen = load_ts_data(ts_data_dir, img_dim)

	# test
	print('Evaluating model...')
	result = model.evaluate_generator(ts_gen, steps=ts_gen.n, verbose=1)
	print('test loss, test acc: ', result)


if __name__ == '__main__':
	main()