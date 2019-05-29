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
from skimage import data
from os import path, listdir
import sys

# Define defaults
EPOCHS = 40 # how many times to train
BATCH_SIZE = 32 # size of a training batch
LEARN_RATE = 0.001 # learning rate for optimizer
ROOT_PATH = "/tmp/data/" # modify path to data directory

def load_data(data_directory):
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
			labels.append(int(d)) # this only works if the directy is a number

	return images, labels

def train():
	print("Training Complete")

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

	# reformat images

	# train model

	# save model


if __name__ == '__main__':
	main()