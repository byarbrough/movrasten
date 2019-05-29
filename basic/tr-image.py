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
from os import path, listdir
import sys

# Define defaults
EPOCHS = 40 # how many times to train
BATCH_SIZE = 32 # size of a training batch
LEARN_RATE = 0.001 # learning rate for optimizer
ROOT_PATH = "" # modify path to data directory

def load_data(data_directory):
	images = []
	labels = []
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
	print("Loading training data from ", train_data_dir)
	images, labels = load_data(train_data_dir)
	print(len(images), 'images loaded')

	# reformat images

	# train model

	# save model


if __name__ == '__main__':
	main()