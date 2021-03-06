"""
Simple file for predicting with Tensor Flow Neural Net
Uses Keras

Prediction images should be in a single directory

byarbrough
July 2019
"""
from os import path
import sys
import numpy as np
from pandas import DataFrame
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

ROOT_PATH = ""		# modify path to data directory


def load_pr_data(pr_data_dir, img_dim):
    """
    Uses keras ImageDataGenerator to load the data from directories.
    Uses keras flow_from_directory to read the iamges from subdirectories
    Uses numpy to convert to arrays.
    Each subdirectroy is a label.

    Args
            pr_data_dir (str): The directory that contains subdirectories
            img_dim (int): Width of square input image. Must match first model layer
    Returns
            pr_imgs (Numpy array): The loaded images
            pr_labels (Numpy array): The loaded labels
    """

    print("Loading prediction data from", pr_data_dir)
    # strucutre to load data
    pr_datagen = ImageDataGenerator(rescale=1. / 255)
    # load prediction data
    pr_gen = pr_datagen.flow_from_directory(pr_data_dir, target_size=(
        img_dim, img_dim), class_mode='categorical', shuffle=False)

    return pr_gen


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


def predict(pr_gen, model):
    """
    Use the loaded model to predict images
    Returns index of highest likelihood category for each file

    Args:
            pr_gen (DirectoryIterator): processed images
            f_names (list): list of filenames
            model (keras model): loaded neural network
    Returns:
            predictions (Array): label for each element
    """
    # do prediction
    preds = model.predict_generator(pr_gen, verbose=1)

    # label index is highest likelihood
    pr_class_indices = np.argmax(preds, axis=-1)
    labels = pr_gen.class_indices
    # match everything up with python mumbojumbo
    labels = dict((v, k) for k, v in labels.items())
    predictions = [labels[k] for k in pr_class_indices]

    return predictions


def save_predictions(pr_gen, predictions):
    """
    Save the predictions to a csv
    Each prediction is paired with the filename

    Args
            pr_gen (DirectoryIterator): the prediction data
            predictions (array): labels for elements
    """
    # use pandas to match everything
    results = DataFrame({"Filename": pr_gen.filenames,
                         "Predictions": predictions})
    # write to csv
    output_file = 'preds.csv'
    results.to_csv(output_file, index=False)

    print('Results saved to', output_file)


def main():
    """
    A simple computer vision model for prediction with Keras.
    Loads images with flow_from_directory, opens a model,
    Uses model to predict and save image classes.
    """
    # check data and model args
    if len(sys.argv) == 1:
        print("Argument required: prediction directory")
    elif len(sys.argv) == 2:
        print("Argument required: model name")
    if len(sys.argv) != 3:
        print("Wrong number of arguments")
        quit()
    # set paths
    pr_data_dir = path.join(ROOT_PATH, sys.argv[1])
    model_location = path.join(ROOT_PATH, sys.argv[2])

    # open model
    model = open_model(model_location)
    img_dim = model.layers[0].input_shape[1]

    # load images
    pr_gen = load_pr_data(pr_data_dir, img_dim)

    # predict
    predictions = predict(pr_gen, model)

    # save
    save_predictions(pr_gen, predictions)


if __name__ == '__main__':
    main()
