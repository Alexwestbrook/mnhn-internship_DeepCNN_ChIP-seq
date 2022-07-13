import tensorflow as tf
import numpy as np
import os
import sys
import argparse
import Modules.utils as utils


def parsing():
    """
    Parse the command-line arguments.

    Arguments
    ---------
    python command-line

    Returns
    -------
    trained_model : trained model saved by tf.keras.model.save
    dataset : dataset in npz format
    output : Path to the output directory and file name
    threshold : probability threshold from which to relabel
    """
    # Declaration of expexted arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--trained_model",
        help="trained model file.",
        type=str,
        required=True)
    parser.add_argument(
        "-d", "--dataset",
        help="dataset file",
        type=str,
        required=True)
    parser.add_argument(
        "-out", "--output",
        help="Path to the output directory and file name.",
        type=str,
        required=True)
    parser.add_argument(
        "-b", "--batch_size",
        help="Number of samples to use for predicting in parallel, default to "
             "1024*16",
        default=1024*16,
        type=int)
    parser.add_argument(
        "-t", "--threshold",
        help="probability threshold from which to relabel, default to 0.9.",
        default=0.9,
        type=float)
    args = parser.parse_args()
    # Check if the input data is valid
    if not os.path.isfile(args.dataset):
        sys.exit(f"{args.dataset} does not exist.\n"
                 "Please enter a valid dataset file path.")
    return args


args = parsing()

# load the dataset
with np.load(args.dataset) as f:
    x_train = f['x_train']
    y_train = f['y_train']
    x_valid = f['x_valid']
    y_valid = f['y_valid']
    x_test = f['x_test']
    y_test = f['y_test']

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

model = tf.keras.models.load_model(args.trained_model)
# Predict on train, valid and test data and relabel when IP prediction is
# below given threshold
pred_train = model.predict(x_train,
                           batch_size=args.batch_size).ravel()
mask_train = np.logical_and(pred_train > args.threshold,
                            y_train == 1)
y_train = np.where(mask_train, 1, 0)

pred_valid = model.predict(x_valid,
                           batch_size=args.batch_size).ravel()
mask_valid = np.logical_and(pred_valid > args.threshold,
                            y_valid == 1)
y_valid = np.where(mask_valid, 1, 0)

pred_test = model.predict(x_test,
                          batch_size=args.batch_size).ravel()
mask_test = np.logical_and(pred_test > args.threshold,
                           y_test == 1)
y_test = np.where(mask_test, 1, 0)
# save new labels
np.savez(args.output,
         y_train=y_train,
         y_valid=y_valid,
         y_test=y_test)
