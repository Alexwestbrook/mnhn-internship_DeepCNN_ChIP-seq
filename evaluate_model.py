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
    output: Path to the output directory and file name
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
        "-data", "--data_part",
        help="data to evaluate the model on, train or test, default to test",
        default='test',
        type=str)
    parser.add_argument(
        "-b", "--batch_size",
        help="Number of samples to use for predicting in parallel, default to "
             "1024*16",
        default=1024*16,
        type=int)
    parser.add_argument(
        "-rel", "--relabeled",
        help="New labels for input dataset file with npz format.",
        type=str)
    args = parser.parse_args()
    # Check if the input data is valid
    if not os.path.isfile(args.dataset):
        sys.exit(f"{args.dataset} does not exist.\n"
                 "Please enter a valid dataset file path.")
    if args.data_part not in ['train', 'test']:
        sys.exit("data should be one of 'train' or 'test',"
                 f"{args.data_part} is invalid")
    # If the data was relabeled, check the new label file
    if args.relabeled:
        if not os.path.isfile(args.relabeled):
            sys.exit(f"{args.relabeled} does not exist.\n"
                     "Please enter a valid new labels file path.")
    return args


args = parsing()

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

model = tf.keras.models.load_model(args.trained_model)
# Do stuff with the model

# load the dataset
f1 = np.load(args.dataset)
if args.data_part == 'test':
    x_test = f1['x_test']
    if args.relabeled:
        f2 = np.load(args.relabeled)
        y_test = f2['y_test']
    else:
        y_test = f1['y_test']
    pred = model.predict(x_test, args.batch_size)
    np.savez(args.output,
             pred=pred)

    # compute accuracy per sample
    predIP = pred[y_test == 1]
    predControl = pred[y_test == 0]
    IP_accuracy = np.size(predIP[predIP > 0.5]) / np.size(predIP)
    Control_accuracy = (np.size(predControl[predControl < 0.5])
                        / np.size(predControl))
    print('accuracy: ', (IP_accuracy + Control_accuracy) / 2)
    print('IP accuracy: ', IP_accuracy)
    print('Control accuracy: ', Control_accuracy)
elif args.data_part == 'train':
    # load train and valid
    x_train = f1['x_train']
    x_valid = f1['x_valid']
    if args.relabeled:
        f2 = np.load(args.relabeled)
        y_train = f2['y_train']
        y_valid = f2['y_valid']
    else:
        y_train = f1['y_train']
        y_valid = f1['y_valid']
    # evaluate on train and valid
    pred_train = model.predict(x_train,
                               batch_size=args.batch_size)
    pred_valid = model.predict(x_valid,
                               batch_size=args.batch_size)
    np.savez(args.output,
             pred_train=pred_train,
             pred_valid=pred_valid)
    # predIP_train = pred_train[y_train == 1]
    # predControl_train = pred_train[y_train == 0]
    # predIP_valid = pred_valid[y_valid == 1]
    # predControl_valid = pred_valid[y_valid == 0]
