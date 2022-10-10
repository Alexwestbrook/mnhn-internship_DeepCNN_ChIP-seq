import tensorflow as tf
import numpy as np
import os
import sys
import argparse
from pathlib import Path
from Modules import tf_utils, models


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
        help="trained model file, or model weights file.",
        type=str,
        required=True)
    parser.add_argument(
        "-d", "--dataset",
        help="dataset file",
        type=str,
        required=True)
    parser.add_argument(
        "-ff", "--from_files",
        help="specifies that the dataset consists of multiple npz archives. "
        "In this case dataset must the name of the directory contaiing the "
        "files. Do not store other files in this directory or the data "
        "generator might try to read them as data.",
        action='store_true')
    parser.add_argument(
        "-out", "--output",
        help="Path to the output directory and file name.",
        type=str,
        required=True)
    parser.add_argument(
        "-tm", "--train_method",
        help="method used during training, 0 for base and 1 for reweighting, "
             "default to 0",
        default=0,
        type=int)
    parser.add_argument(
        "-rl", "--read_length",
        help="Number of base pairs in input reads, default to 101",
        default=101,
        type=int)
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
    parser.add_argument(
        "-arch", "--architecture",
        help='name of the model architecture in "models.py", required to load '
             'model from weights.',
        type=str)
    args = parser.parse_args()
    # Check if the input data is valid
    if args.from_files:
        if not Path(args.dataset).is_dir():
            sys.exit(f"When specifying from_files argument, dataset must be "
                     "directory, {args.dataset} isn't")
    else:
        if not Path(args.dataset).is_file():
            sys.exit(f"{args.dataset} does not exist.\n"
                     "Please enter a valid dataset file path.")
        # If the data was relabeled, check the new label file
        if args.relabeled:
            if not Path(args.relabeled).exists():
                sys.exit(f"{args.relabeled} does not exist.\n"
                         "Please enter a valid new labels file path.")
    if args.data_part not in ['train', 'test']:
        sys.exit("data should be one of 'train' or 'test',"
                 f"{args.data_part} is invalid")
    return args


if __name__ == "__main__":
    # Get arguments
    args = parsing()
    # Maybe build output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    # Limit gpu memory usage
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    # Load trained model
    if args.train_method == 0:
        model = tf.keras.models.load_model(args.trained_model)
    else:
        model = models.build_model(args.architecture,
                                   read_length=args.read_length,
                                   method=args.train_method)
        model.load_weights(args.trained_model)
    # Load dataset
    if args.data_part == 'test':  # Load testing samples
        if args.from_files:  # Load from a sharded dataset
            x_test = tf_utils.DataGeneratorFromFiles(
                args.dataset,
                args.batch_size,
                use_labels=args.relabeled,
                shuffle=False,
                split='test')
            if args.relabeled:
                y_test = np.load('labels_test_0.npz')
            else:
                y_test = None
        else:  # Load from a simple dataset
            f1 = np.load(args.dataset)
            # load test
            x_test = f1['x_test']
            if args.relabeled:
                f2 = np.load(args.relabeled)
                y_test = f2['y_test']
            else:
                y_test = f1['y_test']
        # Predict on test and save
        pred = model.predict(x_test, args.batch_size)
        np.savez(args.output,
                 pred=pred)
        # Compute accuracy per label
        # If labels aren't provided, assume alternating between 1 and 0
        # This is often the case in the sharded dataset
        if y_test is None:
            y_test = np.zeros(len(x_test), dtype=bool)
            y_test[::2] = 1
        predIP = pred[y_test == 1]
        predControl = pred[y_test == 0]
        IP_accuracy = np.size(predIP[predIP > 0.5]) / np.size(predIP)
        Control_accuracy = (np.size(predControl[predControl < 0.5])
                            / np.size(predControl))
        print('accuracy: ', (IP_accuracy + Control_accuracy) / 2)
        print('IP accuracy: ', IP_accuracy)
        print('Control accuracy: ', Control_accuracy)
    elif args.data_part == 'train':  # Load training samples
        if args.from_files:  # Load from a sharded dataset
            x_train = tf_utils.DataGeneratorFromFiles(
                args.dataset,
                args.batch_size,
                use_labels=args.relabeled,
                shuffle=False,
                split='train')
            x_valid = tf_utils.DataGeneratorFromFiles(
                args.dataset,
                args.batch_size,
                use_labels=args.relabeled,
                shuffle=False,
                split='valid')
        else:  # Load from a simple dataset
            f1 = np.load(args.dataset)
            # load train and valid
            x_train = f1['x_train']
            x_valid = f1['x_valid']
        # predict on train/valid and save
        pred_train = model.predict(x_train,
                                   batch_size=args.batch_size)
        pred_valid = model.predict(x_valid,
                                   batch_size=args.batch_size)
        np.savez(args.output,
                 pred_train=pred_train,
                 pred_valid=pred_valid)
