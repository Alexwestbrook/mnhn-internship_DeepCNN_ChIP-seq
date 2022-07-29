import tensorflow as tf
import numpy as np
import os
import sys
import argparse
import Modules.utils as utils
import Modules.models as models


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
        "-g", "--genome_files",
        help="genome files to predict on",
        nargs="+",
        type=str,
        required=True)
    parser.add_argument(
        "-o", "--outputs",
        help="output files to write predictions in.",
        nargs="+",
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
        help="Number of base pairs in reads used for training, default to 101",
        default=101,
        type=int)
    parser.add_argument(
        "-b", "--batch_size",
        help="Number of samples to use for predicting in parallel, default to "
             "1024*16",
        default=1024*16,
        type=int)
    parser.add_argument(
        "-l", "--labels",
        help="labels to use for evaluating with npz format.",
        type=str)
    parser.add_argument(
        "-arch", "--architecture",
        help='name of the model architecture in "models.py", required to load '
             'model from weights.',
        type=str)
    args = parser.parse_args()
    # Check if the input data is valid
    if len(args.genome_files) != len(args.outputs):
        sys.exit("Please specify as many genome files as output files.")
    for genome_file in args.genome_files:
        if not os.path.isfile(genome_file):
            sys.exit(f"{genome_file} does not exist.\n"
                     "Please enter valid genome file paths.")
    # If the data was relabeled, check the new label file
    if args.labels:
        if not os.path.isfile(args.labels):
            sys.exit(f"{args.labels} does not exist.\n"
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

if args.train_method == 0:
    model = tf.keras.models.load_model(args.trained_model)
else:
    model = models.build_model(args.architecture,
                               read_length=args.read_length,
                               method=args.train_method)
    model.load_weights(args.trained_model)


for genome_file, output in zip(args.genome_files, args.outputs):
    indexes, data = utils.load_chr(genome_file, args.read_length)
    if args.labels:
        with np.load(args.labels) as f:
            labels = f['labels']
        if len(labels) != len(data):
            raise ValueError('labels must have same length as data')
    else:
        labels = np.zeros(len(data), dtype=bool)
    generator_chr = utils.DataGenerator(
        indexes,
        data,
        labels,
        args.batch_size,
        shuffle=False)
    pred = model.predict(generator_chr)
    np.savez(output, pred=pred)
