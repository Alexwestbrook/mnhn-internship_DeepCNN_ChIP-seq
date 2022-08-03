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
        "-g", "--genome_dir",
        help="genome directory.",
        type=str,
        required=True)
    parser.add_argument(
        "-c", "--chromosomes",
        help="chromosomes to predict on.",
        nargs="+",
        type=str,
        required=True)
    parser.add_argument(
        "-o", "--output",
        help="output directory to write predictions in.",
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
    for chr_id in args.chromosomes:
        if not os.path.isfile(os.path.join(args.genome_dir,
                                           f'chr{chr_id}.npz')):
            sys.exit(f"chr{chr_id}.npz does not exist.\n"
                     "Please enter valid genome file paths.")
    # If the data was relabeled, check the new label file
    if args.labels:
        if not os.path.isfile(args.labels):
            sys.exit(f"{args.labels} does not exist.\n"
                     "Please enter a valid new labels file path.")
    return args


if __name__ == "__main__":
    # Get arguments
    args = parsing()
    # Maybe create output directory
    if not os.path.isdir(args.output):
        os.makedirs(args.output)
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
    # Initialize predictions
    all_preds = {}
    # Load genome
    for chr_id in args.chromosomes:
        # Load genomic data and maybe labels (labels aren't currently used)
        indexes, data = utils.load_chr(
            os.path.join(args.genome_dir, f'chr{chr_id}.npz'),
            args.read_length)
        if args.labels:
            with np.load(args.labels) as f:
                labels = f['labels']
            if len(labels) != len(data):
                raise ValueError('labels must have same length as data')
        else:
            labels = np.zeros(len(data), dtype=bool)
        # build a generator
        generator_chr = utils.DataGenerator(
            indexes,
            data,
            labels,
            args.batch_size,
            shuffle=False)
        # predict on data and save predictions
        all_preds[f"chr{chr_id}"] = model.predict(generator_chr).ravel()
np.savez_compressed(os.path.join(args.output, f"preds_on_genome"),
                    **all_preds)
