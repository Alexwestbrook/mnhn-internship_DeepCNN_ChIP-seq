import tensorflow as tf
import numpy as np
import sys
import argparse
import warnings
from pathlib import Path
from Modules import utils, tf_utils, models


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
        "-g", "--genome",
        help="one-hot encoded genome file in npz format with an array per "
             "chromosome.",
        type=str,
        required=True)
    parser.add_argument(
        "-o", "--output",
        help="output directory to write predictions in.",
        type=str,
        required=True)
    parser.add_argument(
        "-c", "--chromosomes",
        help="chromosomes to predict on. Specify 'all' if you wish to predict "
             "on all chromosomes",
        nargs="+",
        default=["all"],
        type=str)
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
    parser.add_argument(
        "--multi_file",
        action='store_true',
        help='indicates to store prediction on each chromosome separately'
    )
    args = parser.parse_args()
    # Check if the input data is valid
    if not Path(args.genome).is_file():
        sys.exit(f"{args.genome} does not exist.\n"
                 "Please enter valid genome file path.")
    # If the data was relabeled, check the new label file
    if args.labels:
        if not Path(args.labels).is_file():
            sys.exit(f"{args.labels} does not exist.\n"
                     "Please enter a valid labels file path.")
    return args


if __name__ == "__main__":
    # Get arguments
    args = parsing()
    # Maybe create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)
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
    if not args.multi_file:
        all_preds = {}
    # Load genome
    with np.load(Path(args.genome)) as genome:
        print(args.chromosomes)
        if args.chromosomes == ["all"]:
            chromosomes = genome.keys()
        else:
            chromosomes = args.chromosomes
        print(chromosomes)
        for chr_id in chromosomes:
            # Load genomic data and maybe labels (labels aren't currently used)
            try:
                one_hot_chr = genome[chr_id]
            except KeyError:
                warnings.warn(Warning(
                    f"{chr_id} is not a valid chromosome ID in {args.genome}, "
                    "skipping..."))
                continue
            indexes, data = utils.chunk_chr(
                one_hot_chr,
                args.read_length)
            if args.labels:
                with np.load(args.labels) as f:
                    labels = f['labels']
                if len(labels) != len(data):
                    raise ValueError('labels must have same length as data')
            else:
                labels = np.zeros(len(data), dtype=bool)
            # build a generator
            generator_chr = tf_utils.DataGenerator(
                indexes,
                data,
                labels,
                args.batch_size,
                shuffle=False)
            # predict on data and save predictions
            preds = model.predict(generator_chr).ravel()
            if args.multi_file:
                np.savez_compressed(Path(args.output, f"preds_on_{chr_id}"),
                                    preds=preds)
            else:
                all_preds[chr_id] = preds
        if not args.multi_file:
            np.savez_compressed(Path(args.output,
                                     f"preds_on_{Path(args.genome).name}"),
                                **all_preds)
