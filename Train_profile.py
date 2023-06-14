import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, \
    CSVLogger
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.metrics import binary_crossentropy
from tensorflow.keras.optimizers import Adam
import numpy as np
import sys
import argparse
import time
import datetime
import socket
import json
from Modules import utils, tf_utils, models
from Modules.tf_utils import mae_cor, correlate
from pathlib import Path
import tempfile


def parsing():
    """
    Parse the command-line arguments.

    Arguments
    ---------
    python command-line

    Returns
    -------
    dataset: Input dataset file with npz format
    output: Path to the output directory and file name
    read_length: Number of base pairs in input reads
    learning_rate: Value for learning rate
    epochs: Number of training loops over the dataset
    batch_size: Number of samples to use for training in parallel
    relabeled: if the dataset has been relabeled, this file contains the new
        labels and weights for the dataset
    """
    # Declaration of expexted arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-arch", "--architecture",
        help='Name of the model architecture in "models.py".',
        type=str,
        required=True)
    parser.add_argument(
        "-g", "--genome",
        help="one-hot encoded genome file in npz archive, with one array per "
        "chromosome",
        type=str,
        required=True)
    parser.add_argument(
        "-l", "--labels",
        help="label file in npz archive, with one array per chromosome",
        type=str,
        required=True)
    parser.add_argument(
        "-out", "--output",
        help="Path to the output directory and file name.",
        type=str,
        required=True)
    parser.add_argument(
        "-ct", "--chrom_train",
        help="Chromosomes to use for training",
        nargs='+',
        type=str,
        required=True)
    parser.add_argument(
        "-cv", "--chrom_valid",
        help="Chromosomes to use for validation",
        nargs='+',
        type=str,
        required=True)
    parser.add_argument(
        "-s", "--strand",
        help="strand to perform training on, choose between 'for', 'rev' or "
        "'both'. Default to 'both.",
        type=str,
        default='both')
    parser.add_argument(
        "-w", "--winsize",
        help="Size of the window in bp to use for prediction, default to 2001",
        default=2001,
        type=int)
    parser.add_argument(
        "-lr", "--learn_rate",
        help="Value for learning rate, default to 0.001",
        default=0.001,
        type=float)
    parser.add_argument(
        "-ep", "--epochs",
        help="Number of training loops over the dataset, default to 100",
        default=100,
        type=int)
    parser.add_argument(
        "-b", "--batch_size",
        help="Number of samples to use for training in parallel, default to "
             "1024",
        default=1024,
        type=int)
    parser.add_argument(
        "-ss", "--same_samples",
        help="Indicates to use the same sample at each epoch",
        action='store_true')
    parser.add_argument(
        "-mt", "--max_train",
        help="Maximum number of windows per epoch for training, default to "
        "2*22",
        default=2**22,
        type=int)
    parser.add_argument(
        "-mv", "--max_valid",
        help="Maximum number of windows per epoch for validation, default to "
        "2*20",
        default=2**20,
        type=int)
    parser.add_argument(
        "-bal", "--balance",
        help="'global' indicates to balance weights globally and 'batch' "
        "indicates to balance weights in each batch. If not set, no weights "
        "are used. Default to None",
        default=None,
        type=str)
    parser.add_argument(
        "-h_int", "--head_interval",
        help="Spacing between output head in case of mutliple outputs, "
             "default to None",
        default=None,
        type=int)
    parser.add_argument(
        "-da", "--disable_autotune",
        action='store_true',
        help="Indicates not to use earlystopping.")
    parser.add_argument(
        "-p", "--patience",
        help="Number of epochs without improvement to wait before stopping "
             "training, default to 10",
        default=6,
        type=int)
    parser.add_argument(
        "-dist", "--distribute",
        action='store_true',
        help="Indicates to use both GPUs with MirrorStrategy.")
    parser.add_argument(
        "-v", "--verbose",
        help="0 for silent, 1 for progress bar and 2 for single line",
        default=2,
        type=int)
    args = parser.parse_args()
    # Check if the input data is valid
    if not Path(args.genome).is_file():
        sys.exit(f"{args.genome} does not exist.\n"
                 "Please enter a valid genome file path.")
    if not Path(args.labels).is_file():
        sys.exit(f"{args.labels} does not exist.\n"
                 "Please enter a valid labels file path.")
    genome_name = Path(args.genome).stem
    if genome_name == 'W303':
        args.chrom_train = [
            'chr' + format(int(c), '02d') for c in args.chrom_train]
        args.chrom_valid = [
            'chr' + format(int(c), '02d') for c in args.chrom_valid]
    elif genome_name == 'mm10':
        args.chrom_train = [f'chr{c}' for c in args.chrom_train]
        args.chrom_valid = [f'chr{c}' for c in args.chrom_valid]
    with np.load(args.genome) as g:
        with np.load(args.labels) as s:
            for chr_id in args.chrom_train + args.chrom_valid:
                if not (chr_id in g.keys() and chr_id in s.keys()):
                    sys.exit(f"{chr_id} is not a valid chromosome id in "
                             f"{args.genome} and {args.labels}")
    return args


if __name__ == "__main__":
    tmstmp = datetime.datetime.now()
    # Get arguments
    args = parsing()
    # Maybe build output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)
    # Store arguments in file
    with open(Path(args.output, 'Experiment_info.txt'), 'w') as f:
        json.dump(vars(args), f, indent=4)
        f.write('\n')
        f.write(f'timestamp: {tmstmp}\n')
        f.write(f'machine: {socket.gethostname()}\n')

    # Limit gpu memory usage
    tf.debugging.set_log_device_placement(True)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    # Build model with chosen strategy
    model_dict = {
        'mnase_model': models.mnase_model,
        'mnase_Maxime': models.mnase_Maxime,
        'mnase_model_batchnorm': models.mnase_model_batchnorm,
        'mnase_Maxime_decreasing': models.mnase_Maxime_decreasing,
        'mnase_Maxime_increasing': models.mnase_Maxime_increasing,
        'mnase_Etienne': models.mnase_Etienne,
        'bassenji_Etienne': models.bassenji_Etienne
    }
    model_builder = model_dict[args.architecture]
    if args.distribute:
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model = model_builder(winsize=args.winsize)
            model.compile(optimizer=Adam(learning_rate=args.learn_rate),
                          loss=mae_cor,
                          metrics=["mae", correlate])
    else:
        model = model_builder(winsize=args.winsize)
        model.compile(optimizer=Adam(learning_rate=args.learn_rate),
                      loss=mae_cor,
                      metrics=["mae", correlate])

    # Load the data
    x_train = utils.merge_chroms(args.chrom_train, args.genome)
    x_valid = utils.merge_chroms(args.chrom_valid, args.genome)
    y_train = utils.merge_chroms(args.chrom_train, args.labels)
    y_valid = utils.merge_chroms(args.chrom_valid, args.labels)
    generator_train = tf_utils.WindowGenerator(
        data=x_train,
        labels=y_train,
        winsize=args.winsize,
        batch_size=args.batch_size,
        max_data=args.max_train,
        same_samples=args.same_samples,
        balance=args.balance,
        strand=args.strand,
        head_interval=args.head_interval)
    generator_valid = tf_utils.WindowGenerator(
        data=x_valid,
        labels=y_valid,
        winsize=args.winsize,
        batch_size=args.batch_size,
        max_data=args.max_valid,
        shuffle=False,
        same_samples=True,
        strand=args.strand,
        head_interval=args.head_interval)
    # Create callbacks during training
    callbacks_list = [
        CSVLogger(Path(args.output, "epoch_data.csv"))
        ]
    # Add optional autotune callbakcs
    if not args.disable_autotune:
        callbacks_list.append([
            ModelCheckpoint(filepath=Path(args.output, "Checkpoint"),
                            monitor="val_correlate",
                            save_best_only=True),
            EarlyStopping(monitor="val_loss",
                          patience=args.patience,
                          min_delta=1e-4,
                          restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss',
                              factor=0.1,
                              patience=args.patience//2,
                              min_lr=0.1*args.learn_rate),
        ])
    # Train model
    t0 = time.time()
    model.fit(generator_train,
              validation_data=generator_valid,
              epochs=args.epochs,
              callbacks=callbacks_list,
              verbose=args.verbose,
              shuffle=False)
    train_time = time.time() - t0
    with open(Path(args.output, 'Experiment_info.txt'), 'a') as f:
        f.write(f'training time: {train_time}\n')
    # Save trained model
    model.save(Path(args.output, "model"))
