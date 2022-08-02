#!/usr/bin/env python

# This code takes a model and a dataset and trains a CNN model with similar
# architecture to Yann's with Dropout. The model is saved in a directory
#
# To execute this code run
# ModelYann.py -m <model_file> -d <dataset> -out <path/filename>
# with other options available
#
# parameters :
# - model_file : python file containing a build_model function to create the
#       model architecture to be used
# - dataset : npz file (numpy binary archive) containing 4 arrays named
#       x_train, y_train, x_valid, y_valid (training features, training labels,
#       validation features, validation labels)
# - path/filename : path of the directory to store the model in, and name of
#       the file to store the model in

import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, \
    CSVLogger
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.metrics import binary_crossentropy
import numpy as np
import os
import sys
import argparse
import time
import json
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
        help='name of the model architecture in "models.py".',
        type=str,
        required=True)
    parser.add_argument(
        "-d", "--dataset", help="Input dataset file with npz format.",
        type=str,
        required=True)
    parser.add_argument(
        "-out", "--output",
        help="Path to the output directory and file name.",
        type=str,
        required=True)
    parser.add_argument(
        "-rl", "--read_length",
        help="Number of base pairs in input reads, default to 101",
        default=101,
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
             "1024*16",
        default=1024*16,
        type=int)
    parser.add_argument(
        "-da", "--disable_autotune",
        action='store_true',
        help="indicates not to use earlystopping.")
    parser.add_argument(
        "-p", "--patience",
        help="Number of epochs without improvement to wait before stopping "
             "training, default to 10",
        default=10,
        type=int)
    parser.add_argument(
        "-ee", "--eval_epoch",
        action='store_true',
        help="indicates to evaluate train samples after each epoch.")
    parser.add_argument(
        "-dist", "--distribute",
        action='store_true',
        help="indicates to use both GPUs with MirrorStrategy.")
    parser.add_argument(
        "-rel", "--relabeled",
        help="New labels for input dataset file with npz format.",
        type=str)
    parser.add_argument(
        "-tm", "--train_method",
        help="Method to use for training, 0 for base, 1 for asymmetric "
             "reweighting of positive examples, 2 for experimental "
             "reweighting",
        default=0,
        type=int)
    parser.add_argument(
        "-t", "--temperature",
        help="Temperature to use for reweighting",
        default=1,
        type=int)
    parser.add_argument(
        "-start_tm", "--start_tm",
        help="training iteration from which to start to use the train method",
        default=2000,
        type=int)
    parser.add_argument(
        "-v", "--verbose",
        help="0 for silent, 1 for progress bar and 2 for single line",
        default=2,
        type=int)
    args = parser.parse_args()
    # Check if the input data is valid
    if not os.path.isfile(args.dataset):
        sys.exit(f"{args.dataset} does not exist.\n"
                 "Please enter a valid dataset file path.")
    # If the data was relabeled, check the new label file
    if args.relabeled:
        if not os.path.isfile(args.relabeled):
            sys.exit(f"{args.relabeled} does not exist.\n"
                     "Please enter a valid new labels file path.")
    return args


def train_reweighting_model(model,
                            x_train,
                            y_train,
                            x_valid,
                            y_valid,
                            batch_size,
                            epochs,
                            T=1,
                            train_method=1,
                            output=None,
                            eval_epoch=False,
                            x_test=None,
                            y_test=None,
                            autotune=True,
                            patience=10,
                            verbose=2):
    """ Deprecated."""
    # Create sample weights for first epoch for class balance
    if len(y_train) == 1:
        y_train = np.expand_dims(y_train, axis=1)
    sample_weights = utils.create_sample_weights(y_train)
    # Create generators for evaluating train, valid and test sets
    # ordered_generator_train = utils.DataGenerator(
    #     np.arange(len(y_train)),
    #     x_train,
    #     y_train,
    #     batch_size,
    #     shuffle=False)
    # generator_valid = utils.DataGenerator(
    #     np.arange(len(y_valid)),
    #     x_valid,
    #     y_valid,
    #     batch_size,
    #     shuffle=False)
    # Callbacks
    callbacks_list = [
        CSVLogger(os.path.join(output, "epoch_data.csv"), append=True)
    ]
    # Initialize sample weights and metrics logs
    if output is not None:
        preds_tr = []
        all_sample_weights = []
        if train_method == 2:
            delta_losses = []
        if eval_epoch:
            preds_ts = []
            # generator_test = utils.DataGenerator(
            #     np.arange(len(y_test)),
            #     x_test,
            #     y_test,
            #     batch_size,
            #     shuffle=False)
    # Initialize Early stopping
    if autotune:
        epochs_without_improvement = 0
        min_loss = np.inf
        best_weights = None
    # Training loop
    for epoch_id in range(epochs):
        print('\n\n\nStart epoch\n\n\n')
        t1 = time.time()
        # Apply earlystopping to maybe stop the training loop
        if autotune and epochs_without_improvement >= patience:
            epochs = epoch_id
            if verbose:
                print(f'Earlystopping at epoch {epochs}')
            break
        # Fit the model for one epoch
        print('\n\n\nStart training\n\n\n')
        t0 = time.time()
        history = model.fit(
            x_train,
            y_train,
            sample_weight=sample_weights,
            batch_size=batch_size,
            validation_data=(x_valid, y_valid),
            epochs=epoch_id+1,
            initial_epoch=epoch_id,
            callbacks=callbacks_list,
            verbose=verbose)
        print(f'\n\n\nEnd training, time={time.time()-t0}\n\n\n')
        # Change sample weights according to train loss
        print('\n\n\nStart predicting\n\n\n')
        t0 = time.time()
        pred_tr = model.predict(x_train,
                                verbose=verbose)
        print(f'\n\n\nEnd predicting, time={time.time()-t0}\n\n\n')
        print('\n\n\nStart post_processing\n\n\n')
        t0 = time.time()
        if train_method == 2:
            loss1 = binary_crossentropy(1, pred_tr)
            loss2 = binary_crossentropy(0, pred_tr)
            delta_loss = loss1 - loss2
            sample_weights = utils.change_sample_weights(delta_loss, T=T)
        else:
            loss1 = binary_crossentropy(y_train, pred_tr)
            # loss2 = binary_crossentropy(1 - y_train, pred_tr)
            # delta_loss = loss1 - loss2
            sample_weights = tf.where(
                tf.squeeze(y_train),
                utils.change_sample_weights(loss1, T=T),
                sample_weights
            )
        sample_weights = utils.balance_classes(sample_weights, y_train)
        print(f'\n\n\nEnd post_processing, time={time.time()-t0}\n\n\n')

        # Save train loss per sample, new labels and metrics on test
        if output is not None:
            preds_tr.append(pred_tr.ravel())
            all_sample_weights.append(sample_weights)
            if train_method == 2:
                delta_losses.append(delta_loss)
            np.save(
                os.path.join(output, 'sample_weights'),
                np.reshape(all_sample_weights, (-1, len(x_train)))
            )
            if eval_epoch:
                pred_ts = model.predict(
                    x_test,
                    verbose=verbose
                ).ravel()
                preds_ts.append(pred_ts)
        # Save best model weights according to valid loss
        if autotune:
            loss_valid = history.history['val_loss'][0]
            if min_loss > loss_valid:
                epochs_without_improvement = 0
                min_loss = loss_valid
                best_weights = model.get_weights()
            else:
                epochs_without_improvement += 1
        print(f'\n\n\nEnd epoch, time={time.time()-t1}\n\n\n')
    # Restore best model weights
    if autotune:
        model.set_weights(best_weights)
    # Save trained model
    model.save(os.path.join(output, "model"))
    # Save losses and metrics logs
    if output is not None:
        np.save(
            os.path.join(output, 'preds_tr'),
            np.reshape(preds_tr, (epochs, -1))
        )
        np.save(
            os.path.join(output, 'sample_weights'),
            np.reshape(all_sample_weights, (epochs, -1))
        )
        if train_method == 2:
            np.save(
                os.path.join(output, 'delta_losses'),
                np.reshape(delta_losses, (epochs, -1))
            )
        if eval_epoch:
            np.save(
                os.path.join(output, 'preds_ts'),
                np.reshape(preds_ts, (epochs, -1))
            )


if __name__ == "__main__":
    # Get arguments
    args = parsing()
    # Maybe build output directory
    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    # Store arguments in file
    with open(os.path.join(args.output, 'Experiment_info.txt'), 'w') as f:
        json.dump(vars(args), f, indent=4)
        f.write('\n')

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
    if args.distribute:
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model = models.build_model(args.architecture,
                                       read_length=args.read_length,
                                       learn_rate=args.learn_rate,
                                       method=args.train_method,
                                       T=args.temperature,
                                       start_reweighting=args.start_tm)
    else:
        model = models.build_model(args.architecture,
                                   read_length=args.read_length,
                                   learn_rate=args.learn_rate,
                                   method=args.train_method,
                                   T=args.temperature,
                                   start_reweighting=args.start_tm)

    # Load the dataset
    with np.load(args.dataset) as f1:
        x_train = f1['x_train']
        x_valid = f1['x_valid']
        if args.relabeled:
            with np.load(args.relabeled) as f2:
                y_train = f2['y_train']
                y_valid = f2['y_valid']
        else:
            y_train = f1['y_train']
            y_valid = f1['y_valid']
    if args.eval_epoch:
        with np.load(args.dataset) as f1:
            x_test = f1['x_test']
            if args.relabeled:
                with np.load(args.relabeled) as f2:
                    y_test = f2['y_test']
            else:
                y_test = f1['y_test']
    else:
        x_test = None
        y_test = None

    # Build generators for train, valid and test
    weights_train = utils.create_weights(y_train)
    generator_train = utils.DataGenerator(
        np.arange(len(y_train)),
        x_train,
        y_train,
        args.batch_size,
        weights_train)
    generator_valid = utils.DataGenerator(
        np.arange(len(y_valid)),
        x_valid,
        y_valid,
        args.batch_size,
        shuffle=False)
    # Create callbacks during training
    callbacks_list = [
        CSVLogger(os.path.join(args.output, "epoch_data.csv"))
        ]
    # Add optional autotune callbakcs
    if not args.disable_autotune:
        callbacks_list.append([
            ModelCheckpoint(filepath=os.path.join(args.output, "Checkpoint"),
                            monitor="val_accuracy",
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
    # Add optional callback for evaluating after epoch
    if args.eval_epoch:
        generator_test = utils.DataGenerator(
            np.arange(len(y_test)),
            x_test,
            y_test,
            args.batch_size,
            shuffle=False)
        callbacks_list.append(
            utils.Eval_after_epoch(args.output,
                                   generator_test))
    # Train model
    t0 = time.time()
    model.fit(generator_train,
              validation_data=generator_valid,
              epochs=args.epochs,
              callbacks=callbacks_list,
              verbose=args.verbose)
    train_time = time.time() - t0
    with open(os.path.join(args.output, 'Experiment_info.txt'), 'a') as f:
        f.write(f'training time: {train_time}\n')
    # Save trained model
    if args.train_method == 0:
        model.save(os.path.join(args.output, "model"))
    else:
        model.save_weights(os.path.join(args.output, "model_weights"))
