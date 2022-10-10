#!/usr/bin/env python

# This code takes an IP file of reads and a Control file of reads and creates
# a dataset separated into train, valid and test.
# The dataset in saved into an npz archive in the specified directory
#
# To execute this code run
# build_dataset.py -ip <IP file> -c <Control file> -out <path/filename>
# with other options available
#
# parameters :
# - IP file : npy file containing multiple read sequences from IP
# - Control file : npy file containing multiple read sequences from Control
# - path/filename : path of the directory to store the dataset in, and name of
#       the file to store the dataset in

import sys
import argparse
import numpy as np
from pathlib import Path
from Modules import utils

# from sklearn.preprocessing import OneHotEncoder
# from sklearn.utils import shuffle
# import tensorflow as tf
# import random


def parsing():
    """
    Parse the command-line arguments.

    Arguments
    ---------
    python command-line

    Returns
    -------
    IP : IP reads file with npz format
    Control : Control reads file with npz format
    output: Path to the output directory and file name
    max_size: maximum number of reads to take from IP or Control. The maximum
        size of the whole dataset is max_size*2
    one_hot_type: type to use for one hot encoding
    train_size: proportion of reads to put in the train set
    valid_size: proportion of reads to put in the valid set
    """
    # Declaration of expexted arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-ip", "--ip_file",
        help="IP reads file with npz format.",
        type=str,
        required=True)
    parser.add_argument(
        "-ctrl", "--control_file",
        help="Control reads file with npz format",
        type=str,
        required=True)
    parser.add_argument(
        "-out", "--output",
        help="Path to the output directory and file name.",
        type=str,
        required=True)
    parser.add_argument(
        "-bal", "--balanced",
        action='store_true',
        help="indicates that the dataset should be balanced.")
    parser.add_argument(
        "-tol", "--tolerance",
        help="maximum number of N accepted in reads, default to 0.",
        default=0,
        type=int)
    parser.add_argument(
        "-rl", "--read_length",
        help="length of the reads, default to 101.",
        default=101,
        type=int)
    parser.add_argument(
        "-full", "--full_size",
        action="store_true",
        help="indicates to only use fully sized reads")
    parser.add_argument(
        "-max", "--max_size",
        help="maximum number of reads to take from IP or Control, default to "
             "1e9.",
        default=int(1e9),
        type=int)
    parser.add_argument(
        "-oht", "--one_hot_type",
        help="type to use for one hot encoding, default to bool.",
        default=bool,
        type=type)
    parser.add_argument(
        "-t", "--train_size",
        help="proportion of reads to put in the train set, default to 0.7.",
        default=0.7,
        type=float)
    parser.add_argument(
        "-v", "--valid_size",
        help="proportion of reads to put in the valid set, default to 0.15.",
        default=0.15,
        type=float)  # test_size will be 1 - (train_size + valid_size)
    parser.add_argument(
        "-f", "--format",
        help="format of input data, default to npz.",
        default="npz",
        type=str)
    args = parser.parse_args()
    # Check if the input data is valid
    if not Path(args.ip_file).is_file():
        sys.exit(f"{args.ip_file} does not exist.\n"
                 "Please enter a valid ip file path.")
    if not Path(args.control_file).is_file():
        sys.exit(f"{args.control_file} does not exist.\n"
                 "Please enter a valid control file path.")
    if args.train_size + args.valid_size > 1:
        sys.exit("train_size and valid_size can't sum over 1")
    return args


def separate(reads, train_size, valid_size, max_size):
    """
    Separate reads in the file into train, valid and test set.
    Applies a shuffle to assign them randomly

    Arguments
    ---------
    reads: array containing the reads
    train_size: proportion of reads to put in the train set
    valid_size: proportion of reads to put in the valid set
    max_size: maximum size of reads to consider

    Returns
    -------
    training set of reads, validation set of reads, test set of reads, as lists
    """
    # Truncate the length if it goes over max_size
    length = min(len(reads), max_size)
    # create separators for slicing the dataset
    sep1 = round(train_size * length)
    sep2 = round((train_size + valid_size) * length)
    # shuffle the input files before separating into train, valid and test sets
    np.random.shuffle(reads)
    return (reads[:sep1].tolist(),
            reads[sep1:sep2].tolist(),
            reads[sep2:length].tolist())


# get arguments
args = parsing()

# load numpy binary file
if args.format == 'npz':
    with np.load(args.ip_file) as f:
        IP_reads = f['reads']
        np.random.shuffle(IP_reads)
        if args.full_size:
            IP_reads = utils.remove_reads_with_N(
                IP_reads,
                args.tolerance,
                max_size=args.max_size,
                read_length=args.read_length)
        else:
            IP_reads = utils.remove_reads_with_N(
                IP_reads,
                args.tolerance,
                max_size=args.max_size)
    with np.load(args.control_file) as f:
        Control_reads = f['reads']
        np.random.shuffle(Control_reads)
        if args.full_size:
            Control_reads = utils.remove_reads_with_N(
                Control_reads,
                args.tolerance,
                max_size=args.max_size,
                read_length=args.read_length)
        else:
            Control_reads = utils.remove_reads_with_N(
                Control_reads,
                args.tolerance,
                max_size=args.max_size)
elif args.format == 'npy':
    IP_reads = np.load(args.ip_file)
    Control_reads = np.load(args.control_file)

# if dataset must be balanced, set the max size to the shortest file length
if args.balanced:
    max_size = min(np.shape(IP_reads)[0],
                   np.shape(Control_reads)[0],
                   args.max_size)

np.random.seed(0)
# Create train, valid and test sets
IP_train, IP_valid, IP_test = separate(IP_reads,
                                       args.train_size,
                                       args.valid_size,
                                       max_size)
del IP_reads
Control_train, Control_valid, Control_test = separate(Control_reads,
                                                      args.train_size,
                                                      args.valid_size,
                                                      max_size)
del Control_reads

# Concatenate IP and Control in one array
x_train = IP_train + Control_train
x_valid = IP_valid + Control_valid
x_test = IP_test + Control_test

# label the sets
y_train = np.concatenate((np.ones(len(IP_train), dtype=bool),
                          np.zeros(len(Control_train), dtype=bool)))
y_valid = np.concatenate((np.ones(len(IP_valid), dtype=bool),
                          np.zeros(len(Control_valid), dtype=bool)))
y_test = np.concatenate((np.ones(len(IP_test), dtype=bool),
                         np.zeros(len(Control_test), dtype=bool)))

# shuffle train, valid and test sets along with their labels
order_train = np.random.permutation(len(x_train))
x_train = [x_train[i] for i in order_train]
y_train = y_train[order_train]

order_valid = np.random.permutation(len(x_valid))
x_valid = [x_valid[i] for i in order_valid]
y_valid = y_valid[order_valid]

order_test = np.random.permutation(len(x_test))
x_test = [x_test[i] for i in order_test]
y_test = y_test[order_test]

# one-hot-encoding
x_train = utils.one_hot_encoding(x_train,
                                 read_length=args.read_length,
                                 one_hot_type=args.one_hot_type)
x_valid = utils.one_hot_encoding(x_valid,
                                 read_length=args.read_length,
                                 one_hot_type=args.one_hot_type)
x_test = utils.one_hot_encoding(x_test,
                                read_length=args.read_length,
                                one_hot_type=args.one_hot_type)

# save the dataset into a numpy binary file
np.savez_compressed(args.output,
                    x_train=x_train,
                    y_train=y_train,
                    x_valid=x_valid,
                    y_valid=y_valid,
                    x_test=x_test,
                    y_test=y_test)
