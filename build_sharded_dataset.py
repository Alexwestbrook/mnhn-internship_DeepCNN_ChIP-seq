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

from pathlib import Path
import sys
import argparse
import numpy as np
from itertools import zip_longest, chain, repeat, islice
from more_itertools import grouper, roundrobin
from Modules import utils


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
        "-ip", "--ip_files",
        help="raw sequencing IP fastq files",
        type=str,
        nargs='+',
        required=True)
    parser.add_argument(
        "-c", "--control_files",
        help="raw sequencing Control fastq files",
        type=str,
        nargs='+',
        required=True)
    parser.add_argument(
        "-out", "--out_dir",
        help="output dataset directory",
        type=str,
        required=True)
    parser.add_argument(
        "-rl", "--read_length",
        help="number of base pairs to encode reads on, optional",
        type=int)
    parser.add_argument(
        "-split", "--split_sizes",
        help="numbers of reads to put in the test and valid sets, "
             "must be of length 2, set to 0 to ignore a split, default to "
             "2**22 each.",
        default=[2**23, 2**23],
        type=int,
        nargs='+')
    parser.add_argument(
        "-shard", "--shard_size",
        help="maximum number of reads in s shard, default to 2**24.",
        default=2**24,
        type=int)
    parser.add_argument(
        "-kN", "--keepNs",
        help="indicates to only use fully sized reads",
        action="store_true")
    parser.add_argument(
        "-alt", "--alternate",
        help="maximum number of reads in s shard, default to 2**24.",
        action="store_true")
    args = parser.parse_args()
    # Check if the input data is valid
    for file in chain(args.ip_files, args.control_files):
        if not Path(file).exists():
            sys.exit(f"file {file} does not exist.\n"
                     "Please enter valid fastq file paths.")
    return args


def process_fastq_and_save(ip_files, control_files, out_dir, shard_size=2**24,
                           alternate=True, split_sizes=[2**23, 2**23],
                           read_length=None, keepNs=False):
    """
    Read multiple fastq files and convert them into a sharded numpy dataset.

    The reads in the fastq files are one-hot encoded and stored in shards.
    Each shard is a binary archive containing two arrays ids and one_hots:
    ids is an array of string ids in the fastq file and one_hots are the
    corresponding one-hot encoded sequences.

    The read length is infered from the maximum length in the first shard, all
    reads will be truncated or extended with N values to fit this length.

    Arguments
    ---------
    out_dir (str): name of the output dataset directory, must be empty
    fastq_files (list of str): list of fastq files to read from
    shard_size (int): number of reads in a shard
    alternate (bool): default value True indicates to take reads from each
    file alternatively, set to False to process files entirely before moving
    to the next.
    splits (tuple of int): number of test and valid samples in this order,
    remaining samples are train samples. Set value to 0 to ignore a split.
    read_length (int): number of bases in reads, if None, the read length is
    inferred from the maximum length in the first 100 sequences from each
    file. All reads will be truncated or extended with N values to fit this
    length.
    discardNs (bool): if True, reads with N values are discarded, as well as
    reads shorter than read_length.
    """
    # helper functions
    def save_shard():
        """One-hot encode a shard and save to npz archive"""
        print(f'saving shard {cur_split}_{cur_shard}...')
        one_hots = utils.one_hot_encoding(shard, read_length=read_length)
        np.savez_compressed(Path(out_dir, f'{cur_split}_{cur_shard}'),
                            ids=ids,
                            one_hots=one_hots)

    def get_split_iterators():
        """
        Return shard sizes for each split, in iterators.

        This is a generator function, generating an iterator per split, then
        an infinite iterator
        """
        for split_size in split_sizes:
            q, mod = divmod(split_size, shard_size)
            shard_sizes = [shard_size] * q
            if mod:
                shard_sizes.append(mod)
            yield iter(shard_sizes)
        yield repeat(shard_size)

    def build_file_iterator(files):
        file_iterator = []
        for file in files:
            # Iterate files by entries of 4 lines (fastq format)
            file_iterator.append(grouper(open(file), 4))
        if alternate:
            # Process files alternatively
            file_iterator = roundrobin(*file_iterator)
        else:
            # Process files one by one
            file_iterator = chain(*file_iterator)
        return file_iterator

    # Build output directory if needed
    Path(out_dir).mkdir(parents=True)

    # Infer read length from first 300 sequences in each file
    if read_length is None:
        print('read_length is unspecified, inferring read length from files')
        read_length = max(len(seq.rstrip())
                          for file in chain(ip_files, control_files)
                          for seq in islice(open(file), 1, 400, 4))
        print('read length is', read_length)

    # Handle file iteration
    ip_iter = build_file_iterator(ip_files)
    control_iter = build_file_iterator(control_files)

    # Handle train-valid-test splits
    splits = zip(['test', 'valid', 'train'], get_split_iterators())
    # Initialize first split and counters
    cur_split, cur_split_shards = next(splits)
    cur_shard_size = next(cur_split_shards)
    cur_shard = 0
    ids, shard = [], []
    # Read files
    for read in chain(*zip(ip_iter, control_iter)):
        # Get id and sequence
        id, seq, *_ = read
        if not keepNs and (len(seq.rstrip()) < read_length
                           or 'N' in seq):
            continue
        ids.append(id.rstrip())
        shard.append(seq.rstrip())
        # When shard is full, save it
        if len(shard) == cur_shard_size:
            save_shard()
            # Reinitialize
            ids, shard = [], []
            cur_shard += 1
            while True:
                try:
                    # Update next shard size
                    cur_shard_size = next(cur_split_shards)
                except StopIteration:
                    # Split is done, get to next split,
                    # loop again in case split is empty
                    cur_split, cur_split_shards = next(splits)
                    cur_shard = 0
                else:
                    # cur_shard_size was set successfully
                    break

    # Save last incomplete shard
    if len(shard) != 0:
        save_shard()


if __name__ == '__main__':
    args = parsing()
    process_fastq_and_save(**vars(args))
