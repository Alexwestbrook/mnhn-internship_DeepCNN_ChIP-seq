#!/usr/bin/env python
import os
from pathlib import Path
from collections import defaultdict
from typing import Optional, Union
import re
import json

import numpy as np
from numpy.lib.stride_tricks import as_strided
from numpy.core.numeric import normalize_axis_tuple
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
import scipy
from scipy.signal import gaussian, convolve
from scipy.sparse import coo_matrix

from statsmodels.stats import multitest

import pyBigWig
import pysam


# Constants
hg38_chr_ids = {
    1: 'NC_000001.11',
    2: 'NC_000002.12',
    3: 'NC_000003.12',
    4: 'NC_000004.12',
    5: 'NC_000005.10',
    6: 'NC_000006.12',
    7: 'NC_000007.14',
    8: 'NC_000008.11',
    9: 'NC_000009.12',
    10: 'NC_000010.11',
    11: 'NC_000011.10',
    12: 'NC_000012.12',
    13: 'NC_000013.11',
    14: 'NC_000014.9',
    15: 'NC_000015.10',
    16: 'NC_000016.10',
    17: 'NC_000017.11',
    18: 'NC_000018.10',
    19: 'NC_000019.10',
    20: 'NC_000020.11',
    21: 'NC_000021.9',
    22: 'NC_000022.11',
    'X': 'NC_000023.11',
    'Y': 'NC_000024.10'}
T2T_chr_ids = {
    '1': 'NC_060925.1',
    '2': 'NC_060926.1',
    '3': 'NC_060927.1',
    '4': 'NC_060928.1',
    '5': 'NC_060929.1',
    '6': 'NC_060930.1',
    '7': 'NC_060931.1',
    '8': 'NC_060932.1',
    '9': 'NC_060933.1',
    '10': 'NC_060934.1',
    '11': 'NC_060935.1',
    '12': 'NC_060936.1',
    '13': 'NC_060937.1',
    '14': 'NC_060938.1',
    '15': 'NC_060939.1',
    '16': 'NC_060940.1',
    '17': 'NC_060941.1',
    '18': 'NC_060942.1',
    '19': 'NC_060943.1',
    '20': 'NC_060944.1',
    '21': 'NC_060945.1',
    '22': 'NC_060946.1',
    'X': 'NC_060947.1',
    'Y': 'NC_060948.1', }
GRCh38_header = [
    ("chr1", 248956422),
    ("chr2", 242193529),
    ("chr3", 198295559),
    ("chr4", 190214555),
    ("chr5", 181538259),
    ("chr6", 170805979),
    ("chr7", 159345973),
    ("chr8", 145138636),
    ("chr9", 138394717),
    ("chr10", 133797422),
    ("chr11", 135086622),
    ("chr12", 133275309),
    ("chr13", 114364328),
    ("chr14", 107043718),
    ("chr15", 101991189),
    ("chr16", 90338345),
    ("chr17", 83257441),
    ("chr18", 80373285),
    ("chr19", 58617616),
    ("chr20", 64444167),
    ("chr21", 46709983),
    ("chr22", 50818468),
    ("chrX", 156040895),
    ("chrY", 57227415)]
GRCh38_lengths = dict(GRCh38_header)
T2T_header = [
    ("chr1", 248387328),
    ("chr2", 242696752),
    ("chr3", 201105948),
    ("chr4", 193574945),
    ("chr5", 182045439),
    ("chr6", 172126628),
    ("chr7", 160567428),
    ("chr8", 146259331),
    ("chr9", 150617247),
    ("chr10", 134758134),
    ("chr11", 135127769),
    ("chr12", 133324548),
    ("chr13", 113566686),
    ("chr14", 101161492),
    ("chr15", 99753195),
    ("chr16", 96330374),
    ("chr17", 84276897),
    ("chr18", 80542538),
    ("chr19", 61707364),
    ("chr20", 66210255),
    ("chr21", 45090682),
    ("chr22", 51324926),
    ("chrX", 154259566),
    ("chrY", 62460029)]
T2T_lengths = dict(T2T_header)


def data_generation(IDs, reads, labels, class_weights):
    X = np.empty((len(IDs), *reads[0].shape), dtype='bool')
    Y = np.empty((len(IDs), 1), dtype='bool')
    weights = np.empty((len(IDs), 1), dtype='float')
    for i, ID in enumerate(IDs):
        X[i, ] = reads[ID]
        Y[i] = labels[ID]
        weights[i] = class_weights[labels[ID]]
    return X, Y, weights


def data_generator(dataset_dir, batch_size, class_weights={0: 1, 1: 1},
                   shuffle=True, split='train', relabeled=False, cache=True):
    files = list(Path(dataset_dir).glob(split + '_*'))

    first_loop = True
    new_files = []
    while True:
        if shuffle:
            np.random.shuffle(files)
        for file in files:
            if first_loop:
                with np.load(file) as f:
                    x = f['one_hots']
            else:
                x = np.load(file)
            if relabeled:
                label_file = Path(file.parent, 'labels_' + file.name)
                with np.load(label_file) as f:
                    labels = f['labels']
            else:
                labels = np.zeros(len(x), dtype=bool)
                labels[::2] = 1

            indexes = np.arange(len(x))
            list_IDs = indexes

            n_batch = int(np.ceil(len(list_IDs) / batch_size))
            if shuffle:
                np.random.shuffle(indexes)

            for index in range(n_batch):
                start_batch = index * batch_size
                end_batch = (index + 1) * batch_size
                indexes_batch = indexes[start_batch:end_batch]
                list_IDs_batch = [list_IDs[k] for k in indexes_batch]
                yield data_generation(list_IDs_batch, x, labels,
                                      class_weights)
            if first_loop:
                new_file = Path(file.parent, file.stem + '.npy')
                new_files.append(new_file)
                np.save(new_file, x)
        if first_loop:
            files = new_files
        first_loop = False


# Data loader
def load_chr(chr_file, window_size, remove_Ns=False):
    """
    Load all sliding windows of a chromosome
    """
    with np.load(chr_file) as f:
        one_hot_chr = f['one_hot_genome']
    return chunk_chr(one_hot_chr, window_size, remove_Ns=remove_Ns)


def chunk_chr(one_hot_chr, window_size, remove_Ns=False):
    """
    Load all sliding windows of a chromosome
    """
    sliding_window = sliding_window_view(
        one_hot_chr,
        (window_size, 4),
        axis=(0, 1))
    data = sliding_window.reshape(sliding_window.shape[0],
                                  sliding_window.shape[2],
                                  sliding_window.shape[3])
    if remove_Ns:
        indexes = remove_windows_with_N(one_hot_chr, window_size)
        data = data[indexes]
    else:
        indexes = np.arange(len(data))
    return indexes, data


# Sample weight handling
def create_weights(y):
    """Return weights to balance negative and positive classes.

    Overall sum of weights is maintained equal.

    Parameters
    ----------
    y : ndarray
        1D-array of labels to weight, labels must be 0 and 1

    Returns
    -------
    dict[label] -> weight
        dictonary mapping label to class weight

    See also
    --------
    create_sample_weights : return weights as an array
    """
    n_pos = len(y[y == 1])
    n_neg = len(y[y == 0])
    pos_weight = 1/n_pos * (n_pos+n_neg)/2
    neg_weight = 1/n_neg * (n_pos+n_neg)/2
    return {0: neg_weight, 1: pos_weight}


def create_sample_weights(y):
    """Return sample weights to balance negative and positive classes.

    Analog to `create_weights` returning an array of sample weights.

    Parameters
    ----------
    y : ndarray, shape=(n,)
        1D-array of labels to weight, labels must be 0 and 1

    Returns
    -------
    ndarray, shape=(n,)
        1D-array of weight values for each element of `y`

    See also
    --------
    create_weights : return weights as a dictionary of class weights

    Notes
    -----
    Calls `create_weights`
    """
    weights = create_weights(y)
    sample_weights = np.where(np.squeeze(y) == 1,
                              weights[1],
                              weights[0])
    return sample_weights


# One-hot encoding and decoding
def one_hot_encode(seq, read_length=101, one_hot_type=bool):
    one_hot = np.zeros((read_length, 4), dtype=one_hot_type)
    for i, base in enumerate(seq):
        if i >= read_length:
            break
        if base == 'A':
            one_hot[i, 0] = 1
        elif base == 'C':
            one_hot[i, 1] = 1
        elif base == 'G':
            one_hot[i, 2] = 1
        elif base == 'T':
            one_hot[i, 3] = 1
    return one_hot


def one_hot_decode(one_hot, read_length=101, one_hot_type=bool):
    if len(one_hot.shape) == 2:
        read_length, n_bases = one_hot.shape
    else:
        raise ValueError(
            'input must be a single one hot encoded read with ')
    categories = np.array([['A'], ['C'], ['G'], ['T']])
    encoder = OneHotEncoder(dtype=bool,
                            handle_unknown='ignore',
                            sparse=False)
    encoder.fit(categories)

    seq = encoder.inverse_transform(one_hot)
    seq = seq.ravel()
    seq = ''.join(['N' if value is None else value for value in seq])
    return seq


def one_hot_encoding(array: np.ndarray,
                     read_length: int = 101,
                     one_hot_type: type = bool,
                     order: str = 'ACGT') -> np.ndarray:
    """
    Applies one-hot encoding to every read sequence in an array.

    Parameters
    ----------
    reads: np.ndarray, shape=(n,)
        1D-array of n strings
    read_length : int, default=101
        length to coerce the strings to. Longer strings will be truncated,
        while shorter strings will be filled with N bases
    one_hot_type : type, default=bool
        Type of the values in the one-hot encoding
    order : str, default='ACGT'
        Order of bases to use for one-hot encoding

    Returns
    -------
    new_array: np.ndarray, shape=(n, read_length, 4)
        3D-array with every letter from replaced by a 4 dimensional vector
        containing a 1 in the position corresponding to that letter, and 0
        elsewhere.

    See also
    --------
    one_hot_encoding_v1 : implementation used by this function
    one_hot_encoding_v2 : other implementation, slower

    Notes
    -----
    This function calls `one_hot_encoding_v1` which is currently the fastest
    implementation.
    """
    return one_hot_encoding_v1(array,
                               read_length=read_length,
                               one_hot_type=one_hot_type,
                               order=order)


def one_hot_encoding_v1(array: np.ndarray,
                        read_length: int = 101,
                        one_hot_type: type = bool,
                        order: str = 'ACGT') -> np.ndarray:
    """
    Applies one hot encoding to every read sequence in an array.

    Parameters
    ----------
    reads: np.ndarray, shape=(n,)
        1D-array of n strings
    read_length : int, default=101
        length to coerce the strings to. Longer strings will be truncated,
        while shorter strings will be filled with N bases
    one_hot_type : type, default=bool
        Type of the values in the one-hot encoding
    order : str, default='ACGT'
        Order of bases to use for one-hot encoding

    Returns
    -------
    new_array: np.ndarray, shape=(n, read_length, 4)
        3D-array with every letter from replaced by a 4 dimensional vector
        containing a 1 in the position corresponding to that letter, and 0
        elsewhere.

    See also
    --------
    one_hot_encoding : alias for this function
    one_hot_encoding_v2 : other implementation, slower
    """
    # warning raise in case sequences don't have the appropriate read_length
    new_array = np.zeros((len(array), read_length, 4), dtype=one_hot_type)
    unmatched_lengths = 0
    for i, seq in enumerate(array):
        if len(seq) != read_length:
            unmatched_lengths += 1
        for j in range(min(len(seq), read_length)):
            base = seq[j].upper()
            if base == order[0]:
                new_array[i, j, 0] = 1
            elif base == order[1]:
                new_array[i, j, 1] = 1
            elif base == order[2]:
                new_array[i, j, 2] = 1
            elif base == order[3]:
                new_array[i, j, 3] = 1
    if unmatched_lengths != 0:
        print(f"Warning: {unmatched_lengths} sequences don't have the "
              "appropriate read length")
    return new_array


def one_hot_encoding_v2(reads: np.ndarray,
                        read_length: int = 101,
                        one_hot_type: type = bool,
                        sparse: bool = False) -> np.ndarray:
    """
    Applies one hot encoding to every read sequence in an array.

    Parameters
    ----------
    reads: np.ndarray, shape=(n,)
        1D-array of n strings
    read_length : int, default=101
        length to coerce the strings to. Longer strings will be truncated,
        while shorter strings will be filled with N bases
    one_hot_type : type, default=bool
        Type of the values in the one-hot encoding
    sparse : bool, default=False
        True indicates to return a sparse matrix. False indicates to return a
        regular numpy array

    Returns
    -------
    new_array: np.ndarray, shape=(n, `read_length`, 4)
        3D-array with every letter from replaced by a 4 dimensional vector
        containing a 1 in the position corresponding to that letter, and 0
        elsewhere.

    See also
    --------
    one_hot_encoding : alias for one_hot_encoding_v1
    one_hot_encoding_v1 : other implementation, faster

    Notes
    -----
    This implementation uses scikit-learn's `OneHotEncoder`
    """
    # Change to list of chars for OneHotEncoder
    reads = [[[char] for char in read] for read in reads]
    unmatched_lengths = 0
    for i, read in enumerate(reads):
        if len(read) != read_length:
            unmatched_lengths += 1
            # truncate to read_length or add Ns to reach read_length
            reads[i] = (read[:read_length]
                        + [['N']]*max(0, read_length-len(read)))
    # Raise warning if some sequences do not match the read length
    if unmatched_lengths != 0:
        print(f"Warning: {unmatched_lengths} sequences don't have the "
              "appropriate read length")

    categories = np.array([['A'], ['C'], ['G'], ['T']])
    encoder = OneHotEncoder(dtype=one_hot_type,
                            handle_unknown='ignore',
                            sparse=sparse)
    encoder.fit(categories)

    one_hots = encoder.transform(
        np.reshape(reads, (-1, 1))
    )
    one_hots.shape = (-1, read_length, 4)
    return one_hots


def one_hot_to_seq(reads):
    return one_hot_to_seq_v2(reads)


def fast_one_hot_to_seq(reads):
    """
    Convert one_hot array of reads into list of sequences.

    This doesn't support N values, which will be converted to A.
    """
    if len(reads.shape) != 3:
        raise ValueError('must be an array of one hot encoded read')
    bases = np.array(['A', 'C', 'G', 'T'])
    indexed_reads = np.argmax(reads, axis=2)
    seqs = [''.join([char for char in seq]) for seq in bases[indexed_reads]]
    return seqs


def one_hot_to_seq_v1(reads):
    """
    Convert one_hot array of reads into list of sequences.
    """
    if len(reads.shape) == 3:
        n_reads, read_length, _ = reads.shape
    else:
        raise ValueError('must be an array of one hot encoded read')
    seqs = []
    for i in range(n_reads):
        seq = ''
        for j in range(read_length):
            one_hot = reads[i, j, :]
            if np.allclose(one_hot, np.array([1, 0, 0, 0])):
                seq += 'A'
            elif np.allclose(one_hot, np.array([0, 1, 0, 0])):
                seq += 'C'
            elif np.allclose(one_hot, np.array([0, 0, 1, 0])):
                seq += 'G'
            elif np.allclose(one_hot, np.array([0, 0, 0, 1])):
                seq += 'T'
            else:
                seq += 'N'
        seqs.append(seq)
    return seqs


def one_hot_to_seq_v2(reads):
    """
    Convert one_hot array of reads into list of sequences.

    This implementation uses scikit-learn's OneHotEncoder
    """
    if len(reads.shape) == 3:
        n_reads, read_length, n_bases = reads.shape
    else:
        raise ValueError('must be an array of one hot encoded read')
    categories = np.array([['A'], ['C'], ['G'], ['T']])
    encoder = OneHotEncoder(dtype=bool,
                            handle_unknown='ignore',
                            sparse=False)
    encoder.fit(categories)

    reads.shape = (-1, n_bases)
    seqs = encoder.inverse_transform(reads)
    reads.shape = (n_reads, read_length, n_bases)
    seqs.shape = (n_reads, read_length)
    seqs = [''.join(['N' if value is None else value for value in seq])
            for seq in seqs]
    return seqs


# Sequence manipulation
def remove_reads_with_N(sequences,
                        tolerance=0,
                        max_size=None,
                        read_length=None,
                        verbose=False):
    if max is not None:
        sequences = sequences[:max_size]
    too_short = []
    with_Ns = []
    if tolerance == 0:
        for i, seq in enumerate(sequences):
            if (read_length is not None and len(seq) < read_length):
                too_short.append(i)
            if 'N' in seq:
                with_Ns.append(i)
    else:
        for i, seq in enumerate(sequences):
            start_count = 0
            if read_length is not None:
                start_count = read_length - len(seq)
                assert(start_count >= 0)
            if seq.count('N') + start_count > tolerance:
                with_Ns.append(i)
    if verbose:
        print(too_short, ' reads too short')
        print(with_Ns, ' reads with Ns')
    sequences = np.delete(sequences, too_short + with_Ns)
    return sequences


def check_read_lengths(reads):
    """
    Return all read lengths and occurences.
    """
    dico = {}
    for read in reads:
        if len(read) in dico:
            dico[len(read)] += 1
        else:
            dico[len(read)] = 1
    return dico


def find_duplicates(reads,
                    print_freq=10_000_000,
                    one_hot=False,
                    batch_size=10_000_000):
    return find_duplicates_v1(reads,
                              print_freq=print_freq,
                              one_hot=one_hot,
                              batch_size=batch_size)


def find_duplicates_v1(reads,
                       print_freq=10_000_000,
                       one_hot=False,
                       batch_size=10_000_000):
    """
    Return all unique reads and occurences.

    Can deal with string reads or one_hot reads
    """
    dico = {}
    dup = False
    n_batch = np.ceil(len(reads) / batch_size)
    if n_batch > 1:
        batches = np.split(reads, batch_size*np.arange(1, n_batch, dtype=int))
    else:
        batches = [reads]
    print(len(batches), 'batches')
    for id, batch in enumerate(batches):
        print(f'Processing batch {id}')
        if one_hot:
            batch = one_hot_to_seq(batch)
        for i, read in enumerate(batch):
            if read in dico:
                dico[read] += 1
                dup = True
            else:
                dico[read] = 1
            if (i+1) % print_freq == 0 or i+1 == len(batch):
                msg = f'seq {i+1}/{len(batch)}'
                if dup:
                    msg += ' duplicates'
                print(msg)
    return dico


def find_duplicates_v2(reads, print_freq=10_000_000, one_hot=False):
    """
    Return all unique reads and occurences.
    """
    dico = {}
    dup = False
    for i, read in enumerate(reads):
        if one_hot:
            read = repr(read)
        if read in dico:
            dico[read] += 1
            dup = True
        else:
            dico[read] = 1
        if (i+1) % print_freq == 0:
            msg = f'seq {i+1}/{len(reads)}'
            if dup:
                msg += ' duplicates'
            print(msg)
    return dico


def find_duplicates_v3(reads, print_freq=10_000_000, one_hot=False):
    """
    Return all unique reads and occurences.
    """
    dico = {}
    dup = False
    if one_hot:
        categories = np.array([['A'], ['C'], ['G'], ['T']])
        encoder = OneHotEncoder(dtype=bool,
                                handle_unknown='ignore',
                                sparse=False)
        encoder.fit(categories)
    for i, read in enumerate(reads):
        if one_hot:
            read = encoder.inverse_transform(read).ravel()
            read = ''.join(['N' if value is None else value for value in read])
        if read in dico:
            dico[read] += 1
            dup = True
        else:
            dico[read] = 1
        if (i+1) % print_freq == 0:
            msg = f'seq {i+1}/{len(reads)}'
            if dup:
                msg += ' duplicates'
            print(msg)
    return dico


def remove_duplicates(reads, print_freq=10_000_000):
    dico = find_duplicates(reads, print_freq=print_freq)
    return dico.keys()


def chunck_into_reads(long_reads, read_length=101):
    reads = []
    for i, long in enumerate(long_reads):
        chuncks = [long[i:i+read_length]
                   for i in range(0, len(long), read_length)]
        reads.extend(chuncks)
    return reads


def reverse_complement(seq):
    reverse = ''
    for base in seq[::-1]:
        if base == 'A':
            reverse += 'T'
        elif base == 'C':
            reverse += 'G'
        elif base == 'G':
            reverse += 'C'
        elif base == 'T':
            reverse += 'A'
        else:
            reverse += base
    return reverse


def remove_windows_with_N(one_hot_seq, window_size):
    return remove_windows_with_N_v3(one_hot_seq, window_size)


def remove_windows_with_N_v1(one_hot_seq, window_size):
    """
    Remove windows containing Ns in a one-hot sequence.

    This function returns a boolean mask over the windows. Its implementation
    uses a python loop, although it is faster than the vectorized method found
    so far.
    """
    # mask positions of N values in one_hot_seq, i.e. column is all False
    N_mask = np.all(np.logical_not(one_hot_seq), axis=1)
    # create mask for valid windows, by default none are
    nb_windows = len(one_hot_seq) - window_size + 1
    valid_window_mask = np.zeros(nb_windows, dtype=bool)
    # search for Ns in first positions, before end of first window
    starting_Ns = np.where(N_mask[:window_size-1:])[0]
    # Compute distance to previous N in last_N, considering start as N
    if len(starting_Ns) == 0:
        # No N found, previous N is the start position
        last_N = window_size - 1
    else:
        # At least one N found, previous N is at the highest position
        last_N = window_size - 2 - np.max(starting_Ns)
    for i, isN in enumerate(N_mask[window_size-1:]):
        if isN:
            last_N = 0
        else:
            last_N += 1  # increase distance before testing
            if last_N >= window_size:
                # far enough from previous N for a valid window
                valid_window_mask[i] = True
    return valid_window_mask


def remove_windows_with_N_v2(one_hot_seq, window_size):
    """
    Remove windows containing Ns in a one-hot sequence.

    This function  returns indexes of valid windows. Its implementation is
    vetorized, although slower than the naive approach.
    """
    # Find indexes of N values in one_hot_seq, i.e. column is all False
    N_idx = np.where(np.all(np.logical_not(one_hot_seq), axis=1))[0]
    # Compute distance from each position to previous N
    # Start at 1 to consider start as an N
    last_N_indexes = np.arange(1, len(one_hot_seq)+1)
    # Split at each N, and reset counter
    for split in np.split(last_N_indexes, N_idx)[1:]:
        split -= split[0]
    # Select windows by last element, if it is far enough from last N
    valid_window_mask = np.where(
        last_N_indexes[window_size-1:] >= window_size)[0]
    return valid_window_mask


def remove_windows_with_N_v3(one_hot_seq, window_size):
    """
    Remove windows containing Ns in a one-hot sequence.

    This function returns indexes of valid windows. Its implementation
    uses a python loop, although it is faster than the vectorized method found
    so far.
    For human chromosome 1 : 35s
    """
    # mask positions of N values in one_hot_seq, i.e. column is all False
    N_mask = np.all(np.logical_not(one_hot_seq), axis=1)
    # Store valid window indexes
    valid_window_idx = []
    # Search for Ns in first positions, before end of first window
    starting_Ns = np.where(N_mask[:window_size-1:])[0]
    if len(starting_Ns) == 0:
        # No N found, previous N is the start position
        last_N = window_size - 1
    else:
        # At least one N found, previous N is at the highest position
        last_N = window_size - 2 - np.max(starting_Ns)
    for i, isN in enumerate(N_mask[window_size-1:]):
        if isN:
            last_N = 0
        else:
            last_N += 1  # increase distance before testing
            if last_N >= window_size:
                # far enough from previous N for a valid window
                valid_window_idx.append(i)
    return np.array(valid_window_idx, dtype=int)


# Standard file format functions
def write_fasta(seqs: dict,
                fasta_file: str,
                wrap: int = None) -> None:
    """Write sequences to a fasta file.

    Found on https://www.programcreek.com/python/?code=Ecogenomics%2FGTDBTk%
    2FGTDBTk-master%2Fgtdbtk%2Fbiolib_lite%2Fseq_io.py

    Parameters
    ----------
    seqs : dict[seq_id] -> str
        Sequences indexed by sequence id, works with any iterable.
    fasta_file : str
        Path to write the sequences to.
    wrap: int
        Number of bases before the line is wrapped.
    """
    with open(fasta_file, 'w') as f:
        for id, seq in enumerate(seqs):
            f.write('>{}\n'.format(id))
            if wrap is not None:
                for i in range(0, len(seq), wrap):
                    f.write('{}\n'.format(seq[i:i + wrap]))
            else:
                f.write('{}\n'.format(seq))


def parse_bed_peaks(bed_file, window_size=101, merge=True):
    with open(bed_file, 'r') as f:
        chr_peaks = {}
        for line in f:
            line = line.rstrip()
            chr_id, start, end, _, score, *_ = line.split('\t')
            # chr_id = chr_id[3:]
            start, end, score = tuple(
                int(item) for item in (start, end, score))
            if chr_id in chr_peaks.keys():
                chr_peaks[chr_id].append(np.array([start, end, score]))
            else:
                chr_peaks[chr_id] = [np.array([start, end, score])]
        for key in chr_peaks.keys():
            # convert to array, remove duplicates and adjust indices to window
            chr_peaks[key] = (np.unique(np.array(chr_peaks[key]), axis=0)
                              - np.array([1, 1, 0]) * window_size // 2)
            try:
                # Check if some peaks overlap
                overlaps, _ = self_overlapping_peaks(chr_peaks[key],
                                                     merge=merge)
                assert len(overlaps) == 0
            except AssertionError:
                print(f'Warning: some peaks overlap in chr{key}')
    return chr_peaks


def parse_repeats(repeat_file, window_size=101, header_lines=3):
    with open(repeat_file, 'r') as f:
        # skip first lines
        for i in range(header_lines):
            next(f)
        # build depth 2 dictionnary, first key is chr_id and 2nd key is family
        repeats = defaultdict(lambda: defaultdict(list))
        for line in f:
            line = line.strip()
            _, _, _, _, chr_id, start, end, _, _, _, family, *_ = line.split()
            chr_id = chr_id[3:]
            start, end = tuple(int(item) for item in (start, end))
            repeats[chr_id][family].append(np.array([start, end]))
        for chr_id in repeats.keys():
            for family in repeats[chr_id].keys():
                # convert to array and adjust indices to window
                repeats[chr_id][family] = (np.array(repeats[chr_id][family])
                                           - window_size // 2)
    return repeats


def parse_sam(sam_file: str, verbose=True) -> None:
    with open(sam_file, 'r') as f:
        chr_coord = defaultdict(list)
        header_regexp = re.compile('^@(HD|SQ|RG|PG|CO)')
        rejected_count = 0
        total_count = 0
        for line in f:
            if header_regexp.match(line):  # ignore header
                continue
            # Readline and convert some entries to int
            _, _, rname, pos, _, _, _, _, tlen, *_ = line.split('\t')
            tlen, pos = (int(v) for v in (tlen, pos))
            # Record only the leftmost read of each pair
            if tlen > 0:
                # middle = math.floor((pos + pnext + len(seq)) / 2)
                chr_coord[rname].append([pos, pos + tlen])
            else:
                rejected_count += 1
            total_count += 1
    if verbose:
        print(f'{rejected_count}/{total_count} paired reads rejected')
    return chr_coord


def parse_bam(bam_file: str,
              mapq_thres=None,
              verbose=True,
              paired=True,
              fragment_length=None,
              max_fragment_len=None,
              id_file=None) -> None:
    if id_file:
        with open(id_file) as f_id:
            ids_set = {x.split()[0][1:] for x in f_id}
    with pysam.AlignmentFile(bam_file, 'rb') as f:
        chr_coord = defaultdict(list)
        rejected_count = 0
        total_count = 0
        for read in f.fetch():
            if paired:
                tlen = read.template_length
                if tlen <= 0:
                    continue
                if max_fragment_len is not None and tlen > max_fragment_len:
                    rejected_count += 1
                    total_count += 1
                    continue
            else:
                tlen = fragment_length
            total_count += 1
            if ((mapq_thres is not None
                 and read.mapping_quality < mapq_thres)
                or (max_fragment_len is not None
                    and tlen > max_fragment_len)
                or (id_file is not None
                    and read.query_name not in ids_set)):
                # reject the read
                rejected_count += 1
                continue
            else:
                rname = read.reference_name
                pos = read.reference_start
                chr_coord[rname].append([pos, pos + tlen])
    if verbose:
        print(f'{rejected_count}/{total_count} reads rejected')
    return chr_coord


def load_annotation(file, chr_id, window_size, anchor='center'):
    bw = pyBigWig.open(file)
    values = bw.values(f"chr{chr_id}", 0, -1, numpy=True)
    values[np.isnan(values)] = 0
    values = adapt_to_window(values, window_size, anchor=anchor)
    return values


def adapt_to_window(values: np.ndarray,
                    window_size: int,
                    anchor: str = 'center') -> np.ndarray:
    """Selects a slice from `values` to match a sliding window anchor.

    When anchor is 'center', the slice is adapted to match the middle points
    of the sliding window along values.

    Parameters
    ----------
    values : ndarray
        1D-array of values to slice from
    window_size : int
        Size of the window to slide along `values`, must be smaller than the
        size of values
    anchor : {center, start, end}, default='center'
        Specifies which point of the window the values should match

    Returns
    -------
    ndarray
        1D-array which is a contiguous slice of `values`
    """
    if anchor == 'center':
        return values[(window_size // 2):
                      (- ((window_size+1) // 2) + 1)]
    elif anchor == 'start':
        return values[:-window_size+1]
    elif anchor == 'end':
        return values[window_size-1:]
    else:
        raise ValueError("Choose anchor from 'center', 'start' or 'end'")


# GC content
def GC_content(one_hot_reads: np.ndarray) -> np.ndarray:
    """Compute GC content on all reads in one-hot format

    Parameters
    ----------
    one_hot_reads : np.ndarray, shape=(n, l, 4)
        3D-array containing n reads of length l one-hot encoded on 4 values

    Returns
    -------
    gc : np.ndarray, shape=(n,)
        1D-array of gc content for each read
    """
    assert(len(one_hot_reads.shape) == 3 and one_hot_reads.shape[2] == 4)
    # Compute content of each base
    content = np.sum(one_hot_reads, axis=1)  # shape (nb_reads, 4)
    gc = (content[:, 1] + content[:, 2]) / np.sum(content, axis=1)
    return gc


def classify_1D(features, y, bins):
    """Find best threshold to classify 1D features with label y.

    Computing is done in bins for fast execution, so it isn't exact
    """
    def cumul_count(features, bins):
        feature_bins = np.digitize(features, bins).ravel()
        count = np.bincount(feature_bins, minlength=len(bins)+1)[1:]
        return np.cumsum(count)
    bins = np.histogram(features, bins=bins, range=(0, 1))[1]
    features_pos = features[y == 1]
    features_neg = features[y == 0]
    cumul_pos_count = cumul_count(features_pos, bins)
    cumul_neg_count = cumul_count(features_neg, bins)
    cumul_diff = cumul_pos_count - cumul_neg_count
    bin_thres = np.argmax(np.abs(cumul_diff))
    if cumul_diff[bin_thres] < 0:
        accuracy = (len(features_pos) - cumul_diff[bin_thres]) / len(features)
    else:
        accuracy = (len(features_neg) + cumul_diff[bin_thres]) / len(features)
    # assert(bin_thres != len(bins) - 1)
    thres = (bins[bin_thres] + bins[bin_thres+1]) / 2
    return accuracy, thres


# Signal manipulation
def z_score(preds, rel_indices=None):
    if rel_indices is not None:
        rel_preds = preds[rel_indices]
        mean, std = np.mean(rel_preds), np.std(rel_preds)
    else:
        mean, std = np.mean(preds), np.std(preds)
    return (preds - mean)/std


def smooth(values, window_size, mode='linear', sigma=1):
    if mode == 'linear':
        box = np.ones(window_size) / window_size
    elif mode == 'gaussian':
        box = gaussian(window_size, sigma)
        box /= np.sum(box)
    elif mode == 'triangle':
        box = np.concatenate((np.arange((window_size+1) // 2),
                              np.arange(window_size // 2 - 1, -1, -1)),
                             dtype=float)
        box /= np.sum(box)
    else:
        raise NameError("Invalid mode")
    return convolve(values, box, mode='same')


def binned_alignment_count_from_coord(coord: np.ndarray,
                                      binsize: int = 100,
                                      length: int = None) -> np.ndarray:
    """
    Build alignment count signal from read coordinates on a single chromosome.

    The signal is binned to a specified resolution on the genome. This
    function will build the signal from mid points, resulting in very noisy
    and potentially inaccurate signal for high resolution (low bins).

    Parameters
    ----------
    coord : ndarray, shape=(nb_reads, 2)
        2D-array of coordinates for start and end of each fragment.
    bins : int, default=100
        Length of bins, in bases, to divide the signal into.
    length : int, optional
        Length of the full chromosome

    Returns
    -------
    ndarray
        1D-array of read count in each bin along the chromosome.

    Note
    ----
    Very fast implementation through the use of scipy.sparse's matrix creation
    and conversion.
    """
    binned_mid = np.floor(np.mean(coord, axis=1) // binsize)
    binned_mid = np.array(binned_mid, dtype=int)
    if length is None:
        length = np.max(binned_mid) + 1
    else:
        length = length // binsize + 1
        if length < np.max(binned_mid) + 1:
            raise ValueError("coordinates go beyond the specified length")
    return coo_matrix(
        (np.ones(len(coord), dtype=int),
         (binned_mid, np.zeros(len(coord), dtype=int))),
        shape=(length, 1)
    ).toarray().ravel()


def exact_alignment_count_from_coord(coord: np.ndarray) -> np.ndarray:
    """
    Build alignment count signal from read coordinates on a single chromosome.

    Parameters
    ----------
    coord : ndarray, shape=(nb_reads, 2)
        2D-array of coordinates for start and end of each fragment.

    Returns
    -------
    ndarray
        1D-array of read count for each position along the chromosome.
    """
    # Insert +1 at fragment start and -1 after fragment end
    data = np.ones(2*len(coord), dtype=int)
    data[1::2] = -1
    # Get coordinate of first bp after fragment end
    coord[:, 1] = coord[:, 1] + 1
    # Insert using scipy.sparse implementation
    start_ends = coo_matrix(
        (data, (coord.ravel(), np.zeros(2*len(coord), dtype=int))),
        shape=(np.max(coord)+1, 1)
    ).toarray().ravel()
    # Cumulative sum to propagate full fragments
    return np.cumsum(start_ends)


def bin_preds(preds, bins):
    if len(preds) % bins == 0:
        binned_preds = np.mean(strided_window_view(preds, bins, bins), axis=1)
    else:
        binned_preds = np.append(
            np.mean(strided_window_view(preds, bins, bins), axis=1),
            np.mean(preds[-(len(preds) % bins):]))
    return binned_preds


def full_genome_binned_preds(data,
                             genome,
                             model_name,
                             bins,
                             data_dir='shared_folder'):
    if genome == 'T2T-CHM13v2.0':
        lengths = T2T_lengths
        chr_ids = T2T_chr_ids
    elif genome == 'GRCh38':
        lengths = GRCh38_lengths
        chr_ids = hg38_chr_ids
    # merging chromosomes
    binned_lengths = np.array([x // bins + 1 for x in lengths.values()])
    seperators = np.cumsum(binned_lengths)
    total_length = seperators[-1]
    full = np.zeros(total_length)
    for i, chr_id in enumerate(chr_ids.keys()):
        with np.load(Path(data_dir,
                          data,
                          'results',
                          model_name,
                          f'preds_on_{genome}.npz')) as f:
            preds = f[f'chr{chr_id}']
        binned_preds = bin_preds(preds, bins)
        full[seperators[i]-len(binned_preds):seperators[i]] = binned_preds
    return full, seperators


def enrichment_analysis(signal, ctrl, verbose=True, data='signal'):
    n_binom = signal + ctrl
    p_binom = np.sum(signal) / np.sum(n_binom)
    binom_pvalue = clip_to_nonzero_min(
        1 - scipy.stats.binom.cdf(signal - 1, n_binom, p_binom))
    reject, qvalue, *_ = multitest.multipletests(binom_pvalue, method='fdr_bh')
    binom_qvalue = qvalue
    neg_log_qvalue = -np.log10(qvalue)
    neg_log_pvalue = -np.log10(binom_pvalue)
    significantly_enriched = reject
    if verbose:
        print(f'{np.sum(reject)}/{len(reject)} '
              f'significantly enriched bins in {data}')
    return pd.DataFrame({
        "binom_p_value_complete": binom_pvalue,
        "binom_q_value_complete": binom_qvalue,
        "-log(pvalue_complete)": neg_log_pvalue,
        "-log(qvalue_complete)": neg_log_qvalue,
        "significantly_enriched": significantly_enriched})


def genome_enrichment(ip_coord_file,
                      ctrl_coord_file,
                      chr_sizes_file,
                      out_file,
                      max_frag_len=500,
                      binsize=200,
                      verbose=True):
    # Log parameters
    wdir = Path(out_file).parent()
    log_file = Path(wdir, 'alignment_analysis_log.txt')
    log_file = safe_filename(log_file)
    with open(log_file, 'w') as f:
        f.write(f'chromosome sizes file: {chr_sizes_file}\n'
                f'max fragment length: {max_frag_len}\n'
                f'bin size: {binsize}\n\n')
    # Get chromosome lengths
    chr_lens = json.load(chr_sizes_file)
    binned_chr_lens = np.array([x // binsize + 1 for x in chr_lens.values()])
    chr_seps = np.cumsum(binned_chr_lens)
    total_length = chr_seps[-1]
    # Build DataFrame
    columns = ['pos', 'ip_count', 'ctrl_count', 'pval', 'qval']
    df = pd.DataFrame(np.zeros((total_length, len(columns))), columns=columns)
    # Loop over chromosomes
    for i, chr_id in enumerate(chr_lens.keys()):
        if verbose:
            print(f"Processing chr {chr_id}...")
        # Load chromosome fragment coordinates
        with np.load(ip_coord_file) as f:
            ip_coord_chr = f[chr_id]
        with np.load(ctrl_coord_file) as f:
            ctrl_coord_chr = f[chr_id]
        # Filter out fragments too long
        ip_frag_lens_chr = np.diff(ip_coord_chr, axis=1).ravel()
        ctrl_frag_lens_chr = np.diff(ctrl_coord_chr, axis=1).ravel()
        ip_coord_chr = ip_coord_chr[ip_frag_lens_chr <= max_frag_len, :]
        ctrl_coord_chr = ctrl_coord_chr[ctrl_frag_lens_chr <= max_frag_len, :]
        # Get binned middle alignment
        ip_count_chr = binned_alignment_count_from_coord(
            ip_coord_chr, binsize=binsize, length=chr_lens[chr_id])
        ctrl_count_chr = binned_alignment_count_from_coord(
            ctrl_coord_chr, binsize=binsize, length=chr_lens[chr_id])
        # Insert in DataFrame
        df.iloc[chr_seps[i]-len(ip_count_chr):chr_seps[i], :3] = np.transpose(
            np.vstack((np.arange(0, len(ip_count_chr)*binsize, binsize),
                       ip_count_chr,
                       ctrl_count_chr)))
        # Write log info
        with open(log_file, 'a') as f:
            f.write(f'Processing chr {chr_id}...\n'
                    f'{np.sum(ip_frag_lens_chr >= max_frag_len)} fragments '
                    'longer than {max_frag_len}bp in IP\n'
                    f'{np.sum(ctrl_frag_lens_chr >= max_frag_len)} fragments '
                    'longer than {max_frag_len}bp in Control\n')
    # Save final DataFrame
    out_file = safe_filename(out_file)
    df.to_csv(out_file)


def downsample_enrichment_analysis(data,
                                   genome,
                                   max_fragment_len,
                                   bins_list=[1000],
                                   frac_list=[1],
                                   div_list=None,
                                   reverse=True,
                                   data_dir='../shared_folder',
                                   use_fdr=True):
    if div_list is not None:
        frac_list = 1 / np.array(frac_list)
    mindex = pd.MultiIndex.from_product([bins_list, frac_list])
    if reverse:
        res = pd.DataFrame(
            index=mindex,
            columns=['IP', 'IP_clust', 'Undetermined', 'Ctrl', 'Ctrl_clust',
                     'total_cov'])
    else:
        res = pd.DataFrame(
            index=mindex,
            columns=['IP', 'IP_clust', 'Undetermined', 'total_cov'])
    if genome == 'T2T-CHM13v2.0':
        lengths = T2T_lengths
        chr_ids = T2T_chr_ids
    elif genome == 'GRCh38':
        lengths = GRCh38_lengths
        chr_ids = hg38_chr_ids
    # merging chromosomes
    for bins in bins_list:
        binned_lengths = np.array([x // bins + 1 for x in lengths.values()])
        seperators = np.cumsum(binned_lengths)
        total_length = seperators[-1]
        for i, chr_id in enumerate(chr_ids.keys()):
            df = pd.read_csv(
                Path(data_dir, data, 'results', 'alignments', genome,
                     f'{data}_{genome}_chr{chr_id}_'
                     f'thres_{max_fragment_len}_binned_{bins}.csv'),
                index_col=0)
            if i == 0:
                full_genome = pd.DataFrame(
                    np.zeros((total_length, len(df.columns))),
                    columns=df.columns)
            full_genome.iloc[seperators[i]-len(df):seperators[i], :] = df
        # computing p_value and q_value
        for frac in frac_list:
            frac_IP = integer_histogram_sample(
                full_genome["ip_binned_signal"], frac)
            frac_Ctrl = integer_histogram_sample(
                full_genome["ctrl_binned_signal"], frac)
            n = frac_IP + frac_Ctrl
            cov = np.sum(n)
            p_binom = np.sum(frac_IP) / cov
            p_values = clip_to_nonzero_min(
                1 - scipy.stats.binom.cdf(frac_IP - 1, n, p_binom))
            if use_fdr:
                valid_bins = (n != 0)
                signif_IP = np.zeros(len(full_genome), dtype=bool)
                signif_IP[valid_bins], *_ = multitest.multipletests(
                    p_values[valid_bins], method='fdr_bh')
            else:
                signif_IP = np.array(p_values < 0.05)
            n_signif_IP = np.sum(signif_IP)
            n_signif_IP_clust = nb_boolean_true_clusters(signif_IP)
            tot = len(signif_IP)
            if reverse:
                rev_p_values = clip_to_nonzero_min(
                    1 - scipy.stats.binom.cdf(frac_Ctrl - 1, n, 1 - p_binom))
                if use_fdr:
                    signif_Ctrl = np.zeros(len(full_genome), dtype=bool)
                    signif_Ctrl[valid_bins], *_ = multitest.multipletests(
                        rev_p_values[valid_bins], method='fdr_bh')
                else:
                    signif_Ctrl = np.array(rev_p_values < 0.05)
                n_signif_Ctrl = np.sum(signif_Ctrl)
                n_signif_Ctrl_clust = nb_boolean_true_clusters(signif_Ctrl)
                res.loc[bins, frac] = [n_signif_IP,
                                       n_signif_IP_clust,
                                       tot - n_signif_IP - n_signif_Ctrl,
                                       n_signif_Ctrl,
                                       n_signif_Ctrl_clust,
                                       cov]
            else:
                res.loc[bins, frac] = [n_signif_IP,
                                       n_signif_IP_clust,
                                       tot - n_signif_IP,
                                       cov]
    return res


def pool_experiments(dfs, verbose=True):
    cols_to_take = ['ip_binned_signal', 'ctrl_binned_signal']
    df_pooled = dfs[0][['pos'] + cols_to_take].copy()
    for df in dfs[1:]:
        df_pooled[cols_to_take] += df[cols_to_take]
    # computing p_value and q_value
    sums = df_pooled.sum(axis=0)
    p_binom = sums["ip_binned_signal"] / (sums["ip_binned_signal"]
                                          + sums["ctrl_binned_signal"])
    n_binom = df_pooled["ip_binned_signal"] + df_pooled["ctrl_binned_signal"]
    df_pooled['binom_p_value_complete'] = clip_to_nonzero_min(
        1 - scipy.stats.binom.cdf(df_pooled["ip_binned_signal"] - 1,
                                  n_binom, p_binom))
    reject, q_value, *_ = multitest.multipletests(
        df_pooled["binom_p_value_complete"], method='fdr_bh')
    df_pooled['binom_q_value_complete'] = q_value
    df_pooled['-log(qvalue_complete)'] = -np.log10(q_value)
    df_pooled['-log(pvalue_complete)'] = -np.log10(
        df_pooled["binom_p_value_complete"])
    df_pooled['significantly_enriched'] = reject
    if verbose:
        print(f'{np.sum(reject)}/{len(reject)} '
              f'significantly enriched bins in dataframe')
    return df_pooled


# Peak manipulation
def find_peaks(preds: np.ndarray,
               pred_thres: float,
               length_thres: int = 1,
               tol: int = 1) -> np.ndarray:
    """Determine peaks from prediction signal and threshold.

    Identify when `preds` is above the threshold `pred_thres` pointwise,
    then determine regions of consecutive high prediction, called peaks.

    Parameters
    ----------
    preds : ndarray
        1D-array of predictions along the chromosome
    pred_thres : float
        Threshold above which prediction is considered in a peak
    length_thres : int, default=1
        Minimum length required for peaks, any peak below or equal to that
        length will be discarded
    tol : int, default=1
        Distance between consecutive peaks under which the peaks are merged
        into one. Can be set higher to get a single peak when signal is
        fluctuating too much. Unlike slices, peaks include their end points,
        meaning [1 2] and [4 5] actually contain a gap of one base,
        but the distance is 2 (4-2). The default value of 1 means that no
        peaks will be merged.

    Returns
    -------
    peaks : ndarray, shape=(n, 2)
        2D-array, each line corresponds to a peak. A peak is a 1D-array of
        size 2, with format [peak_start, peak_end]. `peak_start` and
        `peak_end` are indices on the chromosome.
    """
    # Find pointwise peaks as predictions above the threshold
    peak_mask = (preds > pred_thres)
    # Find where peak start and end
    change_idx = np.where(peak_mask[1:] != peak_mask[:-1])[0] + 1
    if peak_mask[0]:
        # If predictions start with a peak, add an index at the start
        change_idx = np.insert(change_idx, 0, 0)
    if peak_mask[-1]:
        # If predictions end with a peak, add an index at the end
        change_idx = np.append(change_idx, len(peak_mask))
    # # Check that change_idx contains as many starts as ends
    # assert (len(change_idx) % 2 == 0)
    # Merge consecutive peaks if their distance is below a threshold
    if tol != 0:
        # Compute difference between end of peak and start of next one
        diffs = change_idx[2::2] - change_idx[1:-1:2]
        # Get index when difference is below threshold, see below for matching
        # index in diffs and in change_idx
        # diff index:   0   1   2  ...     n-1
        # change index:1-2 3-4 5-6 ... (2n-1)-2n
        small_diff_idx, = np.where(diffs <= tol)
        delete_idx = np.concatenate((small_diff_idx*2 + 1,
                                     small_diff_idx*2 + 2))
        # Remove close ends and starts using boolean mask
        mask = np.ones(len(change_idx), dtype=bool)
        mask[delete_idx] = False
        change_idx = change_idx[mask]
    # Reshape as starts and ends
    peaks = np.reshape(change_idx, (-1, 2))
    # Compute lengths of peaks and remove the ones below given threshold
    lengths = np.diff(peaks, axis=1).ravel()
    peaks = peaks[lengths > length_thres]
    return peaks


def find_peaks_in_window(peaks: np.ndarray,
                         window_start: int,
                         window_end: int) -> np.ndarray:
    """Find peaks overlapping with the window and cut them to fit the window.

    Parameters
    ----------
    peaks : ndarray, shape=(n, m)
        2D-array, each line corresponds to a peak. A peak is a 1D-array of
        size m0 = 2 or 3, with format [peak_start, peak_end, *optional_score].
        `peak_start` and `peak_end` must be indices on the chromosome.
        Peaks mustn't overlap, meaning that there is no other peak starting or
        ending between `peak_start` and `peak_end`.
    window_start, window_end : int
        Indices of the start and end of the window to be displayed

    Returns
    -------
    valid_peaks : ndarray, shape=(l, m)
        2D-array of peaks overlapping on the window and cut to fit in the
        window.
    """
    # Sort peaks by peak_start
    sorted_peaks = peaks[np.argsort(peaks[:, 0]), :]
    # Remove score and flatten
    flat_peaks = sorted_peaks[:, :2].ravel()
    # Find first and last peaks to include
    first_id = np.searchsorted(flat_peaks, window_start)
    last_id = np.searchsorted(flat_peaks, window_end - 1)
    # Adapt indices for the 2D-array
    valid_peaks = sorted_peaks[(first_id // 2):((last_id + 1) // 2), :]
    # Cut first and last peaks if they exceed window size
    if first_id % 2 == 1:
        valid_peaks[0, 0] = window_start
    if last_id % 2 == 1:
        valid_peaks[-1, 1] = window_end - 1
    return valid_peaks


def overlap(peak0: np.ndarray,
            peak1: np.ndarray,
            tol: int = 0) -> tuple:  # tuple[bool, bool]:
    """Determine whether peaks overlap and which one ends first.

    Parameters
    ----------
    peak0, peak1 : ndarray
        1D-arrays with format [peak_start, peak_end, *optional_score].
        `peak_start` and `peak_end` must be indices on the chromosome. `peak0`
        and `peak1` may have different sizes since the score is ignored
    tol : int, default=0
        Maximum difference between peak_end and the next peak_start to
        consider as an overlap. This value defaults to 0 because unlike slices,
        peaks include their end points, meaning [1 2] and [2 5] actually
        overlap.

    Returns
    -------
    overlaps : bool
        True if peaks overlap by at least one point.
    end_first : bool
        Index of the peak with lowest end point.
    """
    start0, end0, *_ = peak0
    start1, end1, *_ = peak1
    overlaps = (end0 + tol >= start1) and (end1 + tol >= start0)
    end_first = end0 > end1
    return overlaps, end_first


def overlapping_peaks(peaks0: np.ndarray, peaks1: np.ndarray) -> tuple:
    """Determine overlaps between two arrays of disjoint peaks.

    Parameters
    ----------
    peaks0 : ndarray, shape=(n0, m0)
        2D-array, each line corresponds to a peak. A peak is a 1D-array of
        size m0 = 2 or 3, with format [peak_start, peak_end, *optional_score].
        `peak_start` and `peak_end` must be indices on the chromosome.
        Peaks must be disjoint within a 2D-array, meaning that there is no
        other peak starting or ending between `peak_start` and `peak_end`.
    peaks1 : ndarray, shape=(n1, m1)
        Same as peaks0, but with potentially different shape.

    Returns
    -------
    overlapping : List[List[ndarray], List[ndarray]]
        First list contains peaks from peaks0 overlapping with at least one
        peak from peaks1 and second list contains peaks from peaks1
        overlapping with at least one peak from peaks0.
    non_overlapping : List[List[ndarray], List[ndarray]]
        First list contains peaks from peaks0 that do not overlap with any
        peak from peaks1 and second list contains peaks from peaks1 that do
        not overlap with any peak from peaks0

    See also
    --------
    self_overlapping_peaks : find overlapping peaks within a single array

    Notes
    -----
    Both arrays are sorted then first peaks from each array are tested for
    overlap. The first ending peak is discarded and put into the
    appropriate output list. The remaining peak is then compared to the next
    one in the array from which the previous peak was discarded. The flag
    `remember_overlaps` is set to True anytime we see an overlap, to remember
    that the remaining peak must be put in the overlapping list even if it
    doesn't overlap with the next one.
    """
    # Sort peak lists by peak_start
    sorted_0 = list(peaks0[np.argsort(peaks0[:, 0]), :])
    sorted_1 = list(peaks1[np.argsort(peaks1[:, 0]), :])
    # Merge into one list for simple index accession
    sorted = [sorted_0, sorted_1]
    # Flag
    remember_overlaps = False
    # Initialize output lists
    overlapping = [[], []]
    non_overlapping = [[], []]
    while sorted_0 and sorted_1:
        # Check overlap between first peaks of both lists
        overlaps, end_first = overlap(sorted_0[0], sorted_1[0])
        # Remove first ending peak because it can't overlap with others anymore
        peak = sorted[end_first].pop(0)
        if overlaps:
            # Overlap -> set flag to True and store peak in overlapping
            remember_overlaps = True
            overlapping[end_first].append(peak)
        elif remember_overlaps:
            # No overlap but flag is True -> set flag back to False and
            #                                store peak in overlapping
            remember_overlaps = False
            overlapping[end_first].append(peak)
        else:
            # No overlap, flag is False -> store peak in non overlapping
            non_overlapping[end_first].append(peak)
    # Index of the non empty list
    non_empty = bool(sorted_1)
    if remember_overlaps:
        # Flag is True -> store first remaining peak in overlapping
        overlapping[non_empty].append(sorted[non_empty].pop())
    for peak in sorted[non_empty]:
        # Store all leftover peaks in non overlapping
        non_overlapping[non_empty].append(peak)
    return overlapping, non_overlapping


def self_overlapping_peaks(peaks: np.ndarray,
                           merge: bool = True,
                           tol: int = 1
                           ) -> tuple:
    # tuple(np.ndarray, Optional(np.ndarray)):
    """Determine which peaks within the array overlap

    As opposed to `overlapping_peaks`, here two disjoint but adjacent peaks
    will be considered self-overlapping since then can be merged into one
    contiguous peak.

    Parameters
    ----------
    peaks : ndarray, shape=(n, m)
        2D-array, each line corresponds to a peak. A peak is a 1D-array of
        size m0 = 2 or 3, with format [peak_start, peak_end, *optional_score].
        `peak_start` and `peak_end` must be indices on the chromosome.
        Peaks must be disjoint within a 2D-array, meaning that there is no
        other peak starting or ending between `peak_start` and `peak_end`.
    merge : bool, default=True
        True indicates to return an array with overlapping peaks merged. False
        indicates to not perform this operation, which can be faster
    tol : int, default=1
        Maximum difference between peak_end and the next peak_start to
        consider as an overlap. This value defaults to 1 because unlike slices,
        peaks include their end points, meaning [1 2] and [3 5] are actually
        adjacent.

    Returns
    -------
    overlap_idx : ndarray
        Indices of peaks overlapping with the next one in th array in order of
        increasing start position
    merged : ndarray, shape=(l, k)
        Returned only if `merge` was set to True. The array of merged peaks
        whenever there was an overlap. If no overlaps were found, `peaks` is
        returned as is. Otherwise if a score field was present in `peaks`, it
        is not present in the merged array because the score of a peak merging
        several peaks with different scores is ambiguous.

    See also
    --------
    overlapping_peaks : determine overlaps between two arrays of disjoint peaks
    """
    # Sort peaks by peak_start, remove score and flatten
    sorted_by_starts = peaks[np.argsort(peaks[:, 0]), :2].ravel()
    # Group peak_ends and next peak_starts
    gaps = sorted_by_starts[1:-1].reshape(-1, 2)
    # Compute gap distances and select when it is smaller than the tolerance
    diffs = - np.diff(gaps, axis=1).ravel()
    overlap_idx, = np.where(diffs >= - tol)
    if merge:
        if len(overlap_idx) != 0:
            # Compute indices for the full flatten array
            delete_idx = np.concatenate((overlap_idx*2 + 1,
                                         overlap_idx*2 + 2))
            # Remove overlapping ends and starts using boolean mask
            mask = np.ones(len(sorted_by_starts), dtype=bool)
            mask[delete_idx] = False
            # Select valid indices and reshape in 2D
            merged = sorted_by_starts[mask].reshape(-1, 2)
        else:
            merged = peaks
        return overlap_idx, merged
    return overlap_idx


# numpy helper functions
def is_sorted(array: np.ndarray) -> bool:
    """Check that a 1D-array is sorted.

    Parameters
    ----------
    array : array_like
        1D-array to be checked.

    Returns
    -------
    bool
        True if `array` is sorted, False otherwise.
    """
    return np.all(array[:-1] <= array[1:])


def argmax_last(array: np.ndarray) -> int:
    """Return index of maximal value in a 1D-array.

    Unlike numpy.argmax, this function returns the last occurence of the
    maximal value. It only works for 1D-arrays.

    Parameters
    ----------
    array : array_like
        1D-array to find maximal value in.

    Returns
    -------
    int
        Index of the last occurence of the maximal value in `array`.
    """
    return len(array) - np.argmax(array[::-1]) - 1


def adjust_length(x: np.ndarray, y: np.ndarray) -> tuple:
    """
    Append 0s to the shortest array to adjust their lengths

    Parameters
    ----------
    x, y : ndarray
        1D-arrays of potentially different lengths

    Returns
    -------
    x, y : ndarray
        1D-arrays of same length
    """
    if len(x) < len(y):
        x = np.append(x, np.zeros(len(y) - len(x), dtype=x.dtype))
    else:
        y = np.append(y, np.zeros(len(x) - len(y), dtype=y.dtype))
    return x, y


def sliding_window_view(x, window_shape, axis=None, *,
                        subok=False, writeable=False):
    """Function from the numpy library"""
    window_shape = (tuple(window_shape)
                    if np.iterable(window_shape)
                    else (window_shape,))
    # first convert input to array, possibly keeping subclass
    x = np.array(x, copy=False, subok=subok)

    window_shape_array = np.array(window_shape)
    if np.any(window_shape_array < 0):
        raise ValueError('`window_shape` cannot contain negative values')

    if axis is None:
        axis = tuple(range(x.ndim))
        if len(window_shape) != len(axis):
            raise ValueError(f'Since axis is `None`, must provide '
                             f'window_shape for all dimensions of `x`; '
                             f'got {len(window_shape)} window_shape elements '
                             f'and `x.ndim` is {x.ndim}.')
    else:
        axis = normalize_axis_tuple(axis, x.ndim, allow_duplicate=True)
        if len(window_shape) != len(axis):
            raise ValueError(f'Must provide matching length window_shape and '
                             f'axis; got {len(window_shape)} window_shape '
                             f'elements and {len(axis)} axes elements.')

    out_strides = x.strides + tuple(x.strides[ax] for ax in axis)

    # note: same axis can be windowed repeatedly
    x_shape_trimmed = list(x.shape)
    for ax, dim in zip(axis, window_shape):
        if x_shape_trimmed[ax] < dim:
            raise ValueError(
                'window shape cannot be larger than input array shape')
        x_shape_trimmed[ax] -= dim - 1
    out_shape = tuple(x_shape_trimmed) + window_shape
    return as_strided(x, strides=out_strides, shape=out_shape,
                      subok=subok, writeable=writeable)


def strided_window_view(x, window_shape, stride, out_shape=None,
                        axis=None, *, subok=False, writeable=False):
    """Variant of `sliding_window_view` which supports stride parameter."""
    window_shape = (tuple(window_shape)
                    if np.iterable(window_shape)
                    else (window_shape,))
    # first convert input to array, possibly keeping subclass
    x = np.array(x, copy=False, subok=subok)

    window_shape_array = np.array(window_shape)
    if np.any(window_shape_array < 0):
        raise ValueError('`window_shape` cannot contain negative values')

    if axis is None:
        axis = tuple(range(x.ndim))
        if len(window_shape) != len(axis):
            raise ValueError(f'Since axis is `None`, must provide '
                             f'window_shape for all dimensions of `x`; '
                             f'got {len(window_shape)} window_shape elements '
                             f'and `x.ndim` is {x.ndim}.')
    else:
        axis = normalize_axis_tuple(axis, x.ndim, allow_duplicate=True)
        if len(window_shape) != len(axis):
            raise ValueError(f'Must provide matching length window_shape and '
                             f'axis; got {len(window_shape)} window_shape '
                             f'elements and {len(axis)} axes elements.')

    # CHANGED THIS LINE ####
    # out_strides = x.strides + tuple(x.strides[ax] for ax in axis)
    # TO ###################
    out_strides = (x.strides[0]*stride, ) + tuple(x.strides[1:]) + x.strides
    ########################

    # note: same axis can be windowed repeatedly
    x_shape_trimmed = list(x.shape)
    for ax, dim in zip(axis, window_shape):
        if x_shape_trimmed[ax] < dim:
            raise ValueError(
                'window shape cannot be larger than input array shape')
        # CHANGED THIS LINE ####
        # x_shape_trimmed[ax] -= dim - 1
        # TO ###################
        x_shape_trimmed[ax] = int(np.ceil(
            (x_shape_trimmed[ax] - dim + 1) / stride))
        ########################
    out_shape = tuple(x_shape_trimmed) + window_shape
    return as_strided(x, strides=out_strides, shape=out_shape,
                      subok=subok, writeable=writeable)


def lineWiseCorrcoef(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute pearson correlation between `y` and all lines of `X`.

    Parameters
    ----------
    X : array_like, shape=(n, m)
        2D-array with each line corresponding to a 1D signal.
    y : array_like, shape=(m,)
        1D-array signal to compute correlation with.

    Returns
    -------
    ndarray, shape=(n,)
        1D-array of pearson correlation coefficients between `y` and each line
        of `X`.

    Notes
    -----
    This function is quite efficient through the use of einstein summation

    References
    ----------
    https://stackoverflow.com/questions/19401078/efficient-columnwise-correlation-coefficient-calculation.
    """
    # Make copies because arrays will be changed in place
    X = np.copy(X)
    y = np.copy(y)
    n = y.size
    DX = X - (np.einsum('ij->i', X) / np.double(n)).reshape((-1, 1))
    y -= (np.einsum('i->', y) / np.double(n))
    tmp = np.einsum('ij,ij->i', DX, DX)
    tmp *= np.einsum('i,i->', y, y)
    return np.dot(DX, y) / np.sqrt(tmp)


def vcorrcoef(X, Y):
    Xm = np.reshape(np.mean(X, axis=1), (X.shape[0], 1))
    Ym = np.reshape(np.mean(Y, axis=1), (Y.shape[0], 1))
    r_num = np.sum((X-Xm)*(Y-Ym), axis=1)
    r_den = np.sqrt(np.sum((X-Xm)**2, axis=1)*np.sum((Y-Ym)**2, axis=1))
    r = r_num/r_den
    return r


def moving_average(x, n=2):
    ret = np.cumsum(x)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def clip_to_nonzero_min(array):
    array[array == 0] = array[array != 0].min()
    return array


def nb_boolean_true_clusters(array: np.ndarray) -> int:
    """Compute the number of clusters or True values in array.

    Parameters
    ----------
    array : array_like
        1D-array of boolean values.

    Returns
    -------
    int
        number of clusters of True values
    """
    res = np.sum(np.diff(array)) // 2
    if array[0] or array[-1]:
        res += 1
    return res


def random_rounding(array: np.ndarray) -> np.ndarray:
    rounded = np.floor(array)
    decimal = array - rounded
    rounded += (np.random.rand(len(decimal)) <= decimal)
    return rounded


def integer_histogram_sample(array: np.ndarray, frac: float) -> np.ndarray:
    """Sample a random fraction of a histogram with integer-only values.

    The sampled histogram is a an array of integers of same shape as the
    original histogram, with all values smaller of equal to original histogram
    values.

    Parameters
    ----------
    array : array_like
        1D-array of integer values.
    frac : float
        fraction of the histogram to sample, the cumulative sum of the sampled
        histogram will be the rounded fraction of the original one

    Returns
    -------
    np.ndarray
        1D-array of same length as `array`, containing the sampled histogram
        values
    """
    positions = np.repeat(np.arange(array.size, dtype=int), array)
    rng = np.random.default_rng()
    if frac <= 0.5:
        sampled_pos = rng.choice(
            positions, size=round(len(positions)*frac), replace=False)
        histogram = coo_matrix(
            (np.ones(len(sampled_pos), dtype=int),
             (sampled_pos, np.zeros(len(sampled_pos), dtype=int))),
            shape=(len(array), 1)
        ).toarray().ravel()
        return histogram
    else:
        sampled_pos = rng.choice(
            positions, size=round(len(positions)*(1-frac)), replace=False)
        histogram = coo_matrix(
            (np.ones(len(sampled_pos), dtype=int),
             (sampled_pos, np.zeros(len(sampled_pos), dtype=int))),
            shape=(len(array), 1)
        ).toarray().ravel()
        return array - histogram


def integer_histogram_sample_vect(array: np.ndarray,
                                  frac: np.ndarray) -> np.ndarray:
    """Sample random fractions of a histogram with integer-only values.

    The sampled histogram is a an array of integers of same shape as the
    original histogram, with all values smaller of equal to original histogram
    values.
    This functions supports an array of fractions as input for some vectorized
    speed-ups. It seems 2x faster.

    Parameters
    ----------
    array : array_like, shape=(n,)
        1D-array of integer values.
    frac : np.ndarray, shape=(m,)
        1D-array of fractions of the histogram to sample, the cumulative sum
        of the sampled histograms will be the rounded fractions of the
        original one

    Returns
    -------
    np.ndarray, shape=(m, n)
        2D-array where each line is a sampled histogram with given fraction,
        and columns represent bins
    """
    positions = np.repeat(np.arange(array.size, dtype=int), array)
    rng = np.random.default_rng()
    sizes = np.array(np.round(len(positions)*frac), dtype=int)
    cumsizes = np.insert(np.cumsum(sizes), 0, 0)
    sampled_pos = np.zeros(cumsizes[-1], dtype=int)
    for i in range(len(frac)):
        sampled_pos[cumsizes[i]:cumsizes[i+1]] = rng.choice(
            positions, size=sizes[i], replace=False)
    histogram = coo_matrix(
        (np.ones(len(sampled_pos), dtype=int),
         (np.repeat(np.arange(len(frac)), sizes), sampled_pos)),
        shape=(len(frac), len(array))
    ).toarray()
    return histogram


# Other utils
def s_plural(value: float) -> str:
    """Return s if scalar value induces plural"""
    if value > 1:
        return 's'
    else:
        return ''


def safe_filename(file: Path) -> Path:
    """Make sure file can be build without overriding an other.

    If file already exists, returns a new filename with a number in between
    parenthesis.
    """
    # Build parent directories if needed
    Path(file.parent).mkdir(parents=True, exist_ok=True)
    # Change filename if it already exists
    file_dups = 0
    original_stem = file.stem
    while file.exists():
        file_dups += 1
        file = Path(file.parent,
                    original_stem + f'({file_dups})' + file.suffix)
        # in python 3.9, use file.with_stem(original_stem + f'({file_dups})')
    return file


def ram_usage() -> None:
    """Print RAM memory usage.

    References
    ----------
    https://www.geeksforgeeks.org/how-to-get-current-cpu-and-ram-usage-in-python/
    """
    # Getting all memory using os.popen()
    total_memory, used_memory, free_memory = map(
        int, os.popen('free -t -m').readlines()[-1].split()[1:])
    # Memory usage
    print("RAM memory % used:", round((used_memory/total_memory) * 100, 2))
