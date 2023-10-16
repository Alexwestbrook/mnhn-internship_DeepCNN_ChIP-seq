#!/usr/bin/env python
import os
from pathlib import Path
from collections import defaultdict
import itertools as it
import warnings
from typing import Callable, Optional, Union
import re
import json

import numpy as np
from numpy.lib.stride_tricks import as_strided
from numpy.core.numeric import normalize_axis_tuple
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
import scipy
from scipy import signal
from scipy.signal import gaussian, convolve
from scipy.stats import pearsonr
from scipy.sparse import coo_matrix

from statsmodels.stats import multitest

import pyBigWig
import pysam


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
    data = sliding_window_view(one_hot_chr, (window_size, 4)).squeeze()
    if remove_Ns:
        indexes = remove_windows_with_N(one_hot_chr, window_size)
        data = data[indexes]
    else:
        indexes = np.arange(len(data))
    return indexes, data


def merge_chroms(chr_ids, file):
    annot = []
    with np.load(file) as f:
        for chr_id in chr_ids:
            annot.append(f[chr_id])
            shape, dtype = f[chr_id].shape, f[chr_id].dtype
            annot.append(np.zeros((1,)+shape[1:], dtype=dtype))
    return np.concatenate(annot)


def chunk_regions(array, length):
    leftover = len(array) % length
    if leftover != 0:
        array = array[:-leftover]
    return array.reshape((-1, length) + array.shape[1:])


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
def one_hot_encode(seq, length=None, one_hot_type=bool, order='ACGT'):
    if length is None:
        length = len(seq)
    one_hot = np.zeros((length, 4), dtype=one_hot_type)
    for i, base in enumerate(seq):
        if i >= length:
            break
        if base == order[0]:
            one_hot[i, 0] = 1
        elif base == order[1]:
            one_hot[i, 1] = 1
        elif base == order[2]:
            one_hot[i, 2] = 1
        elif base == order[3]:
            one_hot[i, 3] = 1
    return one_hot


def one_hot_decode(one_hot, order='ACGT'):
    if len(one_hot.shape) != 2:
        raise ValueError(
            'input must be a single one hot encoded read')
    if order == 'ACGT':
        categories = np.array(list('ACGT')).reshape(-1, 1)
        encoder = OneHotEncoder(dtype=one_hot.dtype,
                                handle_unknown='ignore',
                                sparse=False)
        encoder.fit(categories)

        seq = encoder.inverse_transform(one_hot)
        seq = seq.ravel()
        seq = ''.join(['N' if value is None else value for value in seq])
        return seq
    else:
        bases = np.array(list(order))
        seq = bases[np.argmax(one_hot, axis=1)]
        seq[np.sum(one_hot, axis=1) != 1] = 'N'
        return ''.join(seq)


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


def one_hot_to_seq(reads, order='ACGT'):
    if order == 'ACGT':
        return one_hot_to_seq_v2(reads)
    else:
        return one_hot_to_seq_v1(reads, order)


def one_hot_to_seq_v1(reads, order='ACGT'):
    """
    Convert one_hot array of reads into list of sequences.
    """
    if len(reads.shape) != 3:
        raise ValueError('must be an array of one hot encoded reads')
    bases = np.array(list(order))
    seqs = bases[np.argmax(reads, axis=2)]
    seqs[np.sum(reads, axis=2) != 1] = 'N'
    seqs = [''.join([char for char in seq]) for seq in seqs]
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


def np_idx_to_one_hot(idx, order='ACGT', extradims=None):
    """Convert array of indexes into one-hot in np format.

    Parameters
    ----------
    idx : ndarray
        Array of indexes into 'ACGT'
    order : str, default='ACGT'
        String representation of the order in which to encode bases. Default
        value of 'ACGT' means that A has the representation with 1 in first
        position, C with 1 in second position, etc...
    extradims : int or list of int
        Extra dimensions to give to the one_hot, which by default is of shape
        idx.shape + (4,). If extradims is an array there will be an error.

    Returns
    -------
    ndarray
        Array with same shape as idx, in one-hot format.
    """
    assert (len(order) == 4 and set(order) == set('ACGT'))
    converter = np.zeros((4, 4), dtype=bool)
    for i, c in enumerate('ACGT'):
        converter[i, order.find(c)] = 1
    one_hot = converter[idx]
    if extradims is not None:
        one_hot = np.expand_dims(one_hot, axis=extradims)
    return one_hot


# Sequence manipulation
def remove_reads_with_N(sequences,
                        tolerance=0,
                        max_size=None,
                        read_length=None,
                        verbose=False):
    if max_size is not None:
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
                assert start_count >= 0
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
        if isinstance(seqs, dict):
            iterable = seqs.items()
        else:
            iterable = enumerate(seqs)
        for id, seq in iterable:
            f.write(f'>{id}\n')
            if wrap is not None:
                for i in range(0, len(seq), wrap):
                    f.write(f'{seq[i:i + wrap]}\n')
            else:
                f.write(f'{seq}\n')


def parse_bed_peaks(bed_file,
                    window_size=None,
                    remove_duplicates=False,
                    based1=False):
    # compute offset to adjust 1-based bed indices to 0-based chromosome
    # indices, or predictions with given window
    offset = 0
    if based1:
        offset += 1
    if window_size is not None:
        offset += window_size // 2
    with open(bed_file, 'r') as f:
        chr_peaks = {}
        for line in f:
            line = line.rstrip()
            chr_id, start, end, _, score, *_ = line.split('\t')
            start, end, score = int(start), int(end), int(score)
            if chr_id in chr_peaks.keys():
                chr_peaks[chr_id].append(np.array([start, end, score]))
            else:
                chr_peaks[chr_id] = [np.array([start, end, score])]
        for key in chr_peaks.keys():
            # convert to array, remove duplicates
            if remove_duplicates:
                chr_peaks[key] = np.unique(np.array(chr_peaks[key]), axis=0)
            # Adjust indices
            chr_peaks[key] = (np.asarray(chr_peaks[key])
                              - np.array([1, 1, 0]) * offset)
            try:
                # Check if some peaks overlap
                overlaps = self_overlapping_peaks(chr_peaks[key])
                assert len(overlaps) == 0
            except AssertionError:
                print(f'Warning: some peaks overlap in {key}')
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


def inspect_bam_mapq(bam_file):
    mapqs = defaultdict(int)
    with pysam.AlignmentFile(bam_file, 'rb') as f:
        for read in f.fetch():
            mapqs[read.mapping_quality] += 1
    return dict(sorted(mapqs.items()))


def load_bw(filename, nantonum=True):
    labels = {}
    bw = pyBigWig.open(str(filename))
    for chr_id in bw.chroms():
        if nantonum:
            labels[chr_id] = np.nan_to_num(
                bw.values(chr_id, 0, -1, numpy=True))
        else:
            labels[chr_id] = bw.values(chr_id, 0, -1, numpy=True)
    bw.close()
    return labels


def write_bw(filename, signals):
    bw = pyBigWig.open(str(filename), 'w')
    bw.addHeader([(k, len(v)) for k, v in signals.items()])
    for chr_id, val in signals.items():
        bw.addEntries(chr_id, 0, values=val, span=1, step=1)
    bw.close()


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
def GC_content(one_hot_reads: np.ndarray, order: int = 'ACGT') -> np.ndarray:
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
    if one_hot_reads.ndim == 2:
        one_hot_reads = np.expand_dims(one_hot_reads, axis=0)
    assert one_hot_reads.ndim == 3 and one_hot_reads.shape[-1] == 4
    # Compute content of each base
    content = np.sum(one_hot_reads, axis=1)  # shape (nb_reads, 4)
    g_idx, c_idx = order.find('G'), order.find('C')
    gc = (content[:, g_idx] + content[:, c_idx]) / np.sum(content, axis=1)
    return gc


def sliding_GC(one_hot, n, order='ACGT'):
    valid_mask = one_hot.sum(axis=1) != 0
    GC_idx = [order.find('G'), order.find('C')]
    GC_mask = one_hot[:, GC_idx].sum(axis=1)
    return moving_sum(GC_mask, n=n) / moving_sum(valid_mask, n=n)


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
        2D-array of coordinates for start and end (included) of each fragment.
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


def exact_alignment_count_from_coord(coord: np.ndarray,
                                     length: int = None) -> np.ndarray:
    """
    Build alignment count signal from read coordinates on a single chromosome.

    Parameters
    ----------
    coord : ndarray, shape=(nb_reads, 2)
        2D-array of coordinates for start and end (included) of each fragment.

    Returns
    -------
    ndarray
        1D-array of read count for each position along the chromosome.
    """
    # Get coordinate of first bp after fragment end
    coord = coord.copy()
    coord[:, 1] += 1
    if length is None:
        length = np.max(coord)
    # Insert +1 at fragment start and -1 after fragment end
    data = np.ones(2*len(coord), dtype=int)
    data[1::2] = -1
    # Insert using scipy.sparse implementation
    start_ends = coo_matrix(
        (data, (coord.ravel(), np.zeros(2*len(coord), dtype=int))),
        shape=(length + 1, 1)  # length+1 because need a -1 after last end
    ).toarray().ravel()
    # Cumulative sum to propagate full fragments,
    # remove last value which is always 0
    return np.cumsum(start_ends)[:-1]


def bin_values(values: np.ndarray, binsize: int) -> np.ndarray:
    if len(values) % binsize == 0:
        binned_values = np.mean(
            strided_window_view(values, binsize, binsize), axis=1)
    else:
        binned_values = np.append(
            np.mean(strided_window_view(values, binsize, binsize), axis=1),
            np.mean(values[-(len(values) % binsize):]))
    return binned_values


def full_genome_binned_preds(pred_file,
                             chr_sizes_file,
                             binsize,
                             chr_ids):
    with open(chr_sizes_file, 'r') as f:
        chr_lens = json.load(f)
    binned_lengths = np.array([x // binsize + 1 for x in chr_lens.values()])
    separators = np.cumsum(binned_lengths)
    df = np.zeros(separators[-1])
    # merging chromosomes
    for i, chr_id in enumerate(chr_ids.keys()):
        with np.load(pred_file) as f:
            preds = f[f'chr{chr_id}']
        binned_preds = bin_values(preds, binsize)
        df[separators[i]-len(binned_preds):separators[i]] = binned_preds
    return df, separators


def enrichment_analysis(signal, ctrl, verbose=True, data='signal'):
    n_binom = signal + ctrl
    p_binom = np.sum(signal) / np.sum(n_binom)
    binom_pval = clip_to_nonzero_min(
        1 - scipy.stats.binom.cdf(signal - 1, n_binom, p_binom))
    reject, binom_qval, *_ = multitest.multipletests(
        binom_pval, method='fdr_bh')
    signif_qval = reject
    if verbose:
        print(f'{np.sum(reject)}/{len(reject)} '
              f'significantly enriched bins in {data}')
    return pd.DataFrame({
        "pval": binom_pval,
        "qval": binom_qval,
        "-log_pval": -np.log10(binom_pval),
        "-log_qval": -np.log10(binom_qval),
        "signif_qval": signif_qval})


def genome_enrichment(ip_coord_file,
                      ctrl_coord_file,
                      chr_sizes_file,
                      out_file,
                      binsize,
                      max_frag_len=500):
    def process_coord_file(coord_file):
        """Pipeline for both ip and ctrl coord files"""
        try:
            # Load chromosome fragment coordinates
            with np.load(coord_file) as f:
                coord_chr = f[chr_id]
            # Filter out fragments too long
            frag_lens_chr = np.diff(coord_chr, axis=1).ravel()
            coord_chr = coord_chr[frag_lens_chr <= max_frag_len, :]
            # Get binned count of mid points
            count_chr = binned_alignment_count_from_coord(
                coord_chr, binsize=binsize, length=chr_lens[chr_id])
            with open(log_file, 'a') as f:
                f'{np.sum(frag_lens_chr >= max_frag_len)} fragments '
                f'longer than {max_frag_len}bp in {coord_file}\n'
        except KeyError:
            count_chr = np.zeros(chr_lens[chr_id] // binsize + 1)
            with open(log_file, 'a') as f:
                f'No reads in {coord_file}\n'
        return count_chr

    # Log parameters
    out_file = Path(out_file)
    out_file = safe_filename(out_file)
    log_file = Path(out_file.parent, out_file.stem + '_log.txt')
    log_file = safe_filename(log_file)
    with open(log_file, 'w') as f:
        f.write(f'IP coordinates file: {ip_coord_file}\n'
                f'Control coordinates file: {ctrl_coord_file}\n'
                f'chromosome sizes file: {chr_sizes_file}\n'
                f'output file: {out_file}\n'
                f'bin size: {binsize}\n'
                f'max fragment length: {max_frag_len}\n\n')
    # Get chromosome lengths
    with open(chr_sizes_file, 'r') as f:
        chr_lens = json.load(f)
    # Build DataFrame
    mindex = pd.MultiIndex.from_tuples(
        [(chr_id, pos)
         for chr_id in chr_lens.keys()
         for pos in np.arange(0, chr_lens[chr_id], binsize)],
        names=['chr', 'pos']
    )
    columns = ['ip_count', 'ctrl_count', 'pval', 'qval']
    df = pd.DataFrame(0, index=mindex, columns=columns)
    # Loop over chromosomes
    for chr_id in chr_lens.keys():
        with open(log_file, 'a') as f:
            f.write(f'Processing {chr_id}...\n')
        ip_count_chr = process_coord_file(ip_coord_file)
        ctrl_count_chr = process_coord_file(ctrl_coord_file)
        # Insert in DataFrame
        df.loc[chr_id, :'ctrl_count'] = np.transpose(
            np.vstack((ip_count_chr, ctrl_count_chr)))
    # Compute p-values and q-values
    n_binom = df['ip_count'] + df['ctrl_count']
    p_binom = np.sum(df['ip_count']) / np.sum(n_binom)
    df['pval'] = clip_to_nonzero_min(
        1 - scipy.stats.binom.cdf(df['ip_count'] - 1, n_binom, p_binom))
    _, df['qval'], *_ = multitest.multipletests(df['pval'], method='fdr_bh')
    # Save final DataFrame
    df.to_csv(out_file)


def downsample_enrichment_analysis(data,
                                   genome,
                                   max_frag_len,
                                   binsizes=[1000],
                                   fracs=[1],
                                   divs=None,
                                   reverse=True,
                                   data_dir='../shared_folder',
                                   basename='',
                                   use_fdr=True):
    # Convert divs to fracs
    if divs is not None:
        fracs = 1 / np.array(fracs)
    # Build resulting DataFrame
    mindex = pd.MultiIndex.from_product([binsizes, fracs])
    if reverse:
        res = pd.DataFrame(
            index=mindex,
            columns=['IP', 'IP_clust', 'Undetermined', 'Ctrl', 'Ctrl_clust',
                     'total_cov'])
    else:
        res = pd.DataFrame(
            index=mindex,
            columns=['IP', 'IP_clust', 'Undetermined', 'total_cov'])
    # Start analysis
    for binsize in binsizes:
        # Load alignment data
        df = pd.read_csv(
            Path(data_dir, data, 'results', 'alignments', genome,
                 f'{data}_{genome}_{basename}maxfraglen_{max_frag_len}_'
                 f'binsize_{binsize}.csv'),
            index_col=0)
        for frac in fracs:
            # Randomly sample alignment histogram
            frac_IP = integer_histogram_sample(df["ip_count"], frac)
            frac_Ctrl = integer_histogram_sample(df["ctrl_count"], frac)
            # Compute p-values
            n = frac_IP + frac_Ctrl
            cov = np.sum(n)
            p_binom = np.sum(frac_IP) / cov
            pval = clip_to_nonzero_min(
                1 - scipy.stats.binom.cdf(frac_IP - 1, n, p_binom))
            # Extract significant IP bins
            if use_fdr:
                # correct with q-value on non-empty bins
                valid_bins = (n != 0)
                signif_IP = np.zeros(len(df), dtype=bool)
                signif_IP[valid_bins], *_ = multitest.multipletests(
                    pval[valid_bins], method='fdr_bh')
            else:
                signif_IP = np.array(pval < 0.05)
            n_signif_IP = np.sum(signif_IP)
            n_signif_IP_clust = nb_boolean_true_clusters(signif_IP)
            # Extract significant Ctrl bins too
            if reverse:
                rev_pval = clip_to_nonzero_min(
                    1 - scipy.stats.binom.cdf(frac_Ctrl - 1, n, 1 - p_binom))
                if use_fdr:
                    signif_Ctrl = np.zeros(len(df), dtype=bool)
                    signif_Ctrl[valid_bins], *_ = multitest.multipletests(
                        rev_pval[valid_bins], method='fdr_bh')
                else:
                    signif_Ctrl = np.array(rev_pval < 0.05)
                n_signif_Ctrl = np.sum(signif_Ctrl)
                n_signif_Ctrl_clust = nb_boolean_true_clusters(signif_Ctrl)
            # Save results
            if reverse:
                res.loc[binsize, frac] = [
                    n_signif_IP,
                    n_signif_IP_clust,
                    len(df) - n_signif_IP - n_signif_Ctrl,
                    n_signif_Ctrl,
                    n_signif_Ctrl_clust,
                    cov]
            else:
                res.loc[binsize, frac] = [
                    n_signif_IP,
                    n_signif_IP_clust,
                    len(df) - n_signif_IP,
                    cov]
    return res


def pool_experiments(dfs, verbose=True):
    cols_to_take = ['ip_count', 'ctrl_count']
    df_pooled = dfs[0][['pos'] + cols_to_take].copy()
    for df in dfs[1:]:
        df_pooled[cols_to_take] += df[cols_to_take]
    # computing p_value and q_value
    sums = df_pooled.sum(axis=0)
    p_binom = sums["ip_count"] / (sums["ip_count"] + sums["ctrl_count"])
    n_binom = df_pooled["ip_count"] + df_pooled["ctrl_count"]
    df_pooled['pval'] = clip_to_nonzero_min(
        1 - scipy.stats.binom.cdf(df_pooled["ip_count"] - 1, n_binom, p_binom))
    reject, df_pooled['qval'], *_ = multitest.multipletests(
        df_pooled["pval"], method='fdr_bh')
    df_pooled['-log_qval'] = -np.log10(df_pooled['qval'])
    df_pooled['-log_pval'] = -np.log10(df_pooled["pval"])
    df_pooled['signif_qval'] = reject
    if verbose:
        print(f'{np.sum(reject)}/{len(reject)} '
              f'significantly enriched bins in dataframe')
    return df_pooled


def adapt_to_bins(df, df_ref):
    """
    Assign df values to each corresponding bin of df_ref.

    df's binsize must be an integer multiple of df_ref's
    """
    # Input checks
    for _df in [df, df_ref]:
        try:
            assert isinstance(_df, pd.DataFrame)
            assert isinstance(_df.index, pd.MultiIndex)
            assert _df.index.nlevels == 2
            assert _df.index.levels[1].size > 1
        except AssertionError:
            print("Arguments must be DataFrames using MultiIndex with 2 "
                  "levels, and second level must have at least 2 values")
            raise
    try:
        assert np.all(df.index.levels[0] == df_ref.index.levels[0])
    except AssertionError:
        print("Both DataFrames must have the same first levels")
        raise
    # Get and check binsizes
    binsize = df.index.levels[1][1]
    binsize_ref = df_ref.index.levels[1][1]
    try:
        assert binsize >= binsize_ref
        assert binsize % binsize_ref == 0
    except AssertionError:
        print("df's binsize must be an integer multiple of df_ref's")
        raise
    # repeat df `scale` times
    scale = binsize // binsize_ref
    newdf = df.reindex(df.index.repeat(scale))
    # reset indices with binsize_ref intervals
    newdf.index = pd.MultiIndex.from_tuples(
        [(chr_id, pos)
         for chr_id in df.index.levels[0]
         for pos in range(0, df.loc[chr_id].index.max()+binsize, binsize_ref)],
        names=['chr', 'pos']
    )
    # reindex according to df_ref to discard unnecessary trailing indices
    return newdf.reindex(df_ref.index)


def adapt_to_bins_old(df, df_ref, binsize, binsize_ref):
    """
    The two DataFrames must be of the same genome with same number of
    chromosomes so that the scaled separators match within `scale` indices.
    Binsizes must be integer multiples of each other.
    """
    # get separators
    seps = np.append(np.where(df['pos'] == 0)[0], len(df))
    seps_ref = np.append(np.where(df_ref['pos'] == 0)[0], len(df_ref))
    # get binsize scale
    scale = binsize // binsize_ref
    # repeat df2 `scale` times and reset indices
    subdf = df.reindex(df.index.repeat(scale))
    subdf = subdf.reset_index().drop('index', axis=1)
    # cut chromosome edges to conform to df1
    cumul = 0
    for i in range(1, len(seps_ref)):
        subdf.drop(range(seps_ref[i] + cumul, seps[i] * scale), inplace=True)
        cumul = seps[i]*scale - seps_ref[i]
    # reset indices again
    return subdf.reset_index().drop('index', axis=1)


def make_mindex_ser(annotation_file: Path,
                    chr_sizes_file: Path,
                    out_file: Path,
                    binsize: int,
                    name: str = None,
                    coords: bool = False,
                    process_func: Callable = None,
                    annot_ids_dict: dict = None,
                    **kwargs):
    """Build a binned MultiIndex Series from an annotation file."""
    # Log parameters
    out_file = safe_filename(out_file)
    log_file = Path(out_file.parent, out_file.stem + '_log.txt')
    log_file = safe_filename(log_file)
    with open(log_file, 'w') as f:
        f.write(f'annotation file: {annotation_file}\n'
                f'chromosome sizes file: {chr_sizes_file}\n'
                f'output file: {out_file}\n'
                f'bin size: {binsize}\n'
                f'coords: {coords}\n'
                f'process function: {process_func}\n'
                f'annot_ids_dict: {annot_ids_dict}\n\n')
    # Get chromosome lengths
    with open(chr_sizes_file, 'r') as f:
        chr_lens = json.load(f)
    # Build MultiIndex
    mindex = pd.MultiIndex.from_tuples(
        [(chr_id, pos)
         for chr_id, chr_len in chr_lens.items()
         for pos in np.arange(0, chr_len, binsize)],
        names=['chr', 'pos']
    )
    ser = pd.Series(0, index=mindex, name=name)
    # Loop over chromosomes
    for chr_id, chr_len in chr_lens.items():
        with open(log_file, 'a') as f:
            f.write(f'Processing {chr_id}...\n')
        with np.load(annotation_file) as f:
            try:
                if annot_ids_dict is not None:
                    annot_chr = f[annot_ids_dict[chr_id]]
                else:
                    annot_chr = f[chr_id]
                if process_func is not None:
                    # Process annotations before binning
                    annot_chr = process_func(annot_chr, log_file, **kwargs)
                if coords:
                    # Get binned count of mid points
                    annot_chr = binned_alignment_count_from_coord(
                        annot_chr, binsize=binsize, length=chr_len)
                else:
                    annot_chr = bin_values(annot_chr, binsize)
            except KeyError:
                annot_chr = np.zeros(chr_len // binsize + 1)
                with open(log_file, 'a') as f:
                    f.write(
                        f'No annotation for {chr_id} in {annotation_file}\n')
        # Insert in Series
        ser.loc[chr_id] = annot_chr
    ser.to_csv(out_file)


def sliding_correlation(X, Y, offsets):
    slide_corr = []
    for i in offsets:
        if i == 0:
            corr = pearsonr(X, Y)[0]
        elif i > 0:
            corr = pearsonr(X[i:], Y[:-i])[0]
        else:
            corr = pearsonr(Y[-i:], X[:i])[0]
        slide_corr.append(corr)
    return slide_corr


def fast_sliding_correlation(X, Y, offsets):
    """Higher memory footprint, will crash with too much data"""
    min_offset = np.min(offsets)
    max_offset = np.max(offsets)
    max_len = len(X) - max_offset + min_offset
    offsets -= min_offset
    windows = offsets.reshape(-1, 1) + np.arange(max_len).reshape(1, -1)
    X_slides = X[windows]
    slide_corrs = lineWiseCorrcoef(
        X_slides, Y[-min_offset:-min_offset + max_len])
    return slide_corrs


def best_cor_lag(x, y, mode="full"):
    """
    Return the offset which leads to the highest correlation between two
    signals.
    """
    # Not a true
    correlation = signal.correlate(x, y, mode="full")
    lags = signal.correlation_lags(x.size, y.size, mode="full")
    return lags[np.argmax(correlation)]


# Peak manipulation
def make_peaks(peak_mask: np.ndarray,
               length_thres: int = 1,
               tol: int = 1) -> np.ndarray:
    """Format peak array from peak boolean mask.

    Determine regions of consecutive high prediction, called peaks.

    Parameters
    ----------
    peak_mask : ndarray
        1D-array of boolean values along the chromosome
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
    return make_peaks(peak_mask, length_thres, tol)


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
                           merge: bool = False,
                           tol: int = 1
                           ) -> tuple:
    # tuple(np.ndarray, Optional(np.ndarray)):
    """Determine which peaks within the array overlap

    As opposed to `overlapping_peaks`, here two disjoint but adjacent peaks
    will be considered self-overlapping since they can be merged into one
    contiguous peak.

    Parameters
    ----------
    peaks : ndarray, shape=(n, m)
        2D-array, each line corresponds to a peak. A peak is a 1D-array of
        size m0 = 2 or 3, with format [peak_start, peak_end, *optional_score].
        `peak_start` and `peak_end` must be indices on the chromosome.
        Peaks must be disjoint within a 2D-array, meaning that there is no
        other peak starting or ending between `peak_start` and `peak_end`.
    merge : bool, default=False
        True indicates to return an array with overlapping peaks merged. False
        indicates to not perform this operation, which is faster
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


def strided_window_view(x, window_shape, stride,
                        axis=None, *, subok=False, writeable=False):
    """Variant of `sliding_window_view` which supports stride parameter.

    The axis parameter doesn't work, the stride can be of same shape as
    window_shape, providing different stride in each dimension. If shorter
    than window_shape, stride will be filled with ones. it also doesn't
    support multiple windowing on same axis"""
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

    # ADDED THIS ####
    stride = (tuple(stride)
              if np.iterable(stride)
              else (stride,))
    stride_array = np.array(stride)
    if np.any(stride_array < 0):
        raise ValueError('`stride` cannot contain negative values')
    if len(stride) > len(window_shape):
        raise ValueError('`stride` cannot be longer than `window_shape`')
    elif len(stride) < len(window_shape):
        stride += (1,) * (len(window_shape) - len(stride))
    ########################

    # CHANGED THIS LINE ####
    # out_strides = x.strides + tuple(x.strides[ax] for ax in axis)
    # TO ###################
    out_strides = (tuple(x.strides[ax]*stride[ax] for ax in range(x.ndim))
                   + tuple(x.strides[ax] for ax in axis))
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
            (x_shape_trimmed[ax] - dim + 1) / stride[ax]))
        ########################
    out_shape = tuple(x_shape_trimmed) + window_shape
    return as_strided(x, strides=out_strides, shape=out_shape,
                      subok=subok, writeable=writeable)


def strided_sliding_window_view(x, window_shape, stride, sliding_len,
                                axis=None, *, subok=False, writeable=False):
    """Variant of `strided_window_view` which slides in between strides.

    This will provide blocks of sliding window of `sliding_len` windows,
    with first windows spaced by `stride`
    The axis parameter determines where the stride and slide are performed, it
    can only be a single value."""
    window_shape = (tuple(window_shape)
                    if np.iterable(window_shape)
                    else (window_shape,))
    # first convert input to array, possibly keeping subclass
    x = np.array(x, copy=False, subok=subok)

    window_shape_array = np.array(window_shape)
    if np.any(window_shape_array < 0):
        raise ValueError('`window_shape` cannot contain negative values')

    # ADDED THIS ####
    stride = (tuple(stride)
              if np.iterable(stride)
              else (stride,))
    stride_array = np.array(stride)
    if np.any(stride_array < 0):
        raise ValueError('`stride` cannot contain negative values')
    if len(stride) == 1:
        stride += (1,)
    elif len(stride) > 2:
        raise ValueError('`stride` cannot be of length greater than 2')
    if sliding_len % stride[1] != 0:
        raise ValueError('second `stride` must divide `sliding_len` exactly')
    # CHANGED THIS ####
    # if axis is None:
    #     axis = tuple(range(x.ndim))
    #     if len(window_shape) != len(axis):
    #         raise ValueError(f'Since axis is `None`, must provide '
    #                          f'window_shape for all dimensions of `x`; '
    #                          f'got {len(window_shape)} window_shape '
    #                          f'elements and `x.ndim` is {x.ndim}.')
    # else:
    #     axis = normalize_axis_tuple(axis, x.ndim, allow_duplicate=True)
    #     if len(window_shape) != len(axis):
    #         raise ValueError(f'Must provide matching length window_shape '
    #                          f'and axis; got {len(window_shape)} '
    #                          f'window_shape elements and {len(axis)} axes '
    #                          f'elements.')
    # TO ###################
    if axis is None:
        axis = 0
    ########################

    # CHANGED THIS LINE ####
    # out_strides = ((x.strides[0]*stride, )
    #                + tuple(x.strides[1:])
    #                + tuple(x.strides[ax] for ax in axis))
    # TO ###################
    out_strides = (x.strides[:axis]
                   + (x.strides[axis]*stride[0], x.strides[axis]*stride[1])
                   + x.strides[axis:])
    ########################

    # CHANGED THIS ####
    # note: same axis can be windowed repeatedly
    # x_shape_trimmed = list(x.shape)
    # for ax, dim in zip(axis, window_shape):
    #     if x_shape_trimmed[ax] < dim:
    #         raise ValueError(
    #             'window shape cannot be larger than input array shape')
    #     x_shape_trimmed[ax] = int(np.ceil(
    #         (x_shape_trimmed[ax] - dim + 1) / stride))
    # out_shape = tuple(x_shape_trimmed) + window_shape
    # TO ###################
    x_shape_trimmed = [(x.shape[axis]
                        - window_shape[axis]
                        - sliding_len + stride[1]) // stride[0] + 1,
                       sliding_len // stride[1]]
    out_shape = (window_shape[:axis]
                 + tuple(x_shape_trimmed)
                 + window_shape[axis:])
    ########################
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


def continuousjaccard(x, y):
    """Compute continuous Jaccard index.

    Parameters
    ----------
    x, y : array_like
        1D or 2D array_likes. x and y must have the same shape, unless one of
        them is 1D and the other is 2D, in which case their last axis must
        have the same shape.

    Returns
    -------
    ndarray
        Row-wise Jaccard index between x and y. If the dimensions aren't the
        same, each row of the 2D array is compared to the 1D array
    """
    x, y = np.asarray(x), np.asarray(y)
    assert {x.ndim, y.ndim}.union({1, 2}) == {1, 2}
    if x.ndim != y.ndim:
        if x.ndim == 1:
            x, y = y, x
        y = np.tile(y, len(x)).reshape(x.shape)
    assert x.shape == y.shape
    merge = np.vstack([x, y]).reshape((2,) + x.shape)
    sign = np.prod(np.sign(merge), axis=0)
    merge = np.abs(merge)
    return np.sum(np.min(merge, axis=0) * sign, axis=-1
                  ) / np.sum(np.max(merge, axis=0), axis=-1)


def moving_average(x, n=2, keepsize=False):
    if keepsize:
        x = np.concatenate([
            np.zeros(n // 2, dtype=x.dtype),
            x,
            np.zeros((n-1) // 2, dtype=x.dtype)])
    ret = np.cumsum(x)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def moving_sum(x, n=2, axis=None):
    ret = np.cumsum(x, axis)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:]


def repeat_along_diag(a, r):
    """Construct a matrix by repeating a sub_matrix along the diagonal.

    References
    ----------
    https://stackoverflow.com/questions/33508322/create-block-diagonal-numpy-array-from-a-given-numpy-array
    """
    m, n = a.shape
    out = np.zeros((r, m, r, n), dtype=a.dtype)
    diag = np.einsum('ijik->ijk', out)
    diag[:] = a
    return out.reshape(-1, n*r)


def exp_normalize(x, axis=-1):
    res = np.exp(x - np.max(x))
    return res / np.sum(res, axis=axis, keepdims=True)


def simple_slice(arr, slc, axis):
    """Take a slice along an axis of an array.

    Parameters
    ----------
    arr: ndarray
        Array to take the slice from
    slc: slice
        Slice to take from the array
    axis: int
        axis along which to perform the slicing

    Returns
    -------
    ndarray
        A sliced view into the original array

    References
    ----------
    https://stackoverflow.com/questions/24398708/slicing-a-numpy-array-along-a-dynamically-specified-axis
    """
    # this does the same as np.take() except only supports simple slicing, not
    # advanced indexing, and thus is much faster
    full_slc = [slice(None)] * arr.ndim
    full_slc[axis] = slc
    return arr[tuple(full_slc)]


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


def indices_from_starts_ends(starts, ends):
    # ends must be excluded, ends-starts>0
    lens = ends - starts
    np.cumsum(lens, out=lens)
    i = np.ones(lens[-1], dtype=int)
    i[0] = starts[0]
    i[lens[:-1]] += starts[1:]
    i[lens[:-1]] -= ends[:-1]
    np.cumsum(i, out=i)
    return i


def indices_from_peaks(peaks):
    # ends must be excluded, ends-starts>0
    return indices_from_starts_ends(peaks[:, 0], peaks[:, 1])


# Random sequences generation
def kmer_counts(one_hots, k, order='ACGT', includeN=True, as_pandas=True):
    """Compute kmer occurences in one-hot encoded sequence."""
    # Convert input into list-like of one_hot 2D-arrays
    # If 3D-array optionnally use faster implementation
    fast = False
    if isinstance(one_hots, dict):
        one_hots = list(one_hots.values())
    elif isinstance(one_hots, np.ndarray):
        if one_hots.ndim == 2:
            # single array turned into list of one array
            one_hots = [one_hots]
        elif one_hots.ndim == 3:
            # Check that last dimension is 4
            assert one_hots.shape[2] == 4
            fast = True
    if fast:  # Faster on 3D array
        # Initialise kD array
        all_counts = np.zeros(tuple(5 for i in range(k)), dtype=int)
        if k == 1:
            # Count each base
            all_counts[:4] = one_hots.sum(axis=(0, 1))
            # Count leftover as Ns
            all_counts[4] = (len(one_hots) * one_hots.shape[1]
                             - all_counts[:4].sum())
        else:
            # Convert one_hot to integer tokens
            tokens = (np.argmax(one_hots, axis=-1)
                      + 4 * (np.sum(one_hots, axis=-1) != 1))
            # Get kmers with sliding_window_view
            kmers = sliding_window_view(tokens, (1, k)).reshape(-1, k)
            # Count kmers in the kD array
            np.add.at(all_counts, tuple(kmers[:, i] for i in range(k)), 1)
    else:  # Iterate over one-hot encoded arrays
        # Initialise kD array
        all_counts = np.zeros(tuple(5 for i in range(k)), dtype=int)
        for oh in one_hots:
            # Check that arrays are 2D with a shape of 4 in the 2nd dimension
            assert oh.ndim == 2
            assert oh.shape[1] == 4
            if k == 1:
                # Count each base
                all_counts[:4] += oh.sum(axis=0)
                # Count leftover as Ns
                all_counts[4] += len(oh) - oh.sum()
            else:
                # Convert one_hot to integer tokens
                tokens = np.argmax(oh, axis=-1) + 4*(np.sum(oh, axis=-1) != 1)
                # Get kmers with sliding_window_view
                kmers = sliding_window_view(tokens, k)
                # Count kmers in the kD array
                np.add.at(all_counts, tuple(kmers[:, i] for i in range(k)), 1)
    # Format output
    if includeN:
        order += 'N'
    else:
        all_counts = all_counts[tuple(slice(0, -1) for i in range(k))]
    if as_pandas:
        ser = pd.Series(
            all_counts.ravel(),
            index=pd.MultiIndex.from_product([list(order)]*k))
        return ser.sort_index()
    else:
        return all_counts


def kmer_counts_by_seq(one_hots, k, order='ACGT', includeN=True,
                       as_pandas=True):
    assert one_hots.ndim == 3
    # Initialise kD array
    all_counts = np.zeros(tuple(5 for i in range(k)) + (len(one_hots),),
                          dtype=int)
    if k == 1:
        # Count each base
        all_counts[:4] = one_hots.sum(axis=1).T
        # Count leftover as Ns
        all_counts[4] = one_hots.shape[1] - all_counts[:4].sum(axis=0)
    else:
        # Convert one_hot to integer tokens
        tokens = (np.argmax(one_hots, axis=-1)
                  + 4 * (np.sum(one_hots, axis=-1) != 1))
        for i, arr in enumerate(tokens):
            # Get kmers with sliding_window_view
            kmers = sliding_window_view(arr, k)
            # Count kmers in the kD array
            np.add.at(all_counts,
                      tuple(kmers[:, j] for j in range(k)) + (i,),
                      1)
    if includeN:
        order += 'N'
    else:
        all_counts = all_counts[tuple(slice(0, -1) for i in range(k))
                                + (slice(None),)]
    if as_pandas:
        ser = pd.DataFrame(
            all_counts.reshape(len(order)**k, -1),
            index=pd.MultiIndex.from_product([list(order)]*k))
        return ser.sort_index()
    else:
        return all_counts


def sliding_kmer_counts(one_hot, k, winsize, order='ACGT', includeN=True,
                        as_pandas=True):
    assert one_hot.ndim == 2
    n_windows = len(one_hot) - winsize + 1
    # Initialise kD array
    all_counts = np.zeros(tuple(5 for i in range(k)) + (n_windows,),
                          dtype=int)
    if k == 1:
        # Count each base
        all_counts[:4] = moving_sum(one_hot, winsize, axis=0).T
        # Count leftover as Ns
        all_counts[4] = winsize - all_counts[:4].sum(axis=0)
    else:
        # Convert one_hot to integer tokens
        tokens = (np.argmax(one_hot, axis=-1)
                  + 4 * (np.sum(one_hot, axis=-1) != 1))
        # Get kmers with sliding_window_view
        kmers = sliding_window_view(tokens, k)
        # Count kmers in first window in the kD array
        np.add.at(
            all_counts,
            tuple(kmers[:winsize+1-k, j] for j in range(k)) + (0,),
            1)
        for i in range(n_windows - 1):
            # Copy count from previous window
            all_counts[..., i+1] = all_counts[..., i]
            # Remove first kmer of previous window, add last kmer of next one
            np.add.at(
                all_counts,
                tuple(kmers[[i, winsize+1-k+i], j] for j in range(k)) + (i+1,),
                [-1, 1])
    if includeN:
        order += 'N'
    else:
        all_counts = all_counts[tuple(slice(0, -1) for i in range(k))
                                + (slice(None),)]
    if as_pandas:
        ser = pd.DataFrame(
            all_counts.reshape(len(order)**k, -1),
            index=pd.MultiIndex.from_product([list(order)]*k))
        return ser.sort_index()
    else:
        return all_counts


def ref_kmer_frequencies(freq_nucs, k=2):
    ser = pd.Series(
        1,
        index=pd.MultiIndex.from_product([list(flatten(freq_nucs.index))]*k))
    freq_nucs = freq_nucs / freq_nucs.sum(axis=0)
    for tup in ser.index:
        for nuc in tup:
            ser.loc[tup] *= freq_nucs[nuc]
    return ser


def random_shuffles(array, n):
    return array[np.random.rand(n, len(array)).argsort(axis=1)]


def shuffle_along_axis(arr, axis=0):
    """Shuffles a multi-dimensional array along specified axis"""
    assert isinstance(axis, int) and axis >= -1
    if axis == -1:
        axis = arr.ndim - 1
    assert arr.ndim > axis
    return arr[
        tuple(np.expand_dims(np.arange(arr.shape[dim]),
                             axis=tuple(i for i in range(axis+1)
                                        if i != dim))
              for dim in range(axis))
        + (np.random.rand(*arr.shape[:axis+1]).argsort(axis=axis),)]


def string_to_char_array(seq):
    """
    Converts an ASCII string to a NumPy array of byte-long ASCII codes.
    e.g. "ACGT" becomes [65, 67, 71, 84].
    From github.com/kundajelab/deeplift/blob/master/deeplift/dinuc_shuffle.py
    """
    return np.frombuffer(bytearray(seq, "utf8"), dtype=np.int8)


def char_array_to_string(arr):
    """
    Converts a NumPy array of byte-long ASCII codes into an ASCII string.
    e.g. [65, 67, 71, 84] becomes "ACGT".
    From github.com/kundajelab/deeplift/blob/master/deeplift/dinuc_shuffle.py
    """
    return arr.tostring().decode("ascii")


def one_hot_to_tokens(one_hot):
    """
    Converts an L x D one-hot encoding into an L-vector of integers in the
    range [0, D], where the token D is used when the one-hot encoding is all 0.
    This assumes that the one-hot encoding is well-formed, with at most one 1
    in each column (and 0s elsewhere).
    From github.com/kundajelab/deeplift/blob/master/deeplift/dinuc_shuffle.py
    """
    # Vector of all D
    tokens = np.full(one_hot.shape[0], one_hot.shape[1], dtype=int)
    seq_inds, dim_inds = np.where(one_hot)
    tokens[seq_inds] = dim_inds
    return tokens


def tokens_to_one_hot(tokens, one_hot_dim):
    """
    Converts an L-vector of integers in the range [0, D] to an L x D one-hot
    encoding. The value `D` must be provided as `one_hot_dim`. A token of D
    means the one-hot encoding is all 0s.
    From github.com/kundajelab/deeplift/blob/master/deeplift/dinuc_shuffle.py
    """
    identity = np.identity(one_hot_dim + 1)[:, :-1]  # Last row is all 0s
    return identity[tokens]


def dinuc_shuffle(seq, num_shufs=None, rng=None):
    """
    Creates shuffles of the given sequence, in which dinucleotide frequencies
    are preserved.
    Arguments:
        `seq`: either a string of length L, or an L x D NumPy array of one-hot
            encodings
        `num_shufs`: the number of shuffles to create, N; if unspecified, only
            one shuffle will be created
        `rng`: a NumPy RandomState object, to use for performing shuffles
    If `seq` is a string, returns a list of N strings of length L, each one
    being a shuffled version of `seq`. If `seq` is a 2D NumPy array, then the
    result is an N x L x D NumPy array of shuffled versions of `seq`, also
    one-hot encoded. If `num_shufs` is not specified, then the first dimension
    of N will not be present (i.e. a single string will be returned, or an
    L x D array).
    From github.com/kundajelab/deeplift/blob/master/deeplift/dinuc_shuffle.py
    """
    if type(seq) is str:
        arr = string_to_char_array(seq)
    elif type(seq) is np.ndarray and len(seq.shape) == 2:
        seq_len, one_hot_dim = seq.shape
        arr = one_hot_to_tokens(seq)
    else:
        raise ValueError("Expected string or one-hot encoded array")

    if not rng:
        rng = np.random.default_rng()

    # Get the set of all characters, and a mapping of which positions have
    # which characters; use `tokens`, which are integer representations of the
    # original characters
    chars, tokens = np.unique(arr, return_inverse=True)

    # For each token, get a list of indices of all the tokens that come after
    # it
    shuf_next_inds = []
    for t in range(len(chars)):
        mask = tokens[:-1] == t  # Excluding last char
        inds = np.where(mask)[0]
        shuf_next_inds.append(inds + 1)  # Add 1 for next token

    if type(seq) is str:
        all_results = []
    else:
        all_results = np.empty(
            (num_shufs if num_shufs else 1, seq_len, one_hot_dim),
            dtype=seq.dtype
        )

    for i in range(num_shufs if num_shufs else 1):
        # Shuffle the next indices
        for t in range(len(chars)):
            inds = np.arange(len(shuf_next_inds[t]))
            inds[:-1] = rng.permutation(len(inds) - 1)  # Keep last index same
            shuf_next_inds[t] = shuf_next_inds[t][inds]

        counters = [0] * len(chars)

        # Build the resulting array
        ind = 0
        result = np.empty_like(tokens)
        result[0] = tokens[ind]
        for j in range(1, len(tokens)):
            t = tokens[ind]
            ind = shuf_next_inds[t][counters[t]]
            counters[t] += 1
            result[j] = tokens[ind]

        if type(seq) is str:
            all_results.append(char_array_to_string(chars[result]))
        else:
            all_results[i] = tokens_to_one_hot(chars[result], one_hot_dim)
    return all_results if num_shufs else all_results[0]


def random_seq_strict_GC(n_seqs, seq_length, gc):
    gc_count = int(round((seq_length//2)*gc, 0))
    ref_seq = np.array(list(
        'A'*(seq_length % 2) + 'AT'*(seq_length//2-gc_count) + 'GC'*gc_count))
    return random_shuffles(ref_seq, n_seqs)


def random_sequences(n_seqs, seq_length, freq_kmers, seed=None, out='seq'):
    """Generate random DNA sequences with custom kmer distribution.

    Tested for k=2 or 3.

    Parameters
    ----------
    n_seqs : int
        Number of sequences to generate
    seq_length : int
        Length of the sequences to generate, must be greater than k
    freq_kmers : Series
        Series indexed by a k-level MultiIndex with the bases 'ACGT' on each
        level, contains frequencies or occurences of each k-mer
    seed : int, default=None
        Seed to use for random number generation
    out : {'seq', 'idx', 'one_hot'}
        Output format, 'seq' for nucleotide characters, 'idx' for indices into
        'ACGTN' or 'one_hot' for one-hot encoded bases

    Returns
    -------
    ndarray, shape=(`n_seqs`, `seq_length`)
        Generated sequences as a 2D-array of characters, of indices into
        'ACGTN' or 3D-array of one-hot encoded bases

    """
    # Array of bases for fast indexing
    letters = np.array(list('ACGTN'))

    # Get value of k
    k = freq_kmers.index.nlevels

    # Cumulative distribution of each base, given the previous k-1
    groups = freq_kmers.groupby(level=list(i for i in range(k-1)))
    sum = groups.transform("sum")
    cumsum = groups.transform("cumsum")
    p_cum_kmers = cumsum / sum
    # Convert to kD-array
    arr_kmers = np.zeros(tuple([4] * k))
    for tup in it.product(range(4), repeat=k):
        arr_kmers[tup] = np.asarray(
            p_cum_kmers.loc[tuple(letters[i] for i in tup)])
    # Set seed
    if seed is not None:
        np.random.seed(seed)
    # Empty sequences
    seqs = np.array([5]*seq_length*n_seqs).reshape(n_seqs, seq_length)
    # Get first k-mer given k-mer distribution
    r_start = np.random.choice(len(freq_kmers), n_seqs,
                               p=freq_kmers / freq_kmers.sum())
    seqs[:, :k] = np.array(list(it.product(range(4), repeat=k)))[r_start]
    # Generate random numbers for all iterations
    r = np.random.random((n_seqs, seq_length - k))
    # Get other bases given k-mer distribution, previous (k-1)-mer and random
    # numbers
    for i in range(k, seq_length):
        seqs[:, i] = np.argmax(
            arr_kmers[tuple(arr.ravel()
                            for arr in np.split(seqs[:, i-k+1:i], k-1, axis=1))
                      ] >= r[:, [i-k]*4],
            axis=1)
    if out == 'idx':
        return np.asarray(seqs, dtype=np.int8)
    elif out == 'seq':
        return letters[seqs]
    elif out == 'one_hot':
        return np.eye(4, dtype=bool)[seqs]


def random_sequences_as(one_hots, n_seqs, seq_length, k,
                        order='ACGT', seed=None, out='one_hot'):
    """Generate random DNA sequences with kmer distribution similar to input.

    Tested for k=2 or 3.

    Parameters
    ----------
    n_seqs : int
        Number of sequences to generate
    seq_length : int
        Length of the sequences to generate, must be greater than k
    k : int
        Length of k-mers to consider
    one_hots : list-like
        Must be a list, dictionnary or array of one-hot encoded sequences.
    order : str, default='ACGT'
        Order of bases for one-hot encoding
    seed : int, default=None
        Seed to use for random number generation
    out : {'one_hot', 'seq', 'idx'}
        Output format, 'seq' for nucleotide characters, 'idx' for indices into
        'ACGTN' or 'one_hot' for one-hot encoded bases. Default "one_hot"
        makes order parameter irrelevant.

    Returns
    -------
    ndarray, shape=(`n_seqs`, `seq_length`)
        Generated sequences as a 2D-array of characters, of indices into
        'ACGTN' or 3D-array of one-hot encoded bases
    """
    freq_kmers = kmer_counts(one_hots, k, order, includeN=False)
    return random_sequences(n_seqs, seq_length, freq_kmers, seed, out)


# Other utils
def s_plural(value: float) -> str:
    """Return s if scalar value induces plural"""
    if value > 1:
        return 's'
    else:
        return ''


def format_secs(x):
    d, r = divmod(x, 86400)
    h, r = divmod(r, 3600)
    m, s = divmod(r, 60)
    if d != 0:
        print(f'{d}d', end=' ')
    print(f'{h}h{m}m{s}s')


def safe_filename(file: Path) -> Path:
    """Make sure file can be build without overriding an other.

    If file already exists, returns a new filename with a number in between
    parenthesis. If the parent of the file doesn't exist, it is created.

    Raises
    ------
    FileExistsError
        If one of the parents of the file to create is an existing file
    """
    file = Path(file)
    # Build parent directories if needed
    if not file.parent.is_dir():
        print("Building parent directories")
        file.parent.mkdir(parents=True)
    # Change filename if it already exists
    if file.exists():
        original_file = file
        file_dups = 0
        while file.exists():
            file_dups += 1
            file = Path(file.parent,
                        original_file.stem + f'({file_dups})' + file.suffix)
            # python3.9: file.with_stem(original_file.stem + f'({file_dups})')
        print(f'{original_file} exists, changing filename to {file}')
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


def roman_to_int(str):
    sym_values = {'I': 1, 'V': 5, 'X': 10, 'L': 50,
                  'C': 100, 'D': 500, 'M': 1000}
    res = 0
    i = 0
    while (i < len(str)):
        # Get value of current symbol
        s1 = sym_values[str[i]]
        if (i + 1 < len(str)):
            # Get value of next symbol
            s2 = sym_values[str[i + 1]]
            if (s1 >= s2):
                res = res + s1
                i = i + 1
            else:
                res = res + s2 - s1
                i = i + 2
        else:
            res = res + s1
            i = i + 1
    return res


def int_to_roman(number):
    num = [1, 4, 5, 9, 10, 40, 50, 90,
           100, 400, 500, 900, 1000]
    sym = ["I", "IV", "V", "IX", "X", "XL",
           "L", "XC", "C", "CD", "D", "CM", "M"]
    i = 12
    res = ''
    while number:
        div = number // num[i]
        number %= num[i]
        while div:
            res += sym[i]
            div -= 1
        i -= 1
    return res


def flatten(container):
    for i in container:
        if isinstance(i, (list, tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i


# Test functions
def kmer_counts_test(orders=['ACGT', 'ATCG'],
                     fasts=[True, False]):
    # All possible bases including N in different quantities
    one_hot1 = np.array([[1, 0, 0, 0]]
                        + [[0, 1, 0, 0]]*2
                        + [[0, 0, 1, 0]]*3
                        + [[0, 0, 0, 1]]*4
                        + [[0, 0, 0, 0]]*5)
    # All possible bases except N in different quantities
    one_hot2 = np.array([[1, 0, 0, 0]]
                        + [[0, 1, 0, 0]]*2
                        + [[0, 0, 1, 0]]*3
                        + [[0, 0, 0, 1]]*4)
    # Shuffled version of one_hot1, starting with N
    one_hot3 = np.array([[0, 0, 0, 0],
                         [0, 0, 0, 1],
                         [0, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 0],
                         [1, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1],
                         [0, 0, 0, 1],
                         [0, 0, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
    # Shuffled version of one_hot2
    one_hot4 = np.array([[0, 0, 1, 0],
                         [0, 0, 0, 1],
                         [0, 0, 0, 1],
                         [0, 0, 0, 1],
                         [0, 0, 1, 0],
                         [0, 0, 1, 0],
                         [1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1]])
    # No 2mer without N
    one_hot5 = np.array([[0, 0, 1, 0],
                         [0, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 0],
                         [1, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 1],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0]])
    one_hots = [one_hot1, one_hot2, one_hot3, one_hot4, one_hot5]
    one_hots_dict = {i: one_hot for i, one_hot in enumerate(one_hots)}

    k = 1
    includeN = True
    for order in orders:
        ref1 = pd.Series(
            np.arange(1, 6),
            index=pd.MultiIndex.from_product([list(order + 'N')])
        ).sort_index()
        ref2 = pd.Series(
            [i for i in range(1, 5)] + [0],
            index=pd.MultiIndex.from_product([list(order + 'N')])
        ).sort_index()
        ref3 = ref1
        ref4 = ref2
        ref5 = pd.Series(
            [1]*4 + [6],
            index=pd.MultiIndex.from_product([list(order + 'N')])
        ).sort_index()
        refs = [ref1, ref2, ref3, ref4, ref5]
        for fast in fasts:
            for one_hot, ref in zip(one_hots, refs):
                r = kmer_counts(one_hot,
                                k=k,
                                includeN=includeN,
                                order=order)
                assert np.all(r == ref)
                r = kmer_counts([one_hot],
                                k=k,
                                includeN=includeN,
                                order=order)
                assert np.all(r == ref)
                r = kmer_counts({'foo': one_hot},
                                k=k,
                                includeN=includeN,
                                order=order)
                assert np.all(r == ref)
                r = kmer_counts(one_hot.reshape(-1, 5, 4),
                                k=k,
                                includeN=includeN,
                                order=order)
                assert np.all(r == ref)
        ref = pd.Series(
            [5, 9, 13, 17, 16],
            index=pd.MultiIndex.from_product([list(order + 'N')])
        ).sort_index()
        for fast in fasts:
            r = kmer_counts(one_hots,
                            k=k,
                            includeN=includeN,
                            order=order)
            assert np.all(r == ref)
            r = kmer_counts(one_hots_dict,
                            k=k,
                            includeN=includeN,
                            order=order)
            assert np.all(r == ref)

    k = 1
    includeN = False
    for order in orders:
        ref1 = pd.Series(
            np.arange(1, 5),
            index=pd.MultiIndex.from_product([list(order)])
        ).sort_index()
        ref2 = ref1
        ref3 = ref1
        ref4 = ref1
        ref5 = pd.Series(
            [1]*4,
            index=pd.MultiIndex.from_product([list(order)])
        ).sort_index()
        refs = [ref1, ref2, ref3, ref4, ref5]
        for fast in fasts:
            for one_hot, ref in zip(one_hots, refs):
                r = kmer_counts(one_hot,
                                k=k,
                                includeN=includeN,
                                order=order)
                assert np.all(r == ref)
                r = kmer_counts([one_hot],
                                k=k,
                                includeN=includeN,
                                order=order)
                assert np.all(r == ref)
                r = kmer_counts({'foo': one_hot},
                                k=k,
                                includeN=includeN,
                                order=order)
                assert np.all(r == ref)
                r = kmer_counts(one_hot.reshape(-1, 5, 4),
                                k=k,
                                includeN=includeN,
                                order=order)
                assert np.all(r == ref)
        ref = pd.Series(
            [5, 9, 13, 17],
            index=pd.MultiIndex.from_product([list(order)])
        ).sort_index()
        for fast in fasts:
            r = kmer_counts(one_hots,
                            k=k,
                            includeN=includeN,
                            order=order)
            assert np.all(r == ref)
            r = kmer_counts(one_hots_dict,
                            k=k,
                            includeN=includeN,
                            order=order)
            assert np.all(r == ref)

    k = 2
    includeN = True
    for order in orders:
        ref1 = pd.Series(
            [0, 1, 0, 0, 0,
             0, 1, 1, 0, 0,
             0, 0, 2, 1, 0,
             0, 0, 0, 3, 1,
             0, 0, 0, 0, 4],
            index=pd.MultiIndex.from_product([list(order + 'N')]*2)
        ).sort_index()
        ref2 = pd.Series(
            [0, 1, 0, 0, 0,
             0, 1, 1, 0, 0,
             0, 0, 2, 1, 0,
             0, 0, 0, 3, 0,
             0, 0, 0, 0, 0],
            index=pd.MultiIndex.from_product([list(order + 'N')]*2)
        ).sort_index()
        ref3 = pd.Series(
            [0, 0, 0, 0, 1,
             0, 0, 1, 0, 1,
             0, 0, 1, 2, 0,
             0, 0, 0, 1, 2,
             1, 2, 1, 1, 0],
            index=pd.MultiIndex.from_product([list(order + 'N')]*2)
        ).sort_index()
        ref4 = pd.Series(
            [0, 1, 0, 0, 0,
             0, 1, 0, 1, 0,
             1, 0, 1, 1, 0,
             0, 0, 1, 2, 0,
             0, 0, 0, 0, 0],
            index=pd.MultiIndex.from_product([list(order + 'N')]*2)
        ).sort_index()
        ref5 = pd.Series(
            [0, 0, 0, 0, 1,
             0, 0, 0, 0, 1,
             0, 0, 0, 0, 1,
             0, 0, 0, 0, 1,
             1, 1, 0, 1, 2],
            index=pd.MultiIndex.from_product([list(order + 'N')]*2)
        ).sort_index()
        refs = [ref1, ref2, ref3, ref4, ref5]
        for fast in fasts:
            for one_hot, ref in zip(one_hots, refs):
                r = kmer_counts(one_hot,
                                k=k,
                                includeN=includeN,
                                order=order)
                assert np.all(r == ref)
                r = kmer_counts([one_hot],
                                k=k,
                                includeN=includeN,
                                order=order)
                assert np.all(r == ref)
                r = kmer_counts({'foo': one_hot},
                                k=k,
                                includeN=includeN,
                                order=order)
                assert np.all(r == ref)
        ref1 = pd.Series(
            [0, 1, 0, 0, 0,
             0, 1, 1, 0, 0,
             0, 0, 1, 1, 0,
             0, 0, 0, 3, 0,
             0, 0, 0, 0, 4],
            index=pd.MultiIndex.from_product([list(order + 'N')]*2)
        ).sort_index()
        ref2 = pd.Series(
            [0, 1, 0, 0, 0,
             0, 1, 1, 0, 0,
             0, 0, 1, 1, 0,
             0, 0, 0, 3, 0,
             0, 0, 0, 0, 0],
            index=pd.MultiIndex.from_product([list(order + 'N')]*2)
        ).sort_index()
        ref3 = pd.Series(
            [0, 0, 0, 0, 1,
             0, 0, 1, 0, 1,
             0, 0, 1, 2, 0,
             0, 0, 0, 0, 2,
             0, 2, 1, 1, 0],
            index=pd.MultiIndex.from_product([list(order + 'N')]*2)
        ).sort_index()
        ref4 = pd.Series(
            [0, 1, 0, 0, 0,
             0, 1, 0, 1, 0,
             1, 0, 0, 1, 0,
             0, 0, 1, 2, 0,
             0, 0, 0, 0, 0],
            index=pd.MultiIndex.from_product([list(order + 'N')]*2)
        ).sort_index()
        ref5 = pd.Series(
            [0, 0, 0, 0, 0,
             0, 0, 0, 0, 1,
             0, 0, 0, 0, 1,
             0, 0, 0, 0, 1,
             1, 1, 0, 1, 2],
            index=pd.MultiIndex.from_product([list(order + 'N')]*2)
        ).sort_index()
        refs = [ref1, ref2, ref3, ref4, ref5]
        for fast in fasts:
            for one_hot, ref in zip(one_hots, refs):
                r = kmer_counts(one_hot.reshape(-1, 5, 4),
                                k=k,
                                includeN=includeN,
                                order=order)
                assert np.all(r == ref)
        ref = pd.Series(
            [0, 3, 0, 0, 2,
             0, 3, 3, 1, 2,
             1, 0, 6, 5, 1,
             0, 0, 1, 9, 4,
             2, 3, 1, 2, 6],
            index=pd.MultiIndex.from_product([list(order + 'N')]*2)
        ).sort_index()
        for fast in fasts:
            r = kmer_counts(one_hots,
                            k=k,
                            includeN=includeN,
                            order=order)
            assert np.all(r == ref)
            r = kmer_counts(one_hots_dict,
                            k=k,
                            includeN=includeN,
                            order=order)
            assert np.all(r == ref)

    k = 2
    includeN = False
    for order in orders:
        ref1 = pd.Series(
            [0, 1, 0, 0,
             0, 1, 1, 0,
             0, 0, 2, 1,
             0, 0, 0, 3],
            index=pd.MultiIndex.from_product([list(order)]*2)
        ).sort_index()
        ref2 = ref1
        ref3 = pd.Series(
            [0, 0, 0, 0,
             0, 0, 1, 0,
             0, 0, 1, 2,
             0, 0, 0, 1],
            index=pd.MultiIndex.from_product([list(order)]*2)
        ).sort_index()
        ref4 = pd.Series(
            [0, 1, 0, 0,
             0, 1, 0, 1,
             1, 0, 1, 1,
             0, 0, 1, 2],
            index=pd.MultiIndex.from_product([list(order)]*2)
        ).sort_index()
        ref5 = pd.Series(
            [0, 0, 0, 0,
             0, 0, 0, 0,
             0, 0, 0, 0,
             0, 0, 0, 0],
            index=pd.MultiIndex.from_product([list(order)]*2)
        ).sort_index()
        refs = [ref1, ref2, ref3, ref4, ref5]
        for fast in fasts:
            for one_hot, ref in zip(one_hots, refs):
                r = kmer_counts(one_hot,
                                k=k,
                                includeN=includeN,
                                order=order)
                assert np.all(r == ref)
                r = kmer_counts([one_hot],
                                k=k,
                                includeN=includeN,
                                order=order)
                assert np.all(r == ref)
                r = kmer_counts({'foo': one_hot},
                                k=k,
                                includeN=includeN,
                                order=order)
                assert np.all(r == ref)
        ref1 = pd.Series(
            [0, 1, 0, 0,
             0, 1, 1, 0,
             0, 0, 1, 1,
             0, 0, 0, 3],
            index=pd.MultiIndex.from_product([list(order)]*2)
        ).sort_index()
        ref2 = ref1
        ref3 = pd.Series(
            [0, 0, 0, 0,
             0, 0, 1, 0,
             0, 0, 1, 2,
             0, 0, 0, 0],
            index=pd.MultiIndex.from_product([list(order)]*2)
        ).sort_index()
        ref4 = pd.Series(
            [0, 1, 0, 0,
             0, 1, 0, 1,
             1, 0, 0, 1,
             0, 0, 1, 2],
            index=pd.MultiIndex.from_product([list(order)]*2)
        ).sort_index()
        ref5 = pd.Series(
            [0, 0, 0, 0,
             0, 0, 0, 0,
             0, 0, 0, 0,
             0, 0, 0, 0],
            index=pd.MultiIndex.from_product([list(order)]*2)
        ).sort_index()
        refs = [ref1, ref2, ref3, ref4, ref5]
        for fast in fasts:
            for one_hot, ref in zip(one_hots, refs):
                r = kmer_counts(one_hot.reshape(-1, 5, 4),
                                k=k,
                                includeN=includeN,
                                order=order)
                assert np.all(r == ref)
        ref = pd.Series(
            [0, 3, 0, 0,
             0, 3, 3, 1,
             1, 0, 6, 5,
             0, 0, 1, 9],
            index=pd.MultiIndex.from_product([list(order)]*2)
        ).sort_index()
        for fast in fasts:
            r = kmer_counts(one_hots,
                            k=k,
                            includeN=includeN,
                            order=order)
            assert np.all(r == ref)
            r = kmer_counts(one_hots_dict,
                            k=k,
                            includeN=includeN,
                            order=order)
            assert np.all(r == ref)

    k = 3
    includeN = True
    for order in orders:
        ref1 = pd.Series(
            [0, 0, 0, 0, 0,
             0, 1, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,

             0, 0, 0, 0, 0,
             0, 0, 1, 0, 0,
             0, 0, 1, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,

             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 1, 1, 0,
             0, 0, 0, 1, 0,
             0, 0, 0, 0, 0,

             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 2, 1,
             0, 0, 0, 0, 1,

             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 3],
            index=pd.MultiIndex.from_product([list(order + 'N')]*3)
        ).sort_index()
        ref2 = pd.Series(
            [0, 0, 0, 0, 0,
             0, 1, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,

             0, 0, 0, 0, 0,
             0, 0, 1, 0, 0,
             0, 0, 1, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,

             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 1, 1, 0,
             0, 0, 0, 1, 0,
             0, 0, 0, 0, 0,

             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 2, 0,
             0, 0, 0, 0, 0,

             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0],
            index=pd.MultiIndex.from_product([list(order + 'N')]*3)
        ).sort_index()
        ref3 = pd.Series(
            [0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 1, 0, 0, 0,

             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 1, 0,
             0, 0, 0, 0, 0,
             1, 0, 0, 0, 0,

             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 1, 0,
             0, 0, 0, 1, 0,
             0, 0, 0, 0, 0,

             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 1,
             0, 1, 1, 0, 0,

             0, 0, 0, 0, 1,
             0, 0, 1, 0, 1,
             0, 0, 1, 0, 0,
             0, 0, 0, 0, 1,
             0, 0, 0, 0, 0],
            index=pd.MultiIndex.from_product([list(order + 'N')]*3)
        ).sort_index()
        ref4 = pd.Series(
            [0, 0, 0, 0, 0,
             0, 1, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,

             0, 0, 0, 0, 0,
             0, 0, 0, 1, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,

             0, 1, 0, 0, 0,
             0, 0, 0, 0, 0,
             1, 0, 0, 0, 0,
             0, 0, 0, 1, 0,
             0, 0, 0, 0, 0,

             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 1, 0, 0,
             0, 0, 1, 1, 0,
             0, 0, 0, 0, 0,

             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0],
            index=pd.MultiIndex.from_product([list(order + 'N')]*3)
        ).sort_index()
        ref5 = pd.Series(
            [0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 1, 0,

             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             1, 0, 0, 0, 0,

             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 1, 0, 0, 0,

             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 1,

             0, 0, 0, 0, 1,
             0, 0, 0, 0, 1,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 1,
             0, 0, 0, 0, 1],
            index=pd.MultiIndex.from_product([list(order + 'N')]*3)
        ).sort_index()
        refs = [ref1, ref2, ref3, ref4, ref5]
        for fast in fasts:
            for one_hot, ref in zip(one_hots, refs):
                r = kmer_counts(one_hot,
                                k=k,
                                includeN=includeN,
                                order=order)
                assert np.all(r == ref)
                r = kmer_counts([one_hot],
                                k=k,
                                includeN=includeN,
                                order=order)
                assert np.all(r == ref)
                r = kmer_counts({'foo': one_hot},
                                k=k,
                                includeN=includeN,
                                order=order)
                assert np.all(r == ref)
        ref1 = pd.Series(
            [0, 0, 0, 0, 0,
             0, 1, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,

             0, 0, 0, 0, 0,
             0, 0, 1, 0, 0,
             0, 0, 1, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,

             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 1, 0,
             0, 0, 0, 0, 0,

             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 2, 0,
             0, 0, 0, 0, 0,

             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 3],
            index=pd.MultiIndex.from_product([list(order + 'N')]*3)
        ).sort_index()
        ref2 = pd.Series(
            [0, 0, 0, 0, 0,
             0, 1, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,

             0, 0, 0, 0, 0,
             0, 0, 1, 0, 0,
             0, 0, 1, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,

             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 1, 0,
             0, 0, 0, 0, 0,

             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 2, 0,
             0, 0, 0, 0, 0,

             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0],
            index=pd.MultiIndex.from_product([list(order + 'N')]*3)
        ).sort_index()
        ref3 = pd.Series(
            [0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 1, 0, 0, 0,

             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 1, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,

             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 1, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,

             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 1, 1, 0, 0,

             0, 0, 0, 0, 0,
             0, 0, 1, 0, 1,
             0, 0, 1, 0, 0,
             0, 0, 0, 0, 1,
             0, 0, 0, 0, 0],
            index=pd.MultiIndex.from_product([list(order + 'N')]*3)
        ).sort_index()
        ref4 = pd.Series(
            [0, 0, 0, 0, 0,
             0, 1, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,

             0, 0, 0, 0, 0,
             0, 0, 0, 1, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,

             0, 1, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 1, 0,
             0, 0, 0, 0, 0,

             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 1, 1, 0,
             0, 0, 0, 0, 0,

             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0],
            index=pd.MultiIndex.from_product([list(order + 'N')]*3)
        ).sort_index()
        ref5 = pd.Series(
            [0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,

             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             1, 0, 0, 0, 0,

             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 1, 0, 0, 0,

             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 1,

             0, 0, 0, 0, 0,
             0, 0, 0, 0, 1,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 1,
             0, 0, 0, 0, 1],
            index=pd.MultiIndex.from_product([list(order + 'N')]*3)
        ).sort_index()
        refs = [ref1, ref2, ref3, ref4, ref5]
        for fast in fasts:
            for one_hot, ref in zip(one_hots, refs):
                r = kmer_counts(one_hot.reshape(-1, 5, 4),
                                k=k,
                                includeN=includeN,
                                order=order)
                assert np.all(r == ref)
        ref = pd.Series(
            [0, 0, 0, 0, 0,
             0, 3, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 1, 0, 1, 0,

             0, 0, 0, 0, 0,
             0, 0, 2, 1, 0,
             0, 0, 2, 1, 0,
             0, 0, 0, 0, 0,
             2, 0, 0, 0, 0,

             0, 1, 0, 0, 0,
             0, 0, 0, 0, 0,
             1, 0, 2, 3, 0,
             0, 0, 0, 4, 0,
             0, 1, 0, 0, 0,

             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 1, 0, 0,
             0, 0, 1, 5, 2,
             0, 1, 1, 0, 2,

             0, 0, 0, 0, 2,
             0, 0, 1, 0, 2,
             0, 0, 1, 0, 0,
             0, 0, 0, 0, 2,
             0, 0, 0, 0, 4],
            index=pd.MultiIndex.from_product([list(order + 'N')]*3)
        ).sort_index()
        for fast in fasts:
            r = kmer_counts(one_hots,
                            k=k,
                            includeN=includeN,
                            order=order)
            assert np.all(r == ref)
            r = kmer_counts(one_hots_dict,
                            k=k,
                            includeN=includeN,
                            order=order)
            assert np.all(r == ref)

    k = 3
    includeN = False
    for order in orders:
        ref1 = pd.Series(
            [0, 0, 0, 0,
             0, 1, 0, 0,
             0, 0, 0, 0,
             0, 0, 0, 0,

             0, 0, 0, 0,
             0, 0, 1, 0,
             0, 0, 1, 0,
             0, 0, 0, 0,

             0, 0, 0, 0,
             0, 0, 0, 0,
             0, 0, 1, 1,
             0, 0, 0, 1,

             0, 0, 0, 0,
             0, 0, 0, 0,
             0, 0, 0, 0,
             0, 0, 0, 2],
            index=pd.MultiIndex.from_product([list(order)]*3)
        ).sort_index()
        ref2 = ref1
        ref3 = pd.Series(
            [0, 0, 0, 0,
             0, 0, 0, 0,
             0, 0, 0, 0,
             0, 0, 0, 0,

             0, 0, 0, 0,
             0, 0, 0, 0,
             0, 0, 0, 1,
             0, 0, 0, 0,

             0, 0, 0, 0,
             0, 0, 0, 0,
             0, 0, 0, 1,
             0, 0, 0, 1,

             0, 0, 0, 0,
             0, 0, 0, 0,
             0, 0, 0, 0,
             0, 0, 0, 0],
            index=pd.MultiIndex.from_product([list(order)]*3)
        ).sort_index()
        ref4 = pd.Series(
            [0, 0, 0, 0,
             0, 1, 0, 0,
             0, 0, 0, 0,
             0, 0, 0, 0,

             0, 0, 0, 0,
             0, 0, 0, 1,
             0, 0, 0, 0,
             0, 0, 0, 0,

             0, 1, 0, 0,
             0, 0, 0, 0,
             1, 0, 0, 0,
             0, 0, 0, 1,

             0, 0, 0, 0,
             0, 0, 0, 0,
             0, 0, 1, 0,
             0, 0, 1, 1],
            index=pd.MultiIndex.from_product([list(order)]*3)
        ).sort_index()
        ref5 = pd.Series(
            [0, 0, 0, 0,
             0, 0, 0, 0,
             0, 0, 0, 0,
             0, 0, 0, 0,

             0, 0, 0, 0,
             0, 0, 0, 0,
             0, 0, 0, 0,
             0, 0, 0, 0,

             0, 0, 0, 0,
             0, 0, 0, 0,
             0, 0, 0, 0,
             0, 0, 0, 0,

             0, 0, 0, 0,
             0, 0, 0, 0,
             0, 0, 0, 0,
             0, 0, 0, 0],
            index=pd.MultiIndex.from_product([list(order)]*3)
        ).sort_index()
        refs = [ref1, ref2, ref3, ref4, ref5]
        for fast in fasts:
            for one_hot, ref in zip(one_hots, refs):
                r = kmer_counts(one_hot,
                                k=k,
                                includeN=includeN,
                                order=order)
                assert np.all(r == ref)
                r = kmer_counts([one_hot],
                                k=k,
                                includeN=includeN,
                                order=order)
                assert np.all(r == ref)
                r = kmer_counts({'foo': one_hot},
                                k=k,
                                includeN=includeN,
                                order=order)
                assert np.all(r == ref)
        ref1 = pd.Series(
            [0, 0, 0, 0,
             0, 1, 0, 0,
             0, 0, 0, 0,
             0, 0, 0, 0,

             0, 0, 0, 0,
             0, 0, 1, 0,
             0, 0, 1, 0,
             0, 0, 0, 0,

             0, 0, 0, 0,
             0, 0, 0, 0,
             0, 0, 0, 0,
             0, 0, 0, 1,

             0, 0, 0, 0,
             0, 0, 0, 0,
             0, 0, 0, 0,
             0, 0, 0, 2],
            index=pd.MultiIndex.from_product([list(order)]*3)
        ).sort_index()
        ref2 = ref1
        ref3 = pd.Series(
            [0, 0, 0, 0,
             0, 0, 0, 0,
             0, 0, 0, 0,
             0, 0, 0, 0,

             0, 0, 0, 0,
             0, 0, 0, 0,
             0, 0, 0, 1,
             0, 0, 0, 0,

             0, 0, 0, 0,
             0, 0, 0, 0,
             0, 0, 0, 1,
             0, 0, 0, 0,

             0, 0, 0, 0,
             0, 0, 0, 0,
             0, 0, 0, 0,
             0, 0, 0, 0],
            index=pd.MultiIndex.from_product([list(order)]*3)
        ).sort_index()
        ref4 = pd.Series(
            [0, 0, 0, 0,
             0, 1, 0, 0,
             0, 0, 0, 0,
             0, 0, 0, 0,

             0, 0, 0, 0,
             0, 0, 0, 1,
             0, 0, 0, 0,
             0, 0, 0, 0,

             0, 1, 0, 0,
             0, 0, 0, 0,
             0, 0, 0, 0,
             0, 0, 0, 1,

             0, 0, 0, 0,
             0, 0, 0, 0,
             0, 0, 0, 0,
             0, 0, 1, 1],
            index=pd.MultiIndex.from_product([list(order)]*3)
        ).sort_index()
        ref5 = pd.Series(
            [0, 0, 0, 0,
             0, 0, 0, 0,
             0, 0, 0, 0,
             0, 0, 0, 0,

             0, 0, 0, 0,
             0, 0, 0, 0,
             0, 0, 0, 0,
             0, 0, 0, 0,

             0, 0, 0, 0,
             0, 0, 0, 0,
             0, 0, 0, 0,
             0, 0, 0, 0,

             0, 0, 0, 0,
             0, 0, 0, 0,
             0, 0, 0, 0,
             0, 0, 0, 0],
            index=pd.MultiIndex.from_product([list(order)]*3)
        ).sort_index()
        refs = [ref1, ref2, ref3, ref4, ref5]
        for fast in fasts:
            for one_hot, ref in zip(one_hots, refs):
                r = kmer_counts(one_hot.reshape(-1, 5, 4),
                                k=k,
                                includeN=includeN,
                                order=order)
                assert np.all(r == ref)
        ref = pd.Series(
            [0, 0, 0, 0,
             0, 3, 0, 0,
             0, 0, 0, 0,
             0, 0, 0, 0,

             0, 0, 0, 0,
             0, 0, 2, 1,
             0, 0, 2, 1,
             0, 0, 0, 0,

             0, 1, 0, 0,
             0, 0, 0, 0,
             1, 0, 2, 3,
             0, 0, 0, 4,

             0, 0, 0, 0,
             0, 0, 0, 0,
             0, 0, 1, 0,
             0, 0, 1, 5],
            index=pd.MultiIndex.from_product([list(order)]*3)
        ).sort_index()
        for fast in fasts:
            r = kmer_counts(one_hots,
                            k=k,
                            includeN=includeN,
                            order=order)
            assert np.all(r == ref)
            r = kmer_counts(one_hots_dict,
                            k=k,
                            includeN=includeN,
                            order=order)
            assert np.all(r == ref)

    for k, includeN, order, one_hot, winsize in it.product(range(1, 4),
                                                           [True, False],
                                                           orders,
                                                           one_hots,
                                                           range(k, 6)):
        assert np.all(
            kmer_counts_by_seq(
                sliding_window_view(one_hot, (winsize, 4)).squeeze(),
                k, order=order, includeN=includeN)
            == sliding_kmer_counts(
                one_hot, k, winsize, order=order, includeN=includeN))


if __name__ == "__main__":
    kmer_counts_test()
