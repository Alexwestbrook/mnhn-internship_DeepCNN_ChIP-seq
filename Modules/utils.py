#!/usr/bin/env python
import itertools as it
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Generator, Iterable, List, Tuple, Union

import numpy as np
import pandas as pd
import pyBigWig
import pysam
import scipy
from numpy.core.numeric import normalize_axis_tuple
from numpy.lib.stride_tricks import as_strided
from scipy import signal
from scipy.signal import convolve, gaussian
from scipy.sparse import coo_matrix
from scipy.stats import pearsonr
from sklearn.preprocessing import OneHotEncoder
from statsmodels.stats import multitest


def data_generation(IDs, reads, labels, class_weights):
    X = np.empty((len(IDs), *reads[0].shape), dtype="bool")
    Y = np.empty((len(IDs), 1), dtype="bool")
    weights = np.empty((len(IDs), 1), dtype="float")
    for i, ID in enumerate(IDs):
        X[i,] = reads[ID]
        Y[i] = labels[ID]
        weights[i] = class_weights[labels[ID]]
    return X, Y, weights


def data_generator(
    dataset_dir: str,
    batch_size: int,
    class_weights: Dict[int, int] = {0: 1, 1: 1},
    shuffle: bool = True,
    split: str = "train",
    relabeled: bool = False,
    rng: np.random.Generator = np.random.default_rng(),
    cache: bool = True,
) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray], None, None]:
    files = list(Path(dataset_dir).glob(split + "_*"))

    first_loop = True
    new_files = []
    while True:
        if shuffle:
            rng.shuffle(files)
        for file in files:
            if first_loop:
                with np.load(file) as f:
                    x = f["one_hots"]
            else:
                x = np.load(file)
            if relabeled:
                label_file = Path(file.parent, "labels_" + file.name)
                with np.load(label_file) as f:
                    labels = f["labels"]
            else:
                labels = np.zeros(len(x), dtype=bool)
                labels[::2] = 1

            indexes = np.arange(len(x))
            list_IDs = indexes

            n_batch = int(np.ceil(len(list_IDs) / batch_size))
            if shuffle:
                rng.shuffle(indexes)

            for index in range(n_batch):
                start_batch = index * batch_size
                end_batch = (index + 1) * batch_size
                indexes_batch = indexes[start_batch:end_batch]
                list_IDs_batch = [list_IDs[k] for k in indexes_batch]
                yield data_generation(list_IDs_batch, x, labels, class_weights)
            if first_loop:
                new_file = Path(file.parent, file.stem + ".npy")
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
        one_hot_chr = f["one_hot_genome"]
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


def merge_chroms(chr_ids: Iterable[str], file: str):
    annot = []
    with np.load(file) as f:
        for chr_id in chr_ids:
            annot.append(f[chr_id])
            shape, dtype = f[chr_id].shape, f[chr_id].dtype
            annot.append(np.zeros((1,) + shape[1:], dtype=dtype))
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
    pos_weight = 1 / n_pos * (n_pos + n_neg) / 2
    neg_weight = 1 / n_neg * (n_pos + n_neg) / 2
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
    sample_weights = np.where(np.squeeze(y) == 1, weights[1], weights[0])
    return sample_weights


# One-hot encoding and decoding
def one_hot_encode(seq, length=None, one_hot_type=bool, order="ACGT"):
    if length is None:
        length = len(seq)
    one_hot = np.zeros((length, 4), dtype=one_hot_type)
    for i, base in enumerate(seq):
        if i >= length:
            break
        if base.upper() == order[0]:
            one_hot[i, 0] = 1
        elif base.upper() == order[1]:
            one_hot[i, 1] = 1
        elif base.upper() == order[2]:
            one_hot[i, 2] = 1
        elif base.upper() == order[3]:
            one_hot[i, 3] = 1
    return one_hot


def one_hot_decode(one_hot, order="ACGT"):
    if len(one_hot.shape) != 2:
        raise ValueError("input must be a single one hot encoded read")
    if order == "ACGT":
        categories = np.array(list("ACGT")).reshape(-1, 1)
        encoder = OneHotEncoder(
            dtype=one_hot.dtype, handle_unknown="ignore", sparse=False
        )
        encoder.fit(categories)

        seq = encoder.inverse_transform(one_hot)
        seq = seq.ravel()
        seq = "".join(["N" if value is None else value for value in seq])
        return seq
    else:
        bases = np.array(list(order))
        seq = bases[np.argmax(one_hot, axis=1)]
        seq[np.sum(one_hot, axis=1) != 1] = "N"
        return "".join(seq)


def one_hot_encoding(
    array: np.ndarray,
    read_length: int = 101,
    one_hot_type: type = bool,
    order: str = "ACGT",
) -> np.ndarray:
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
    return one_hot_encoding_v1(
        array, read_length=read_length, one_hot_type=one_hot_type, order=order
    )


def one_hot_encoding_v1(
    array: np.ndarray,
    read_length: int = 101,
    one_hot_type: type = bool,
    order: str = "ACGT",
) -> np.ndarray:
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
        print(
            f"Warning: {unmatched_lengths} sequences don't have the "
            "appropriate read length"
        )
    return new_array


def one_hot_encoding_v2(
    reads: np.ndarray,
    read_length: int = 101,
    one_hot_type: type = bool,
    sparse: bool = False,
) -> np.ndarray:
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
            reads[i] = read[:read_length] + [["N"]] * max(0, read_length - len(read))
    # Raise warning if some sequences do not match the read length
    if unmatched_lengths != 0:
        print(
            f"Warning: {unmatched_lengths} sequences don't have the "
            "appropriate read length"
        )

    categories = np.array([["A"], ["C"], ["G"], ["T"]])
    encoder = OneHotEncoder(dtype=one_hot_type, handle_unknown="ignore", sparse=sparse)
    encoder.fit(categories)

    one_hots = encoder.transform(np.reshape(reads, (-1, 1)))
    one_hots.shape = (-1, read_length, 4)
    return one_hots


def one_hot_to_seq(reads, order="ACGT"):
    if order == "ACGT":
        return one_hot_to_seq_v2(reads)
    else:
        return one_hot_to_seq_v1(reads, order)


def one_hot_to_seq_v1(reads, order="ACGT"):
    """
    Convert one_hot array of reads into list of sequences.
    """
    if len(reads.shape) != 3:
        raise ValueError("must be an array of one hot encoded reads")
    bases = np.array(list(order))
    seqs = bases[np.argmax(reads, axis=2)]
    seqs[np.sum(reads, axis=2) != 1] = "N"
    seqs = ["".join([char for char in seq]) for seq in seqs]
    return seqs


def one_hot_to_seq_v2(reads):
    """
    Convert one_hot array of reads into list of sequences.

    This implementation uses scikit-learn's OneHotEncoder
    """
    if len(reads.shape) == 3:
        n_reads, read_length, n_bases = reads.shape
    else:
        raise ValueError("must be an array of one hot encoded read")
    categories = np.array([["A"], ["C"], ["G"], ["T"]])
    encoder = OneHotEncoder(dtype=bool, handle_unknown="ignore", sparse=False)
    encoder.fit(categories)

    reads.shape = (-1, n_bases)
    seqs = encoder.inverse_transform(reads)
    reads.shape = (n_reads, read_length, n_bases)
    seqs.shape = (n_reads, read_length)
    seqs = ["".join(["N" if value is None else value for value in seq]) for seq in seqs]
    return seqs


def np_idx_to_one_hot(idx, order="ACGT", extradims=None):
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
        Array in one-hot format.
    """
    assert len(order) == 4 and set(order) == set("ACGT")
    converter = np.zeros((5, 4), dtype=bool)
    for i, c in enumerate("ACGT"):
        converter[i, order.find(c)] = 1
    one_hot = converter[idx]
    if extradims is not None:
        one_hot = np.expand_dims(one_hot, axis=extradims)
    return one_hot


def one_hot_to_idx(one_hot, order="ACGT"):
    """Convert array in one-hot format into indexes.

    Parameters
    ----------
    one_hot : ndarray
        Array of one-hot encoded DNA, with one_hot values along last axis
    order : str, default='ACGT'
        String representation of the order in which to encode bases. Default
        value of 'ACGT' means that A has the representation with 1 in first
        position, C with 1 in second position, etc...

    Returns
    -------
    ndarray
        Array of indexes with same shape as one_hot, with last axis collapsed.
    """
    if order != "ACGT":
        converter = np.zeros(4, dtype=int)
        for i, c in enumerate("ACGT"):
            converter[order.find(c)] = i
        one_hot = one_hot[..., converter]
    return np.argmax(one_hot, axis=-1) + 4 * (np.sum(one_hot, axis=-1) != 1)


def RC_one_hot(one_hot, order):
    """Compute reverse complement of one_hot array.

    Parameters
    ----------
    one_hot : ndarray, shape=(n, 4)
        Array of one-hot encoded DNA, with one_hot values along last axis
    order : str, default='ACGT'
        String representation of the order in which to encode bases. Default
        value of 'ACGT' means that A has the representation with 1 in first
        position, C with 1 in second position, etc...

    Returns
    -------
    ndarray
        Reverse complement of one_hot.
    """
    # Dictionary mapping base to its complement
    base_to_comp = dict(zip("ACGT", "TGCA"))
    # Array to reorder one_hot columns
    converter = np.zeros(4, dtype=int)
    for i, c in enumerate(order):
        converter[order.find(base_to_comp[c])] = i
    return one_hot[::-1, converter]


def RCdna(s):
    """Reverse complement a string DNA sequence"""
    res = []
    for c in s[::-1]:
        if c == "A":
            res.append("T")
        elif c == "a":
            res.append("t")
        elif c == "C":
            res.append("G")
        elif c == "c":
            res.append("g")
        elif c == "G":
            res.append("C")
        elif c == "g":
            res.append("c")
        elif c == "T":
            res.append("A")
        elif c == "t":
            res.append("a")
        else:
            res.append(c)
    return "".join(res)


def str_to_idx(seqstr, order="ACGT"):
    bases, index = np.unique(np.array(list(seqstr)), return_inverse=True)
    return np.array([order.find(b) for b in bases])[index]


# Sequence manipulation
def remove_reads_with_N(
    sequences, tolerance=0, max_size=None, read_length=None, verbose=False
):
    if max_size is not None:
        sequences = sequences[:max_size]
    too_short = []
    with_Ns = []
    if tolerance == 0:
        for i, seq in enumerate(sequences):
            if read_length is not None and len(seq) < read_length:
                too_short.append(i)
            if "N" in seq:
                with_Ns.append(i)
    else:
        for i, seq in enumerate(sequences):
            start_count = 0
            if read_length is not None:
                start_count = read_length - len(seq)
                assert start_count >= 0
            if seq.count("N") + start_count > tolerance:
                with_Ns.append(i)
    if verbose:
        print(too_short, " reads too short")
        print(with_Ns, " reads with Ns")
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


def find_duplicates(reads, print_freq=10_000_000, one_hot=False, batch_size=10_000_000):
    return find_duplicates_v1(
        reads, print_freq=print_freq, one_hot=one_hot, batch_size=batch_size
    )


def find_duplicates_v1(
    reads, print_freq=10_000_000, one_hot=False, batch_size=10_000_000
):
    """
    Return all unique reads and occurences.

    Can deal with string reads or one_hot reads
    """
    dico = {}
    dup = False
    n_batch = np.ceil(len(reads) / batch_size)
    if n_batch > 1:
        batches = np.split(reads, batch_size * np.arange(1, n_batch, dtype=int))
    else:
        batches = [reads]
    print(len(batches), "batches")
    for id, batch in enumerate(batches):
        print(f"Processing batch {id}")
        if one_hot:
            batch = one_hot_to_seq(batch)
        for i, read in enumerate(batch):
            if read in dico:
                dico[read] += 1
                dup = True
            else:
                dico[read] = 1
            if (i + 1) % print_freq == 0 or i + 1 == len(batch):
                msg = f"seq {i+1}/{len(batch)}"
                if dup:
                    msg += " duplicates"
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
        if (i + 1) % print_freq == 0:
            msg = f"seq {i+1}/{len(reads)}"
            if dup:
                msg += " duplicates"
            print(msg)
    return dico


def find_duplicates_v3(reads, print_freq=10_000_000, one_hot=False):
    """
    Return all unique reads and occurences.
    """
    dico = {}
    dup = False
    if one_hot:
        categories = np.array([["A"], ["C"], ["G"], ["T"]])
        encoder = OneHotEncoder(dtype=bool, handle_unknown="ignore", sparse=False)
        encoder.fit(categories)
    for i, read in enumerate(reads):
        if one_hot:
            read = encoder.inverse_transform(read).ravel()
            read = "".join(["N" if value is None else value for value in read])
        if read in dico:
            dico[read] += 1
            dup = True
        else:
            dico[read] = 1
        if (i + 1) % print_freq == 0:
            msg = f"seq {i+1}/{len(reads)}"
            if dup:
                msg += " duplicates"
            print(msg)
    return dico


def remove_duplicates(reads, print_freq=10_000_000):
    dico = find_duplicates(reads, print_freq=print_freq)
    return dico.keys()


def chunck_into_reads(long_reads, read_length=101):
    reads = []
    for i, long in enumerate(long_reads):
        chuncks = [long[i : i + read_length] for i in range(0, len(long), read_length)]
        reads.extend(chuncks)
    return reads


def reverse_complement(seq):
    reverse = ""
    for base in seq[::-1]:
        if base == "A":
            reverse += "T"
        elif base == "C":
            reverse += "G"
        elif base == "G":
            reverse += "C"
        elif base == "T":
            reverse += "A"
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
    starting_Ns = np.where(N_mask[: window_size - 1 :])[0]
    # Compute distance to previous N in last_N, considering start as N
    if len(starting_Ns) == 0:
        # No N found, previous N is the start position
        last_N = window_size - 1
    else:
        # At least one N found, previous N is at the highest position
        last_N = window_size - 2 - np.max(starting_Ns)
    for i, isN in enumerate(N_mask[window_size - 1 :]):
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
    last_N_indexes = np.arange(1, len(one_hot_seq) + 1)
    # Split at each N, and reset counter
    for split in np.split(last_N_indexes, N_idx)[1:]:
        split -= split[0]
    # Select windows by last element, if it is far enough from last N
    valid_window_mask = np.where(last_N_indexes[window_size - 1 :] >= window_size)[0]
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
    starting_Ns = np.where(N_mask[: window_size - 1 :])[0]
    if len(starting_Ns) == 0:
        # No N found, previous N is the start position
        last_N = window_size - 1
    else:
        # At least one N found, previous N is at the highest position
        last_N = window_size - 2 - np.max(starting_Ns)
    for i, isN in enumerate(N_mask[window_size - 1 :]):
        if isN:
            last_N = 0
        else:
            last_N += 1  # increase distance before testing
            if last_N >= window_size:
                # far enough from previous N for a valid window
                valid_window_idx.append(i)
    return np.array(valid_window_idx, dtype=int)


# Standard file format functions
def write_fasta(seqs: dict, fasta_file: str, wrap: int = None) -> None:
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
    with open(fasta_file, "w") as f:
        if isinstance(seqs, dict):
            iterable = seqs.items()
        else:
            iterable = enumerate(seqs)
        for id, seq in iterable:
            f.write(f">{id}\n")
            if wrap is not None:
                for i in range(0, len(seq), wrap):
                    f.write(f"{seq[i:i + wrap]}\n")
            else:
                f.write(f"{seq}\n")


def read_fasta(file):
    """Parse a fasta file as a dictionary."""
    with open(file) as f:
        genome = {}
        seq, seqname = "", ""
        for line in f:
            if line.startswith(">"):
                if seqname != "" or seq != "":
                    genome[seqname] = seq
                seqname = line[1:].rstrip()
                seq = ""
            else:
                seq += line.rstrip()
        if seq != "":
            genome[seqname] = seq
    return genome


def parse_bed_peaks(bed_file, window_size=None, remove_duplicates=False, based1=False):
    # compute offset to adjust 1-based bed indices to 0-based chromosome
    # indices, or predictions with given window
    offset = 0
    if based1:
        offset += 1
    if window_size is not None:
        offset += window_size // 2
    with open(bed_file, "r") as f:
        chr_peaks = {}
        for line in f:
            line = line.rstrip()
            splits = line.split("\t")
            if len(splits) >= 5:
                chr_id, start, end, _, score, *_ = splits
            else:
                chr_id, start, end, *_ = splits
                score = -1
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
            chr_peaks[key] = np.asarray(chr_peaks[key]) - np.array([1, 1, 0]) * offset
            try:
                # Check if some peaks overlap
                overlaps = self_overlapping_peaks(chr_peaks[key])
                assert len(overlaps) == 0
            except AssertionError:
                print(f"Warning: some peaks overlap in {key}")
    return chr_peaks


def parse_repeats(repeat_file, window_size=101, header_lines=3):
    with open(repeat_file, "r") as f:
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
                repeats[chr_id][family] = (
                    np.array(repeats[chr_id][family]) - window_size // 2
                )
    return repeats


def parse_sam(sam_file: str, verbose=True) -> None:
    with open(sam_file, "r") as f:
        chr_coord = defaultdict(list)
        header_regexp = re.compile("^@(HD|SQ|RG|PG|CO)")
        rejected_count = 0
        total_count = 0
        for line in f:
            if header_regexp.match(line):  # ignore header
                continue
            # Readline and convert some entries to int
            _, _, rname, pos, _, _, _, _, tlen, *_ = line.split("\t")
            tlen, pos = (int(v) for v in (tlen, pos))
            # Record only the leftmost read of each pair
            if tlen > 0:
                # middle = math.floor((pos + pnext + len(seq)) / 2)
                chr_coord[rname].append([pos, pos + tlen])
            else:
                rejected_count += 1
            total_count += 1
    if verbose:
        print(f"{rejected_count}/{total_count} paired reads rejected")
    return chr_coord


def parse_bam(
    bam_file: str,
    mapq_thres=None,
    verbose=True,
    paired=True,
    fragment_length=None,
    max_fragment_len=None,
    id_file=None,
) -> None:
    if id_file:
        with open(id_file) as f_id:
            ids_set = {x.split()[0][1:] for x in f_id}
    with pysam.AlignmentFile(bam_file, "rb") as f:
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
            if (
                (mapq_thres is not None and read.mapping_quality < mapq_thres)
                or (max_fragment_len is not None and tlen > max_fragment_len)
                or (id_file is not None and read.query_name not in ids_set)
            ):
                # reject the read
                rejected_count += 1
                continue
            else:
                rname = read.reference_name
                pos = read.reference_start
                chr_coord[rname].append([pos, pos + tlen])
    if verbose:
        print(f"{rejected_count}/{total_count} reads rejected")
    return chr_coord


def split_bam(
    bam_file: str, n_splits: int, rng: np.random.Generator = np.random.default_rng()
) -> None:
    """Split a bam file into multiple equally covered bam files

    Parameters
    ----------
    bam_file: str
        Pathname of the bam file to split
    n_splits: int
        Number of files to split the bam into.
    rng: np.random.Generator, optional
        Random number generator.
    """
    # Build split filenames from the input filename
    bam_file_path = Path(bam_file)
    stem = bam_file_path.stem
    if stem.endswith(".sorted"):
        stem = stem[:-7]
    prefix = str(Path(bam_file_path.parent, stem))
    split_names = [f"{prefix}_{i}.bam" for i in range(n_splits)]

    with pysam.AlignmentFile(bam_file, "rb") as f:
        # Build the split files with the input file's header and keep handles
        split_files = [
            pysam.AlignmentFile(name, "wb", template=f) for name in split_names
        ]
        # Build an iterator to randomly choose where to write next read, while balancing each file
        n_query_names = (f.mapped + f.unmapped - f.nocoordinate) / 2
        shuffled_file_idx = balanced_randint(n_splits, size=n_query_names)
        iter_file_idx = iter(shuffled_file_idx)
        # Loop over all reads to write them to the split files
        seen = {}
        for read in f.fetch():
            if read.query_name in seen:
                # The mate is already seen, write them both to a file
                try:
                    next_file_idx = next(iter_file_idx)
                except StopIteration:
                    # If Iterator is exhausted, select random file without worrying about balancing
                    next_file_idx = rng.integers(10)
                fw = split_files[next_file_idx]
                fw.write(seen[read.query_name])
                fw.write(read)
                # Remove from the dictionary to save space
                seen.pop(read.query_name)
            else:
                # Cache read in dictionary waiting for its mate
                seen[read.query_name] = read
        # Write reads for which mates couldn't be found
        print(f"{len(seen)} reads have no mate")
        # Extend iterator if needed
        shuffled_file_idx = list(iter_file_idx)
        extra_file_idx = balanced_randint(
            n_splits, size=len(seen) - len(shuffled_file_idx)
        )
        shuffled_file_idx = np.concatenate((shuffled_file_idx, extra_file_idx))
        iter_file_idx = iter(shuffled_file_idx)
        for read in seen.values():
            split_files[next(iter_file_idx)].write(read)
        # Close the split files
        for fw in split_files:
            fw.close()


def sample_bam(
    bam_file: str,
    frac: float,
    out_file: str,
    rng: np.random.Generator = np.random.default_rng(),
) -> None:
    """Randomly sample a fraction of reads in a bam file.

    For frac=0.5, it takes 2s per 1M reads

    Parameters
    ----------
    bam_file: str
        Pathname of the bam file to split.
    frac: float
        Fraction of the bam to sample.
    out_file: str
        Pathname of the output file
    rng: np.random.Generator, optional
        Random number generator.
    """
    with pysam.AlignmentFile(bam_file, "rb") as f:
        # Build output file
        with pysam.AlignmentFile(out_file, "wb", template=f) as fw:
            # Loop over all reads to write them to the split files
            seen = {}
            for read in f.fetch():
                if read.query_name in seen:
                    # choose whether to write the pair or not
                    if rng.random() < frac:
                        fw.write(seen[read.query_name])
                        fw.write(read)
                    # Remove from the dictionary to save space
                    seen.pop(read.query_name)
                else:
                    # Cache read in dictionary waiting for its mate
                    seen[read.query_name] = read
            # Write reads for which mates couldn't be found
            print(f"{len(seen)} reads have no mate")
            for read in seen.values():
                if rng.random() < frac:
                    fw.write(seen[read.query_name])
                    fw.write(read)


def inspect_bam_mapq(bam_file):
    mapqs = defaultdict(int)
    with pysam.AlignmentFile(bam_file, "rb") as f:
        for read in f.fetch():
            mapqs[read.mapping_quality] += 1
    return dict(sorted(mapqs.items()))


def load_bw(filename, nantonum=True):
    labels = {}
    with pyBigWig.open(str(filename)) as bw:
        for chr_id in bw.chroms():
            if nantonum:
                labels[chr_id] = np.nan_to_num(bw.values(chr_id, 0, -1, numpy=True))
            else:
                labels[chr_id] = bw.values(chr_id, 0, -1, numpy=True)
    return labels


def write_bw(filename, signals):
    bw = pyBigWig.open(str(filename), "w")
    bw.addHeader([(k, len(v)) for k, v in signals.items()])
    for chr_id, val in signals.items():
        bw.addEntries(chr_id, 0, values=val, span=1, step=1)
    bw.close()


def load_annotation(file, chr_id, window_size, anchor="center"):
    bw = pyBigWig.open(file)
    values = bw.values(f"chr{chr_id}", 0, -1, numpy=True)
    values[np.isnan(values)] = 0
    values = adapt_to_window(values, window_size, anchor=anchor)
    return values


def adapt_to_window(
    values: np.ndarray, window_size: int, anchor: str = "center"
) -> np.ndarray:
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
    if anchor == "center":
        return values[(window_size // 2) : (-((window_size + 1) // 2) + 1)]
    elif anchor == "start":
        return values[: -window_size + 1]
    elif anchor == "end":
        return values[window_size - 1 :]
    else:
        raise ValueError("Choose anchor from 'center', 'start' or 'end'")


# GC content
def GC_content(one_hot_reads: np.ndarray, order: int = "ACGT") -> np.ndarray:
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
    g_idx, c_idx = order.find("G"), order.find("C")
    gc = (content[:, g_idx] + content[:, c_idx]) / np.sum(content, axis=1)
    return gc


def sliding_GC(
    seqs: np.ndarray, n: int, axis: int = -1, order: str = "ACGT", form: str = "token"
) -> np.ndarray:
    """Compute sliding GC content on encoded DNA sequences

    Sequences can be either tokenized or one-hot encoded.
    GC content will be computed by considering only valid tokens or one-hots.
    Valid tokens are between in the range [0, 4[, and valid one_hots have exactly one value equal to 1. However we check only if they sum to one.

    Parameters
    ----------
    seqs: ndarray
        Input sequences. Sequences are assumed to be read along last axis, otherwise change `axis` parameter.
    n: int
        Length of window to compute GC content on, must be greater than 0.
    axis: int, optional
        Axis along which to compute GC content. The default (-1) assumes sequences are read along last axis.
        Currently, form 'one_hot' doesn't support the axis parameter, it assumes one-hot values on last axis, and sequence on second to last axis.
    order: str, optional
        Order of encoding, must contain each letter of ACGT exactly once.
        If `form` is 'token', then value i corresponds to base at index i in order.
        If `form` is 'one_hot', then vector of zeros with a 1 at position i corresponds to base at index i in order.
    form: {'token', 'one_hot'}, optional
        Form of input array. 'token' for indexes of bases and 'one_hot' for one-hot encoded sequences, with an extra dimension.

    Returns
    -------
    ndarray
        Array of sliding GC content, with size along `axis` reduced by `n`-1, and optional one-hot dimension removed.
    """
    if form == "token":
        valid_mask = (seqs >= 0) & (seqs < 4)
        GC_mask = (seqs == order.find("C")) | (seqs == order.find("G"))
        if n > seqs.shape[axis]:
            n = seqs.shape[axis]
        return moving_sum(GC_mask, n, axis=axis) / moving_sum(valid_mask, n, axis=axis)
    elif form == "one_hot":
        valid_mask = seqs.sum(axis=-1) != 0
        GC_mask = seqs[:, [order.find("C"), order.find("G")]].sum(axis=-1)
        if n > seqs.shape[-1]:
            n = seqs.shape[-1]
        return moving_sum(GC_mask, n=n, axis=-1) / moving_sum(valid_mask, n=n, axis=-1)
    else:
        raise ValueError(f"form must be 'token' or 'one_hot', not {form}")


def classify_1D(features, y, bins):
    """Find best threshold to classify 1D features with label y.

    Computing is done in bins for fast execution, so it isn't exact
    """

    def cumul_count(features, bins):
        feature_bins = np.digitize(features, bins).ravel()
        count = np.bincount(feature_bins, minlength=len(bins) + 1)[1:]
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
    thres = (bins[bin_thres] + bins[bin_thres + 1]) / 2
    return accuracy, thres


# Signal manipulation
def z_score(preds, rel_indices=None):
    if rel_indices is not None:
        rel_preds = preds[rel_indices]
        mean, std = np.mean(rel_preds), np.std(rel_preds)
    else:
        mean, std = np.mean(preds), np.std(preds)
    return (preds - mean) / std


def smooth(values, window_size, mode="linear", sigma=1, padding="same"):
    if mode == "linear":
        box = np.ones(window_size) / window_size
    elif mode == "gaussian":
        box = gaussian(window_size, sigma)
        box /= np.sum(box)
    elif mode == "triangle":
        box = np.concatenate(
            (
                np.arange((window_size + 1) // 2),
                np.arange(window_size // 2 - 1, -1, -1),
            ),
            dtype=float,
        )
        box /= np.sum(box)
    else:
        raise NameError("Invalid mode")
    return convolve(values, box, mode=padding)


def binned_alignment_count_from_coord(
    coord: np.ndarray, binsize: int = 100, length: int = None
) -> np.ndarray:
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
    return (
        coo_matrix(
            (
                np.ones(len(coord), dtype=int),
                (binned_mid, np.zeros(len(coord), dtype=int)),
            ),
            shape=(length, 1),
        )
        .toarray()
        .ravel()
    )


def exact_alignment_count_from_coord(
    coord: np.ndarray, length: int = None
) -> np.ndarray:
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
    data = np.ones(2 * len(coord), dtype=int)
    data[1::2] = -1
    # Insert using scipy.sparse implementation
    start_ends = (
        coo_matrix(
            (data, (coord.ravel(), np.zeros(2 * len(coord), dtype=int))),
            shape=(length + 1, 1),  # length+1 because need a -1 after last end
        )
        .toarray()
        .ravel()
    )
    # Cumulative sum to propagate full fragments,
    # remove last value which is always 0
    return np.cumsum(start_ends)[:-1]


def bin_values(array: np.ndarray, binsize: int, func=np.mean):
    """Compute summary statistics on bins of an array

    If array length isn't divisible by binsize, the last bin will be smaller

    Parameters
    ----------
    array: np.ndarray
        1D input data
    binsize: int
        length of bins to consider, must be greater than 0
    func: callable
        function computing summary statistic, must support axis parameter
        (ex: np.mean, np.sum)

    Returns
    -------
    """
    if binsize <= 0:
        raise ValueError("binsize must be greater than 0")
    nbins, r = divmod(len(array), binsize)
    res = func(array[: nbins * binsize].reshape(nbins, binsize), axis=1)
    if r != 0:
        res = np.append(res, func(array[-r:]))
    return res


def full_genome_binned_preds(pred_file, chr_sizes_file, binsize, chr_ids):
    with open(chr_sizes_file, "r") as f:
        chr_lens = json.load(f)
    binned_lengths = np.array([x // binsize + 1 for x in chr_lens.values()])
    separators = np.cumsum(binned_lengths)
    df = np.zeros(separators[-1])
    # merging chromosomes
    for i, chr_id in enumerate(chr_ids.keys()):
        with np.load(pred_file) as f:
            preds = f[f"chr{chr_id}"]
        binned_preds = bin_values(preds, binsize)
        df[separators[i] - len(binned_preds) : separators[i]] = binned_preds
    return df, separators


def enrichment_analysis(signal, ctrl, verbose=True, data="signal"):
    n_binom = signal + ctrl
    p_binom = np.sum(signal) / np.sum(n_binom)
    binom_pval = clip_to_nonzero_min(
        1 - scipy.stats.binom.cdf(signal - 1, n_binom, p_binom)
    )
    reject, binom_qval, *_ = multitest.multipletests(binom_pval, method="fdr_bh")
    signif_qval = reject
    if verbose:
        print(
            f"{np.sum(reject)}/{len(reject)} " f"significantly enriched bins in {data}"
        )
    return pd.DataFrame(
        {
            "pval": binom_pval,
            "qval": binom_qval,
            "-log_pval": -np.log10(binom_pval),
            "-log_qval": -np.log10(binom_qval),
            "signif_qval": signif_qval,
        }
    )


def genome_enrichment(
    ip_coord_file,
    ctrl_coord_file,
    chr_sizes_file,
    out_file,
    binsize,
    max_frag_len=500,
    from_bw=False,
):
    def process_coord_file(coord_file):
        """Pipeline for both ip and ctrl coord files"""
        try:
            if from_bw:
                with pyBigWig.open(coord_file) as bw:
                    mids = bw.values(chr_id, 0, -1, numpy=True)
                count_chr = bin_values(mids, binsize, func=np.sum)
            else:
                # Load chromosome fragment coordinates
                with np.load(coord_file) as f:
                    coord_chr = f[chr_id]
                # Filter out fragments too long
                frag_lens_chr = np.diff(coord_chr, axis=1).ravel()
                coord_chr = coord_chr[frag_lens_chr <= max_frag_len, :]
                # Get binned count of mid points
                count_chr = binned_alignment_count_from_coord(
                    coord_chr, binsize=binsize, length=chr_lens[chr_id]
                )
                with open(log_file, "a") as f:
                    f"{np.sum(frag_lens_chr >= max_frag_len)} fragments "
                    f"longer than {max_frag_len}bp in {coord_file}\n"
        except KeyError:
            count_chr = np.zeros(chr_lens[chr_id] // binsize + 1)
            with open(log_file, "a") as f:
                f"No reads in {coord_file}\n"
        return count_chr

    # Log parameters
    out_file = Path(out_file)
    out_file = safe_filename(out_file)
    log_file = Path(out_file.parent, out_file.stem + "_log.txt")
    log_file = safe_filename(log_file)
    with open(log_file, "w") as f:
        f.write(
            f"IP coordinates file: {ip_coord_file}\n"
            f"Control coordinates file: {ctrl_coord_file}\n"
            f"chromosome sizes file: {chr_sizes_file}\n"
            f"output file: {out_file}\n"
            f"bin size: {binsize}\n"
            f"max fragment length: {max_frag_len}\n\n"
        )
    # Get chromosome lengths
    with open(chr_sizes_file, "r") as f:
        chr_lens = json.load(f)
    # Build DataFrame
    mindex = pd.MultiIndex.from_tuples(
        [
            (chr_id, pos)
            for chr_id in chr_lens.keys()
            for pos in np.arange(0, chr_lens[chr_id], binsize)
        ],
        names=["chr", "pos"],
    )
    columns = ["ip_count", "ctrl_count", "pval", "qval"]
    df = pd.DataFrame(0, index=mindex, columns=columns)
    # Loop over chromosomes
    for chr_id in chr_lens.keys():
        with open(log_file, "a") as f:
            f.write(f"Processing {chr_id}...\n")
        ip_count_chr = process_coord_file(ip_coord_file)
        ctrl_count_chr = process_coord_file(ctrl_coord_file)
        # Insert in DataFrame
        df.loc[chr_id, :"ctrl_count"] = np.transpose(
            np.vstack((ip_count_chr, ctrl_count_chr))
        )
    # Compute p-values and q-values
    n_binom = df["ip_count"] + df["ctrl_count"]
    p_binom = np.sum(df["ip_count"]) / np.sum(n_binom)
    df["pval"] = clip_to_nonzero_min(
        1 - scipy.stats.binom.cdf(df["ip_count"] - 1, n_binom, p_binom)
    )
    _, df["qval"], *_ = multitest.multipletests(df["pval"], method="fdr_bh")
    # Save final DataFrame
    df.to_csv(out_file)


def downsample_enrichment_analysis(
    data,
    genome,
    max_frag_len,
    binsizes=[1000],
    fracs=[1],
    divs=None,
    reverse=True,
    data_dir="../shared_folder",
    basename="",
    use_fdr=True,
):
    # Convert divs to fracs
    if divs is not None:
        fracs = 1 / np.array(divs)
    # Build resulting DataFrame
    mindex = pd.MultiIndex.from_product([binsizes, fracs])
    if reverse:
        res = pd.DataFrame(
            index=mindex,
            columns=[
                "IP",
                "IP_clust",
                "Undetermined",
                "Ctrl",
                "Ctrl_clust",
                "total_cov",
            ],
        )
    else:
        res = pd.DataFrame(
            index=mindex, columns=["IP", "IP_clust", "Undetermined", "total_cov"]
        )
    # Start analysis
    for binsize in binsizes:
        # Load alignment data
        df = pd.read_csv(
            Path(
                data_dir,
                data,
                "results",
                "alignments",
                genome,
                f"{data}_{genome}_{basename}maxfraglen_{max_frag_len}_"
                f"binsize_{binsize}.csv",
            ),
            index_col=0,
        )
        for frac in fracs:
            # Randomly sample alignment histogram
            frac_IP = integer_histogram_sample(df["ip_count"], frac)
            frac_Ctrl = integer_histogram_sample(df["ctrl_count"], frac)
            # Compute p-values
            n = frac_IP + frac_Ctrl
            cov = np.sum(n)
            p_binom = np.sum(frac_IP) / cov
            pval = clip_to_nonzero_min(
                1 - scipy.stats.binom.cdf(frac_IP - 1, n, p_binom)
            )
            # Extract significant IP bins
            if use_fdr:
                # correct with q-value on non-empty bins
                valid_bins = n != 0
                signif_IP = np.zeros(len(df), dtype=bool)
                signif_IP[valid_bins], *_ = multitest.multipletests(
                    pval[valid_bins], method="fdr_bh"
                )
            else:
                signif_IP = np.array(pval < 0.05)
            n_signif_IP = np.sum(signif_IP)
            n_signif_IP_clust = nb_boolean_true_clusters(signif_IP)
            # Extract significant Ctrl bins too
            if reverse:
                rev_pval = clip_to_nonzero_min(
                    1 - scipy.stats.binom.cdf(frac_Ctrl - 1, n, 1 - p_binom)
                )
                if use_fdr:
                    signif_Ctrl = np.zeros(len(df), dtype=bool)
                    signif_Ctrl[valid_bins], *_ = multitest.multipletests(
                        rev_pval[valid_bins], method="fdr_bh"
                    )
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
                    cov,
                ]
            else:
                res.loc[binsize, frac] = [
                    n_signif_IP,
                    n_signif_IP_clust,
                    len(df) - n_signif_IP,
                    cov,
                ]
    return res


def pool_experiments(dfs, verbose=True):
    cols_to_take = ["ip_count", "ctrl_count"]
    df_pooled = dfs[0][["pos"] + cols_to_take].copy()
    for df in dfs[1:]:
        df_pooled[cols_to_take] += df[cols_to_take]
    # computing p_value and q_value
    sums = df_pooled.sum(axis=0)
    p_binom = sums["ip_count"] / (sums["ip_count"] + sums["ctrl_count"])
    n_binom = df_pooled["ip_count"] + df_pooled["ctrl_count"]
    df_pooled["pval"] = clip_to_nonzero_min(
        1 - scipy.stats.binom.cdf(df_pooled["ip_count"] - 1, n_binom, p_binom)
    )
    reject, df_pooled["qval"], *_ = multitest.multipletests(
        df_pooled["pval"], method="fdr_bh"
    )
    df_pooled["-log_qval"] = -np.log10(df_pooled["qval"])
    df_pooled["-log_pval"] = -np.log10(df_pooled["pval"])
    df_pooled["signif_qval"] = reject
    if verbose:
        print(
            f"{np.sum(reject)}/{len(reject)} "
            f"significantly enriched bins in dataframe"
        )
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
            print(
                "Arguments must be DataFrames using MultiIndex with 2 "
                "levels, and second level must have at least 2 values"
            )
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
        [
            (chr_id, pos)
            for chr_id in df.index.levels[0]
            for pos in range(0, df.loc[chr_id].index.max() + binsize, binsize_ref)
        ],
        names=["chr", "pos"],
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
    seps = np.append(np.where(df["pos"] == 0)[0], len(df))
    seps_ref = np.append(np.where(df_ref["pos"] == 0)[0], len(df_ref))
    # get binsize scale
    scale = binsize // binsize_ref
    # repeat df2 `scale` times and reset indices
    subdf = df.reindex(df.index.repeat(scale))
    subdf = subdf.reset_index().drop("index", axis=1)
    # cut chromosome edges to conform to df1
    cumul = 0
    for i in range(1, len(seps_ref)):
        subdf.drop(range(seps_ref[i] + cumul, seps[i] * scale), inplace=True)
        cumul = seps[i] * scale - seps_ref[i]
    # reset indices again
    return subdf.reset_index().drop("index", axis=1)


def make_mindex_ser(
    annotation_file: Path,
    chr_sizes_file: Path,
    out_file: Path,
    binsize: int,
    name: str = None,
    coords: bool = False,
    process_func: Callable = None,
    annot_ids_dict: dict = None,
    **kwargs,
):
    """Build a binned MultiIndex Series from an annotation file."""
    # Log parameters
    out_file = safe_filename(out_file)
    log_file = Path(out_file.parent, out_file.stem + "_log.txt")
    log_file = safe_filename(log_file)
    with open(log_file, "w") as f:
        f.write(
            f"annotation file: {annotation_file}\n"
            f"chromosome sizes file: {chr_sizes_file}\n"
            f"output file: {out_file}\n"
            f"bin size: {binsize}\n"
            f"coords: {coords}\n"
            f"process function: {process_func}\n"
            f"annot_ids_dict: {annot_ids_dict}\n\n"
        )
    # Get chromosome lengths
    with open(chr_sizes_file, "r") as f:
        chr_lens = json.load(f)
    # Build MultiIndex
    mindex = pd.MultiIndex.from_tuples(
        [
            (chr_id, pos)
            for chr_id, chr_len in chr_lens.items()
            for pos in np.arange(0, chr_len, binsize)
        ],
        names=["chr", "pos"],
    )
    ser = pd.Series(0, index=mindex, name=name)
    # Loop over chromosomes
    for chr_id, chr_len in chr_lens.items():
        with open(log_file, "a") as f:
            f.write(f"Processing {chr_id}...\n")
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
                        annot_chr, binsize=binsize, length=chr_len
                    )
                else:
                    annot_chr = bin_values(annot_chr, binsize)
            except KeyError:
                annot_chr = np.zeros(chr_len // binsize + 1)
                with open(log_file, "a") as f:
                    f.write(f"No annotation for {chr_id} in {annotation_file}\n")
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
    # The correlation is computed on as many values for each offset.
    # The results may be different depending on the max and min offset
    min_offset = np.min(offsets)
    max_offset = np.max(offsets)
    negmin = max(-min_offset, 0)
    max_len = len(X) - max_offset - negmin
    offsets += negmin
    windows = offsets.reshape(-1, 1) + np.arange(max_len).reshape(1, -1)
    X_slides = X[windows]
    slide_corrs = lineWiseCorrcoef(X_slides, Y[negmin : negmin + max_len])
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
def make_peaks(
    peak_mask: np.ndarray, length_thres: int = 1, tol: int = 1
) -> np.ndarray:
    """Format peak array from peak boolean mask.

    Determine regions of consecutive high prediction, called peaks.

    Parameters
    ----------
    peak_mask : ndarray
        1D-array of boolean values along the chromosome
    length_thres : int, default=1
        Minimum length required for peaks, any peak strictly below that
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
    if tol > 1:
        # Compute difference between end of peak and start of next one
        diffs = change_idx[2::2] - change_idx[1:-1:2]
        # Get index when difference is below threshold, see below for matching
        # index in diffs and in change_idx
        # diff index:   0   1   2  ...     n-1
        # change index:1-2 3-4 5-6 ... (2n-1)-2n
        (small_diff_idx,) = np.where(diffs <= tol)
        delete_idx = np.concatenate((small_diff_idx * 2 + 1, small_diff_idx * 2 + 2))
        # Remove close ends and starts using boolean mask
        mask = np.ones(len(change_idx), dtype=bool)
        mask[delete_idx] = False
        change_idx = change_idx[mask]
    # Reshape as starts and ends
    peaks = np.reshape(change_idx, (-1, 2))
    # Compute lengths of peaks and remove the ones below given threshold
    lengths = np.diff(peaks, axis=1).ravel()
    peaks = peaks[lengths >= length_thres]
    return peaks


def find_peaks(
    preds: np.ndarray, pred_thres: float, length_thres: int = 1, tol: int = 1
) -> np.ndarray:
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
    peak_mask = preds > pred_thres
    return make_peaks(peak_mask, length_thres, tol)


def find_peaks_in_window(
    peaks: np.ndarray, window_start: int, window_end: int
) -> np.ndarray:
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
    valid_peaks = sorted_peaks[(first_id // 2) : ((last_id + 1) // 2), :]
    # Cut first and last peaks if they exceed window size
    if first_id % 2 == 1:
        valid_peaks[0, 0] = window_start
    if last_id % 2 == 1:
        valid_peaks[-1, 1] = window_end - 1
    return valid_peaks


def overlap(
    peak0: np.ndarray, peak1: np.ndarray, tol: int = 0
) -> tuple:  # tuple[bool, bool]:
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


def self_overlapping_peaks(
    peaks: np.ndarray, merge: bool = False, tol: int = 1
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
    diffs = -np.diff(gaps, axis=1).ravel()
    (overlap_idx,) = np.where(diffs >= -tol)
    if merge:
        if len(overlap_idx) != 0:
            # Compute indices for the full flatten array
            delete_idx = np.concatenate((overlap_idx * 2 + 1, overlap_idx * 2 + 2))
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


def sliding_window_view(x, window_shape, axis=None, *, subok=False, writeable=False):
    """Function from the numpy library"""
    window_shape = tuple(window_shape) if np.iterable(window_shape) else (window_shape,)
    # first convert input to array, possibly keeping subclass
    x = np.array(x, copy=False, subok=subok)

    window_shape_array = np.array(window_shape)
    if np.any(window_shape_array < 0):
        raise ValueError("`window_shape` cannot contain negative values")

    if axis is None:
        axis = tuple(range(x.ndim))
        if len(window_shape) != len(axis):
            raise ValueError(
                f"Since axis is `None`, must provide "
                f"window_shape for all dimensions of `x`; "
                f"got {len(window_shape)} window_shape elements "
                f"and `x.ndim` is {x.ndim}."
            )
    else:
        axis = normalize_axis_tuple(axis, x.ndim, allow_duplicate=True)
        if len(window_shape) != len(axis):
            raise ValueError(
                f"Must provide matching length window_shape and "
                f"axis; got {len(window_shape)} window_shape "
                f"elements and {len(axis)} axes elements."
            )

    out_strides = x.strides + tuple(x.strides[ax] for ax in axis)

    # note: same axis can be windowed repeatedly
    x_shape_trimmed = list(x.shape)
    for ax, dim in zip(axis, window_shape):
        if x_shape_trimmed[ax] < dim:
            raise ValueError("window shape cannot be larger than input array shape")
        x_shape_trimmed[ax] -= dim - 1
    out_shape = tuple(x_shape_trimmed) + window_shape
    return as_strided(
        x, strides=out_strides, shape=out_shape, subok=subok, writeable=writeable
    )


def strided_window_view(
    x, window_shape, stride, axis=None, *, subok=False, writeable=False
):
    """Variant of `sliding_window_view` which supports stride parameter.

    The axis parameter doesn't work, the stride can be of same shape as
    window_shape, providing different stride in each dimension. If shorter
    than window_shape, stride will be filled with ones. it also doesn't
    support multiple windowing on same axis"""
    window_shape = tuple(window_shape) if np.iterable(window_shape) else (window_shape,)
    # first convert input to array, possibly keeping subclass
    x = np.array(x, copy=False, subok=subok)

    window_shape_array = np.array(window_shape)
    if np.any(window_shape_array < 0):
        raise ValueError("`window_shape` cannot contain negative values")

    if axis is None:
        axis = tuple(range(x.ndim))
        if len(window_shape) != len(axis):
            raise ValueError(
                f"Since axis is `None`, must provide "
                f"window_shape for all dimensions of `x`; "
                f"got {len(window_shape)} window_shape elements "
                f"and `x.ndim` is {x.ndim}."
            )
    else:
        axis = normalize_axis_tuple(axis, x.ndim, allow_duplicate=True)
        if len(window_shape) != len(axis):
            raise ValueError(
                f"Must provide matching length window_shape and "
                f"axis; got {len(window_shape)} window_shape "
                f"elements and {len(axis)} axes elements."
            )

    # ADDED THIS ####
    stride = tuple(stride) if np.iterable(stride) else (stride,)
    stride_array = np.array(stride)
    if np.any(stride_array < 0):
        raise ValueError("`stride` cannot contain negative values")
    if len(stride) > len(window_shape):
        raise ValueError("`stride` cannot be longer than `window_shape`")
    elif len(stride) < len(window_shape):
        stride += (1,) * (len(window_shape) - len(stride))
    ########################

    # CHANGED THIS LINE ####
    # out_strides = x.strides + tuple(x.strides[ax] for ax in axis)
    # TO ###################
    out_strides = tuple(x.strides[ax] * stride[ax] for ax in range(x.ndim)) + tuple(
        x.strides[ax] for ax in axis
    )
    ########################

    # note: same axis can be windowed repeatedly
    x_shape_trimmed = list(x.shape)
    for ax, dim in zip(axis, window_shape):
        if x_shape_trimmed[ax] < dim:
            raise ValueError("window shape cannot be larger than input array shape")
        # CHANGED THIS LINE ####
        # x_shape_trimmed[ax] -= dim - 1
        # TO ###################
        x_shape_trimmed[ax] = int(np.ceil((x_shape_trimmed[ax] - dim + 1) / stride[ax]))
        ########################
    out_shape = tuple(x_shape_trimmed) + window_shape
    return as_strided(
        x, strides=out_strides, shape=out_shape, subok=subok, writeable=writeable
    )


def strided_sliding_window_view(
    x, window_shape, stride, sliding_len, axis=None, *, subok=False, writeable=False
):
    """Variant of `strided_window_view` which slides in between strides.

    This will provide blocks of sliding window of `sliding_len` windows,
    with first windows spaced by `stride`
    The axis parameter determines where the stride and slide are performed, it
    can only be a single value."""
    window_shape = tuple(window_shape) if np.iterable(window_shape) else (window_shape,)
    # first convert input to array, possibly keeping subclass
    x = np.array(x, copy=False, subok=subok)

    window_shape_array = np.array(window_shape)
    if np.any(window_shape_array < 0):
        raise ValueError("`window_shape` cannot contain negative values")

    # ADDED THIS ####
    stride = tuple(stride) if np.iterable(stride) else (stride,)
    stride_array = np.array(stride)
    if np.any(stride_array < 0):
        raise ValueError("`stride` cannot contain negative values")
    if len(stride) == 1:
        stride += (1,)
    elif len(stride) > 2:
        raise ValueError("`stride` cannot be of length greater than 2")
    if sliding_len % stride[1] != 0:
        raise ValueError("second `stride` must divide `sliding_len` exactly")
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
    out_strides = (
        x.strides[:axis]
        + (x.strides[axis] * stride[0], x.strides[axis] * stride[1])
        + x.strides[axis:]
    )
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
    x_shape_trimmed = [
        (x.shape[axis] - window_shape[axis] - sliding_len + stride[1]) // stride[0] + 1,
        sliding_len // stride[1],
    ]
    out_shape = window_shape[:axis] + tuple(x_shape_trimmed) + window_shape[axis:]
    ########################
    return as_strided(
        x, strides=out_strides, shape=out_shape, subok=subok, writeable=writeable
    )


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
    DX = X - (np.einsum("ij->i", X) / np.double(n)).reshape((-1, 1))
    y -= np.einsum("i->", y) / np.double(n)
    tmp = np.einsum("ij,ij->i", DX, DX)
    tmp *= np.einsum("i,i->", y, y)
    return np.dot(DX, y) / np.sqrt(tmp)


def vcorrcoef(X, Y):
    Xm = np.reshape(np.mean(X, axis=1), (X.shape[0], 1))
    Ym = np.reshape(np.mean(Y, axis=1), (Y.shape[0], 1))
    r_num = np.sum((X - Xm) * (Y - Ym), axis=1)
    r_den = np.sqrt(np.sum((X - Xm) ** 2, axis=1) * np.sum((Y - Ym) ** 2, axis=1))
    r = r_num / r_den
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
    return np.sum(np.min(merge, axis=0) * sign, axis=-1) / np.sum(
        np.max(merge, axis=0), axis=-1
    )


def moving_average(x, n=2, keepsize=False):
    if keepsize:
        x = np.concatenate(
            [np.zeros(n // 2, dtype=x.dtype), x, np.zeros((n - 1) // 2, dtype=x.dtype)]
        )
    ret = np.cumsum(x)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def slicer_on_axis(
    arr: np.ndarray,
    slc: Union[slice, Iterable[slice]],
    axis: Union[None, int, Iterable[int]] = None,
) -> Tuple[slice]:
    """Build slice of array along specified axis.

    This function can be used to build slices for arrays with many or unknown number of dimensions.

    Parameters
    ----------
    arr: ndarray
        Input array
    slc: slice or iterable of slices
        Slices of the array to take.
    axis: None or int or iterable of ints
        Axis along which to perform slicing. The default (None) is to take slices along first len(`slc`) dimensions.

    Returns
    -------
    tuple of slices
        Full tuple of slices to use to slice array

    Examples
    --------
    >>> arr = np.arange(24).reshape(2, 3, 4)
    >>> arr
    array([[[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]],

           [[12, 13, 14, 15],
            [16, 17, 18, 19],
            [20, 21, 22, 23]]])
    >>> arr[slicer_on_axis(arr, slice(1, 3), axis=2)]
    array([[[ 1,  2],
            [ 5,  6],
            [ 9, 10]],

           [[13, 14],
            [17, 18],
            [21, 22]]])
    >>> arr[slicer_on_axis(arr, slice(2, None), axis=1)]
    array([[[ 8,  9, 10, 11]],

           [[20, 21, 22, 23]]])
    >>> arr[slicer_on_axis(arr, slice(None, -1))]
    array([[[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]]])
    >>> arr[slicer_on_axis(arr, [slice(None, -1), slice(1, 3)], axis=[0, 2])]
    array([[[ 1,  2],
            [ 5,  6],
            [ 9, 10]]])
    >>> arr[slicer_on_axis(arr, [slice(None, -1), slice(1, 3)])]
    array([[[ 4,  5,  6,  7],
            [ 8,  9, 10, 11]]])
    """
    full_slice = [slice(None)] * arr.ndim
    if isinstance(slc, slice):
        if axis is None:
            axis = 0
        if isinstance(axis, int):
            full_slice[axis] = slc
        else:
            for ax in axis:
                full_slice[ax] = slc
    else:
        if axis is None:
            axis = list(range(len(slc)))
        elif not isinstance(axis, Iterable):
            raise ValueError("if slc is an iterable, axis must be an iterable too")
        elif len(axis) != len(slc):
            raise ValueError("axis and slc must have same length")
        for s, ax in zip(slc, axis):
            if full_slice[ax] != slice(None):
                raise ValueError("Can't set slice on same axis twice")
            full_slice[ax] = s
    return tuple(full_slice)


def moving_sum(arr: np.ndarray, n: int, axis: Union[None, int] = None) -> np.ndarray:
    """Compute moving sum of array

    Parameters
    ----------
    arr: ndarray
        Input array
    n: int
        Length of window to compute sum on, must be greater than 0
    axis: None or int, optional
        Axis along which the moving sum is computed, the default (None) is to compute the moving sum over the flattened array.

    Returns
    -------
    ndarray
        Array of moving sum, with size along `axis` reduced by `n`-1.

    Examples
    --------
    >>> moving_sum(np.arange(10), n=2)
    array([ 1,  3,  5,  7,  9, 11, 13, 15, 17])
    >>> arr = np.arange(24).reshape(2, 3, 4)
    >>> moving_sum(arr, n=2, axis=-1)
    array([[[ 1,  3,  5],
            [ 9, 11, 13],
            [17, 19, 21]],

           [[25, 27, 29],
            [33, 35, 37],
            [41, 43, 45]]])
    """
    if n <= 0:
        raise ValueError(f"n must be greater than 0, but is equal to {n}")
    elif axis is None and n > arr.size:
        raise ValueError(
            f"Can't compute moving_sum of {n} values on flattened array with length {arr.size}"
        )
    elif axis is not None and n > arr.shape[axis]:
        raise ValueError(
            f"Can't compute moving_sum of {n} values on axis {axis} with length {arr.shape[axis]}"
        )
    res = np.cumsum(arr, axis=axis)
    res[slicer_on_axis(res, slice(n, None), axis=axis)] = (
        res[slicer_on_axis(res, slice(n, None), axis=axis)]
        - res[slicer_on_axis(res, slice(None, -n), axis=axis)]
    )
    return res[slicer_on_axis(res, slice(n - 1, None), axis=axis)]


def repeat_along_diag(a, r):
    """Construct a matrix by repeating a sub_matrix along the diagonal.

    References
    ----------
    https://stackoverflow.com/questions/33508322/create-block-diagonal-numpy-array-from-a-given-numpy-array
    """
    m, n = a.shape
    out = np.zeros((r, m, r, n), dtype=a.dtype)
    diag = np.einsum("ijik->ijk", out)
    diag[:] = a
    return out.reshape(-1, n * r)


def exp_normalize(x, axis=-1):
    res = np.exp(x - np.max(x, axis=axis, keepdims=True))
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


def zero_to_epsilon(array: np.ndarray, copy: bool = True) -> np.ndarray:
    """Change all zero values to the minimal non-zero value.

    Parameters
    ----------
    array : np.ndarray
        Input array

    Returns
    -------
    np.ndarray
        Array with zeros replaced
    """
    if copy:
        array = array.copy()
    array[array == 0] = np.finfo(array.dtype).eps
    return array


def clip_to_nonzero_min(array: np.ndarray, copy: bool = True) -> np.ndarray:
    """Change all zero values to the minimal non-zero value.

    Parameters
    ----------
    array : np.ndarray
        Input array

    Returns
    -------
    np.ndarray
        Array with zeros replaced
    """
    if copy:
        array = array.copy()
    array[array == 0] = array[array != 0].min()
    return array


def clipnorm(signals, q=0.99):
    """Clip max signal to a given quantile and normalize between 0 and 1."""
    full = np.concatenate(list(signals.values()))
    quant = np.quantile(full, q)
    return {k: np.clip(v, None, quant) / quant for k, v in signals.items()}


def nb_boolean_true_clusters(array: np.ndarray) -> int:
    """Compute the number of clusters of True values in array.

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


def random_rounding(
    array: np.ndarray, rng: np.random.Generator = np.random.default_rng()
) -> np.ndarray:
    rounded = np.floor(array)
    decimal = array - rounded
    rounded += rng.random(len(decimal)) <= decimal
    return rounded


def integer_histogram_sample(
    array: np.ndarray,
    frac: float,
    return_complement: bool = False,
    rng: np.random.Generator = np.random.default_rng(),
) -> np.ndarray:
    """Randomly sample a fraction of a histogram with integer-only values.

    The sampled histogram is a an array of integers of same shape as the
    original histogram, with all values smaller of equal to original histogram
    values.

    Parameters
    ----------
    array : np.ndarray
        1D-array of integer values.
    frac : float
        fraction of the histogram to sample, the cumulative sum of the sampled
        histogram will be the rounded fraction of the original one
    return_complement: bool
        If True, return the complement sample as well
    rng: np.random.Generator, optional
        Random number generator.

    Returns
    -------
    np.ndarray or tuple of np.ndarray
        1D-array of same length as `array`, containing the sampled histogram
        values, or if return_complement is set, a tuple containing the sample
        and its complement
    """

    def get_subhistogram(frac: int) -> np.ndarray:
        sampled_pos = rng.choice(
            positions, size=round(len(positions) * frac), replace=False
        )
        histogram = (
            scipy.sparse.coo_matrix(
                (
                    np.ones(len(sampled_pos), dtype=int),
                    (sampled_pos, np.zeros(len(sampled_pos), dtype=int)),
                ),
                shape=(len(array), 1),
            )
            .toarray()
            .ravel()
        )
        return histogram

    positions = np.repeat(np.arange(len(array), dtype=int), array)
    # get_subhistogram complexity is linear in frac, computing complement may save time
    if frac <= 0.5:
        res = get_subhistogram(frac)
        if return_complement:
            return res, array - res
        else:
            return res
    else:
        comp = get_subhistogram(1 - frac)
        if return_complement:
            return array - comp, comp
        else:
            return array - comp


def integer_histogram_serie_sample(
    counts: pd.Series,
    frac: float = 0.5,
    return_complement: bool = False,
    dtype=np.int32,
    rng: np.random.Generator = np.random.default_rng(),
) -> pd.Series:
    """Randomly sample a fraction of a histogram with integer-only values.

    The sampled histogram is a an series of integers of same shape as the
    original histogram, with all values smaller of equal to original histogram
    values. This version is faster than integer_histogram_sample when there are
    not too many unique values relative to the total count of the histogram. It
    also uses less memory.

    Parameters
    ----------
    counts : pd.Series
        series of integer values.
    frac : float
        fraction of the histogram to sample, the cumulative sum of the sampled
        histogram will be the rounded fraction of the original one
    return_complement : bool
        If True, return the complement sample as well
    rng : np.random.Generator, optional
        Random number generator

    Returns
    -------
    pd.Series or tuple of pd.Series
        Series of same length as `counts`, containing the sampled histogram
        values, or if return_complement is set, a tuple containing the sample
        and its complement
    """
    s1 = pd.Series(data=0, index=counts.index, dtype=dtype)

    gb = counts[counts > 0].groupby(counts)
    for value, g in gb:
        # indices = g.index
        s1[g.index] = rng.binomial(value, frac, size=len(g))

    if return_complement:
        s2 = counts - s1
        return s1.astype(dtype), s2.astype(dtype)
    else:
        return s1.astype(dtype)


def integer_histogram_sample_vect(
    array: np.ndarray,
    frac: np.ndarray,
    rng: np.random.Generator = np.random.default_rng(),
) -> np.ndarray:
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
    rng: np.random.Generator, optional
        Random number generator.

    Returns
    -------
    np.ndarray, shape=(m, n)
        2D-array where each line is a sampled histogram with given fraction,
        and columns represent bins
    """
    positions = np.repeat(np.arange(array.size, dtype=int), array)
    sizes = np.array(np.round(len(positions) * frac), dtype=int)
    cumsizes = np.insert(np.cumsum(sizes), 0, 0)
    sampled_pos = np.zeros(cumsizes[-1], dtype=int)
    for i in range(len(frac)):
        sampled_pos[cumsizes[i] : cumsizes[i + 1]] = rng.choice(
            positions, size=sizes[i], replace=False
        )
    histogram = coo_matrix(
        (
            np.ones(len(sampled_pos), dtype=int),
            (np.repeat(np.arange(len(frac)), sizes), sampled_pos),
        ),
        shape=(len(frac), len(array)),
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


def apply_on_index(func, *args, length=None, neutral=0):
    """Applies a function on arrays specified as indices and values.

    Parameters
    ----------
    func : callable
        Function to compute on arrays, must take the number of values as first
        argument, then as many arrays as provided in args.
    args : tuple of tuple of arrays
        Each argument represents an array, it must be a tuple
        (indices, values) for each array. Indices must be 1D, values can be
        multi-dimensional, in which case indices will be taken along the last
        axis.
    length : int, default=None
        Length of the output array. If None, the smallest possible length is
        inferred from the indices in args.
    neutral : int, default=0
        Neutral element for the desired operation. This function requires that
        func has a neutral element.

    Returns
    -------
    ndarray
        Result array of the full length, including nans where none of the
        arrays had any values.
    """
    idxes, vals = zip(*args)
    if length is None:
        length = np.max([np.max(idx) for idx in idxes]) + 1
    n_per_pos = np.zeros((1, length))
    newvals = []
    for idx, val in args:
        val2D = val.reshape(-1, val.shape[-1])
        newval = np.full((len(val2D), length), neutral, dtype=val.dtype)
        newval[:, idx] = val2D
        n_per_pos[0, idx] += 1
        newvals.append(newval)
    res = func(n_per_pos, *newvals)
    res = np.where(n_per_pos == 0, np.nan, res)
    return res.reshape(vals[0].shape[:-1] + (-1,))


def mean_on_index(*args, length=None):
    """Computes the mean of arrays specified as indices and values.

    Parameters
    ----------
    args : tuple of tuple of arrays
        Each argument represents an array, it must be a tuple
        (indices, values) for each array. Indices must be 1D, values can be
        multi-dimensional, in which case indices will be taken along the last
        axis.
    length : int, default=None
        Length of the output array. If None, the smallest possible length is
        inferred from the indices in args.

    Returns
    -------
    ndarray
        Result array of the full length, including nans where none of the
        arrays had any values.
    """
    return apply_on_index(lambda n, *args: sum(args) / n, *args, length=length)


def geometric_mean_on_index(*args, length=None):
    """Computes the geometric mean of arrays specified as indices and values.

    Parameters
    ----------
    args : tuple of tuple of arrays
        Each argument represents an array, it must be a tuple
        (indices, values) for each array. Indices must be 1D, values can be
        multi-dimensional, in which case indices will be taken along the last
        axis.
    length : int, default=None
        Length of the output array. If None, the smallest possible length is
        inferred from the indices in args.

    Returns
    -------
    ndarray
        Result array of the full length, including nans where none of the
        arrays had any values.
    """
    return apply_on_index(
        lambda n, *args: np.product(args, axis=0) ** (1 / n),
        *args,
        length=length,
        neutral=1,
    )


def max_on_index(*args, length=None):
    """Computes the mx of arrays specified as indices and values.

    Parameters
    ----------
    args : tuple of tuple of arrays
        Each argument represents an array, it must be a tuple
        (indices, values) for each array. Indices must be 1D, values can be
        multi-dimensional, in which case indices will be taken along the last
        axis.
    length : int, default=None
        Length of the output array. If None, the smallest possible length is
        inferred from the indices in args.

    Returns
    -------
    ndarray
        Result array of the full length, including nans where none of the
        arrays had any values.
    """
    return apply_on_index(
        lambda n, *args: np.max(args, axis=0), *args, length=length, neutral=-np.inf
    )


# Random sequences generation
def kmer_counts(
    one_hots: Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]],
    k: int,
    order: str = "ACGT",
    includeN: bool = True,
    as_pandas: bool = True,
) -> Union[np.ndarray, pd.Series]:
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
            all_counts[4] = len(one_hots) * one_hots.shape[1] - all_counts[:4].sum()
        else:
            # Convert one_hot to integer tokens
            tokens = np.argmax(one_hots, axis=-1) + 4 * (np.sum(one_hots, axis=-1) != 1)
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
                tokens = np.argmax(oh, axis=-1) + 4 * (np.sum(oh, axis=-1) != 1)
                # Get kmers with sliding_window_view
                kmers = sliding_window_view(tokens, k)
                # Count kmers in the kD array
                np.add.at(all_counts, tuple(kmers[:, i] for i in range(k)), 1)
    # Format output
    if includeN:
        order += "N"
    else:
        all_counts = all_counts[tuple(slice(0, -1) for i in range(k))]
    if as_pandas:
        ser = pd.Series(
            all_counts.ravel(), index=pd.MultiIndex.from_product([list(order)] * k)
        )
        return ser.sort_index()
    else:
        return all_counts


def kmer_counts_by_seq(one_hots, k, order="ACGT", includeN=True, as_pandas=True):
    if one_hots.ndim != 3:
        raise ValueError(f"one_hots must be 3D, not {one_hots.ndim}")
    # Initialise kD array
    all_counts = np.zeros(tuple(5 for i in range(k)) + (len(one_hots),), dtype=int)
    if k == 1:
        # Count each base
        all_counts[:4] = one_hots.sum(axis=1).T
        # Count leftover as Ns
        all_counts[4] = one_hots.shape[1] - all_counts[:4].sum(axis=0)
    else:
        # Convert one_hot to integer tokens
        tokens = np.argmax(one_hots, axis=-1) + 4 * (np.sum(one_hots, axis=-1) != 1)
        for i, arr in enumerate(tokens):
            # Get kmers with sliding_window_view
            kmers = sliding_window_view(arr, k)
            # Count kmers in the kD array
            np.add.at(all_counts, tuple(kmers[:, j] for j in range(k)) + (i,), 1)
    if includeN:
        order += "N"
    else:
        all_counts = all_counts[tuple(slice(0, -1) for i in range(k)) + (slice(None),)]
    if as_pandas:
        ser = pd.DataFrame(
            all_counts.reshape(len(order) ** k, -1),
            index=pd.MultiIndex.from_product([list(order)] * k),
        )
        return ser.sort_index()
    else:
        return all_counts


def sliding_kmer_counts(
    one_hot, k, winsize, order="ACGT", includeN=True, as_pandas=True
):
    if one_hot.ndim != 2:
        raise ValueError(f"one_hot must be 2D, not {one_hot.ndim}")
    n_windows = len(one_hot) - winsize + 1
    # Initialise kD array
    all_counts = np.zeros(tuple(5 for i in range(k)) + (n_windows,), dtype=int)
    if k == 1:
        # Count each base
        all_counts[:4] = moving_sum(one_hot, winsize, axis=0).T
        # Count leftover as Ns
        all_counts[4] = winsize - all_counts[:4].sum(axis=0)
    else:
        # Convert one_hot to integer tokens
        tokens = np.argmax(one_hot, axis=-1) + 4 * (np.sum(one_hot, axis=-1) != 1)
        # Get kmers with sliding_window_view
        kmers = sliding_window_view(tokens, k)
        # Count kmers in first window in the kD array
        np.add.at(
            all_counts, tuple(kmers[: winsize + 1 - k, j] for j in range(k)) + (0,), 1
        )
        for i in range(n_windows - 1):
            # Copy count from previous window
            all_counts[..., i + 1] = all_counts[..., i]
            # Remove first kmer of previous window, add last kmer of next one
            np.add.at(
                all_counts,
                tuple(kmers[[i, winsize + 1 - k + i], j] for j in range(k)) + (i + 1,),
                [-1, 1],
            )
    if includeN:
        order += "N"
    else:
        all_counts = all_counts[tuple(slice(0, -1) for i in range(k)) + (slice(None),)]
    if as_pandas:
        ser = pd.DataFrame(
            all_counts.reshape(len(order) ** k, -1),
            index=pd.MultiIndex.from_product([list(order)] * k),
        )
        return ser.sort_index()
    else:
        return all_counts


def ref_kmer_frequencies(freq_nucs, k=2):
    ser = pd.Series(
        1, index=pd.MultiIndex.from_product([list(flatten(freq_nucs.index))] * k)
    )
    freq_nucs = freq_nucs / freq_nucs.sum(axis=0)
    for tup in ser.index:
        for nuc in tup:
            ser.loc[tup] *= freq_nucs[nuc]
    return ser


def random_shuffles(
    array: np.ndarray, n: int, rng: np.random.Generator = np.random.default_rng()
) -> np.ndarray:
    return array[rng.random((n, len(array))).argsort(axis=1)]


def shuffle_along_axis(
    arr: np.ndarray, axis: int = 0, rng: np.random.Generator = np.random.default_rng()
) -> np.ndarray:
    """Shuffles a multi-dimensional array along specified axis"""
    assert isinstance(axis, int) and axis >= -1
    if axis == -1:
        axis = arr.ndim - 1
    assert arr.ndim > axis
    return arr[
        tuple(
            np.expand_dims(
                np.arange(arr.shape[dim]),
                axis=tuple(i for i in range(axis + 1) if i != dim),
            )
            for dim in range(axis)
        )
        + (rng.random(arr.shape[: axis + 1]).argsort(axis=axis),)
    ]


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


def tokens_to_one_hot(tokens, one_hot_dim, dtype=None):
    """
    Converts an L-vector of integers in the range [0, D] to an L x D one-hot
    encoding. The value `D` must be provided as `one_hot_dim`. A token of D
    means the one-hot encoding is all 0s.
    From github.com/kundajelab/deeplift/blob/master/deeplift/dinuc_shuffle.py
    """
    identity = np.identity(one_hot_dim + 1, dtype=dtype)[:, :-1]  # Last row is all 0s
    return identity[tokens]


def dinuc_shuffle(
    seq: Union[str, np.ndarray],
    num_shufs: int = None,
    rng: np.random.Generator = np.random.default_rng(),
) -> Union[str, List[str], np.ndarray]:
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
    if isinstance(seq, str):
        arr = string_to_char_array(seq)
    elif type(seq) is np.ndarray and len(seq.shape) == 2:
        seq_len, one_hot_dim = seq.shape
        arr = one_hot_to_tokens(seq)
    else:
        raise ValueError("Expected string or one-hot encoded array")

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

    if isinstance(seq, str):
        all_results = []
    else:
        all_results = np.empty(
            (num_shufs if num_shufs else 1, seq_len, one_hot_dim), dtype=seq.dtype
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

        if isinstance(seq, str):
            all_results.append(char_array_to_string(chars[result]))
        else:
            all_results[i] = tokens_to_one_hot(chars[result], one_hot_dim)
    return all_results if num_shufs else all_results[0]


def random_seq_strict_GC(n_seqs, seq_length, gc):
    gc_count = int(round((seq_length // 2) * gc, 0))
    ref_seq = np.array(
        list(
            "A" * (seq_length % 2)
            + "AT" * (seq_length // 2 - gc_count)
            + "GC" * gc_count
        )
    )
    return random_shuffles(ref_seq, n_seqs)


def random_sequences(
    n_seqs: int,
    seq_length: int,
    freq_kmers: pd.Series,
    out: str = "seq",
    rng: np.random.Generator = np.random.default_rng(),
) -> np.ndarray:
    """Generate random DNA sequences with custom kmer distribution.

    Parameters
    ----------
    n_seqs : int
        Number of sequences to generate
    seq_length : int
        Length of the sequences to generate, must be greater than k
    freq_kmers : Series
        Series containing frequencies or occurences of each k-mer.
        Must be indexed by a k-level MultiIndex with the bases 'ACGT' on each
        level in this order.
    out : {'seq', 'idx', 'one_hot'}
        Output format, 'seq' for nucleotide characters, 'idx' for indices into
        'ACGT' or 'one_hot' for one-hot encoded bases
    rng: np.random.Generator, optional
        Random number generator.

    Returns
    -------
    ndarray, shape=(`n_seqs`, `seq_length`)
        Generated sequences as a 2D-array of characters, of indices into
        'ACGT' or 3D-array of one-hot encoded bases

    """
    assert n_seqs >= 1 and seq_length >= 0
    # Array of bases for fast indexing
    letters = np.array(list("ACGTN"))

    # Get value of k
    k = freq_kmers.index.nlevels
    if k == 1:
        seqs = rng.choice(4, size=(n_seqs, seq_length), p=freq_kmers)
    else:
        # Cumulative distribution of each base, given the previous k-1
        groups = freq_kmers.groupby(level=list(i for i in range(k - 1)))
        sum = groups.transform("sum")
        cumsum = groups.transform("cumsum")
        p_cum_kmers = cumsum / sum
        # Convert to kD-array
        arr_kmers = np.zeros(tuple([4] * k))
        for tup in it.product(range(4), repeat=k):
            arr_kmers[tup] = np.asarray(p_cum_kmers.loc[tuple(letters[i] for i in tup)])
        # Empty sequences
        seqs = np.array([4] * seq_length * n_seqs).reshape(n_seqs, seq_length)
        # Get first k-mer given k-mer distribution
        r_start = rng.choice(len(freq_kmers), n_seqs, p=freq_kmers / freq_kmers.sum())
        seqs[:, :k] = np.array(list(it.product(range(4), repeat=k)))[
            r_start, :seq_length
        ]
        # Generate random numbers for all iterations
        if seq_length > k:
            r = rng.random((n_seqs, seq_length - k))
        # Get other bases given k-mer distribution, previous (k-1)-mer and random
        # numbers
        for i in range(k, seq_length):
            seqs[:, i] = np.argmax(
                arr_kmers[
                    tuple(
                        arr.ravel()
                        for arr in np.split(seqs[:, i - k + 1 : i], k - 1, axis=1)
                    )
                ]
                >= r[:, [i - k] * 4],
                axis=1,
            )
    if out == "idx":
        return np.asarray(seqs, dtype=np.int8)
    elif out == "seq":
        return letters[seqs]
    elif out == "one_hot":
        return np.eye(4, dtype=bool)[seqs]


def random_sequences_as(
    one_hots: Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]],
    n_seqs: int,
    seq_length: int,
    k: int,
    order: str = "ACGT",
    out: str = "one_hot",
    rng: np.random.Generator = np.random.default_rng(),
) -> np.ndarray:
    """Generate random DNA sequences with kmer distribution similar to input.

    Parameters
    ----------
    one_hots : list-like
        Must be a list, dictionnary or array of one-hot encoded sequences.
    n_seqs : int
        Number of sequences to generate
    seq_length : int
        Length of the sequences to generate, must be greater than k
    k : int
        Length of k-mers to consider
    order : str, default='ACGT'
        Order of bases for one-hot encoding
    out : {'one_hot', 'seq', 'idx'}
        Output format, 'seq' for nucleotide characters, 'idx' for indices into
        'ACGTN' or 'one_hot' for one-hot encoded bases. Default "one_hot"
        makes order parameter irrelevant.
    rng: np.random.Generator, optional
        Random number generator.

    Returns
    -------
    ndarray, shape=(`n_seqs`, `seq_length`)
        Generated sequences as a 2D-array of characters, of indices into
        'ACGTN' or 3D-array of one-hot encoded bases
    """
    freq_kmers = kmer_counts(one_hots, k, order, includeN=False)
    return random_sequences(n_seqs, seq_length, freq_kmers, out, rng)


def balanced_randint(
    low: int,
    high: Union[int, None] = None,
    size: int = 1,
    dtype: type = int,
    rng: np.random.Generator = np.random.default_rng(),
) -> np.ndarray:
    """Return random integers from low to high in a balanced manner.

    The resulting array is balanced in the sense that each integer will appear the same
    number of times, give or take 1 whenever the size isn't a multiple of the number of
    integers.

    Parameters
    ----------
    low: int
        Lowest (signed) integer to be drawn from the distribution (unless high=None, in
        which case this parameter is one above the highest integer)
    high: int, optional
        If provided, one above the largest (signed) integer to be drawn from the
        distribution (see above for behavior if high=None).
    size: int, optional
        Output length. Default is 1.
    dtype: type, optional
        Desired dtype of the result. Byteorder must be native. The default value is int.
    rng: np.random.Generator, optional
        Random number generator.

    Returns
    -------
    out: np.ndarray
        Array of randomly ordered integer values
    """
    if size <= 0:
        return np.array([], dtype=dtype)
    if high is None:
        high = low
        low = 0
    q, r = divmod(size, (high - low))
    extras = rng.choice(np.arange(low, high, dtype=dtype), size=r, replace=False)
    res = np.concatenate((np.repeat(np.arange(high - low, dtype=dtype), q), extras))
    rng.shuffle(res)
    return res


def nanpearson(a, b):
    """Compute pearson correlation coefficient on arrays with nans"""
    nan_a = np.isnan(a)
    nan_b = np.isnan(b)
    if np.any(nan_a != nan_b):
        a = np.array(a, dtype=float).copy()
        b = np.array(b, dtype=float).copy()
        nan_mask = nan_a | nan_b
        a[nan_mask] = np.nan
        b[nan_mask] = np.nan
    A = a - np.nanmean(a)
    B = b - np.nanmean(b)
    std_AB = np.nansum(A * B)
    std_A = np.sqrt(np.nansum(A * A))
    std_B = np.sqrt(np.nansum(B * B))
    return std_AB / (std_A * std_B + np.finfo(float).eps)


def nan_compress(array_list):
    """Remove all indices where at least one array is nan"""
    keep = ~np.logical_or.reduce(tuple(np.isnan(a) for a in array_list))
    return [np.compress(keep, array) for array in array_list]


# Other utils
def s_plural(value: float) -> str:
    """Return s if scalar value induces plural"""
    if value > 1:
        return "s"
    else:
        return ""


def format_secs(x):
    d, r = divmod(x, 86400)
    h, r = divmod(r, 3600)
    m, s = divmod(r, 60)
    if d != 0:
        print(f"{d}d", end=" ")
    print(f"{h}h{m}m{s}s")


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
            file = Path(
                file.parent, original_file.stem + f"({file_dups})" + file.suffix
            )
            # python3.9: file.with_stem(original_file.stem + f'({file_dups})')
        print(f"{original_file} exists, changing filename to {file}")
    return file


def ram_usage() -> None:
    """Print RAM memory usage.

    References
    ----------
    https://www.geeksforgeeks.org/how-to-get-current-cpu-and-ram-usage-in-python/
    """
    # Getting all memory using os.popen()
    total_memory, used_memory, free_memory = map(
        int, os.popen("free -t -m").readlines()[-1].split()[1:]
    )
    # Memory usage
    print("RAM memory % used:", round((used_memory / total_memory) * 100, 2))


def roman_to_int(str):
    sym_values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    res = 0
    i = 0
    while i < len(str):
        # Get value of current symbol
        s1 = sym_values[str[i]]
        if i + 1 < len(str):
            # Get value of next symbol
            s2 = sym_values[str[i + 1]]
            if s1 >= s2:
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
    num = [1, 4, 5, 9, 10, 40, 50, 90, 100, 400, 500, 900, 1000]
    sym = ["I", "IV", "V", "IX", "X", "XL", "L", "XC", "C", "CD", "D", "CM", "M"]
    i = 12
    res = ""
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
