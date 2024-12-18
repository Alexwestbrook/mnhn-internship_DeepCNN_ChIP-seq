import argparse
import datetime
import gc
import json
import math
import socket
from pathlib import Path
from typing import Callable, Iterable, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from Modules import utils
from Modules.tf_utils import correlate, mae_cor, np_mae_cor

YEAST_GC = 0.3834


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
        "-o", "--output_dir", type=str, required=True, help="output directory"
    )
    parser.add_argument(
        "-m",
        "--model_file",
        type=str,
        default="/home/alex/shared_folder/SCerevisiae/Trainedmodels/"
        "model_myco_pol_17/model",
        help="trained model file as hdf5 or in tf2 format (default: %(default)s)",
    )
    parser.add_argument(
        "-ord",
        "--one_hot_order",
        type=str,
        default="ACGT",
        help="order of the one-hot encoding for the model (default: %(default)s)",
    )
    parser.add_argument(
        "-kfile",
        "--kmer_file",
        type=str,
        help="file with kmer distribution to use for initializing sequences. "
        "The first named column is considered for the frequency values. "
        "There should be k columns before that, forming the kmers as a multi-index",
    )
    parser.add_argument(
        "-n",
        "--n_seqs",
        type=int,
        default=1,
        help="number of sequences to generate simultaneously (default: %(default)s)",
    )
    parser.add_argument(
        "-l",
        "--length",
        type=int,
        default=2175,
        help="length of the sequences to generate (default: %(default)s)",
    )
    parser.add_argument(
        "-start",
        "--start_seqs",
        type=str,
        help="numpy binary file containing starting sequences as indexes with "
        "shape (n_seqs, length). If set overrides n_seqs and length. (default: %(default)s)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="number of steps to perform (default: %(default)s)",
    )
    parser.add_argument(
        "-p",
        "--patience",
        type=int,
        default=10,
        help="number of steps without improvement before stopping iteration (default: %(default)s)",
    )
    parser.add_argument(
        "-dist",
        "--distribute",
        action="store_true",
        help="indicates to use MirrorStrategy for prediction (default: %(default)s)",
    )
    parser.add_argument(
        "-s",
        "--stride",
        type=int,
        default=1,
        help="specifies a stride in predictions to go faster (default: %(default)s)",
    )
    parser.add_argument(
        "-mid",
        "--middle_pred",
        action="store_true",
        help="specifies to predict only on middle window (default: %(default)s)",
    )
    parser.add_argument(
        "--flanks",
        type=str,
        help="file with flanking sequences to use for prediction, can also be "
        "'random' to get random flanks with specified kmer distribution "
        "or 'self' to use itself as flank as in tandem repeats (default: %(default)s)",
    )
    parser.add_argument(
        "-targ",
        "--target",
        type=float,
        nargs="+",
        default=[0],
        help="target profile for the designed sequences. If a single value, "
        "consider a flat target for the entire sequence, otherwise must "
        "be of same length as the sequences. (default: %(default)s)",
    )
    parser.add_argument(
        "-targ_rev",
        "--target_rev",
        type=float,
        nargs="+",
        help="target profile for the designed sequences on the reverse "
        "strand, if unspecified, consider the same target for forward "
        "and reverse. If a single value, consider a flat target for the "
        "entire sequence, otherwise must be of same length as the "
        "sequences. (default: %(default)s)",
    )
    parser.add_argument(
        "-amp",
        "--amplitude",
        type=float,
        default=1,
        help="amplitude for target profile (default: %(default)s)",
    )
    parser.add_argument(
        "-ilen",
        "--insertlen",
        type=int,
        help="length of insert in target (default: %(default)s)",
    )
    parser.add_argument(
        "-ishape",
        "--insertshape",
        type=str,
        default="linear",
        help="shape of insert. Must be either 'block', 'deplete', "
        "'linear', 'sigmoid' or 'gaussian' (default: %(default)s)",
    )
    parser.add_argument(
        "-istart",
        "--insertstart",
        type=int,
        help="Index of the position where the insert should start (default: %(default)s)",
    )
    parser.add_argument(
        "-bg",
        "--background",
        type=str,
        nargs=2,
        default=["low", "low"],
        help="background signal to put at the left and right of the insert (default: %(default)s)",
    )
    parser.add_argument(
        "-stdf",
        "--std_factor",
        type=float,
        default=1 / 4,
        help="standard deviation for gaussian peak as fraction of length (default: %(default)s)",
    )
    parser.add_argument(
        "-sigs",
        "--sig_spread",
        type=float,
        default=6,
        help="absolute value to compute the sigmoid up to on each side (default: %(default)s)",
    )
    parser.add_argument(
        "-per",
        "--period",
        type=int,
        help="period of the target, makes it multipeak (default: %(default)s)",
    )
    parser.add_argument(
        "-plen",
        "--periodlen",
        type=int,
        help="length of periodic peak in target (default: %(default)s)",
    )
    parser.add_argument(
        "-pshape",
        "--periodshape",
        type=str,
        default="gaussian",
        help="shape of periodic peak. Must be either 'block', 'deplete', "
        "'linear', 'sigmoid' or 'gaussian' (default: %(default)s)",
    )
    parser.add_argument(
        "--target_file",
        type=str,
        help="numpy binary file containing the target profile for the "
        "designed sequences. If set, overrides target, target_rev and "
        "also length, unless start_seqs is set, in which case it must be "
        "of same length as start_seqs. Can also be an npz archive with 2 "
        "different targets for each strand, with keys named 'forward' "
        "and 'reverse'. (default: %(default)s)",
    )
    parser.add_argument(
        "-targ_gc",
        "--target_gc",
        type=float,
        default=YEAST_GC,
        help="target GC content for the designed sequences (default: %(default)s)",
    )
    parser.add_argument(
        "-gctol",
        type=float,
        default=0.0001,
        help="precision of GC_content (default: %(default)s)",
    )
    parser.add_argument(
        "-gclen",
        "--gc_constrlen",
        type=int,
        default=1000,
        help="length over which to apply the GC constraint to (default: %(default)s)",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="rmse",
        help="loss to use for comparing prediction and target (default: %(default)s)",
    )
    parser.add_argument(
        "-w",
        "--weights",
        type=float,
        nargs=4,
        default=[1, 1, 1, 1],
        help="weights for the energy terms in order E_gc, E_for, E_rev, E_mut (default: %(default)s)",
    )
    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=0.1,
        help="temperature for kMC (default: %(default)s)",
    )
    parser.add_argument(
        "--mutfree_pos_file",
        type=str,
        help="Numpy binary file with positions where mutations can be "
        "performed. (default: %(default)s)",
    )
    parser.add_argument(
        "--nmut_step",
        type=int,
        default=4000,
        help="Number of mutation positions to scan per step. Set to 0 to scan all mutations at each step (default: %(default)s)",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=1024,
        help="number of windows to predict on at once (default: %(default)s)",
    )
    parser.add_argument(
        "-mbs",
        "--mut_batch_size",
        type=int,
        default=48000,
        help="number of mutated sequences to fit into memory. It is scaled down by `length`/1000, "
        "so to keep the same value for different lengths of sequences."
        "48k takes around 6G of RAM (default: %(default)s)",
    )
    parser.add_argument(
        "-c",
        "--chunk_size",
        type=int,
        default=128000,
        help="number of sequence slides that to fit into memory (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="seed to use for random generations (default: %(default)s)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="whether to print information messages (default: %(default)s)",
    )
    args = parser.parse_args()
    # Basic checks
    assert len(args.one_hot_order) == 4 and set(args.one_hot_order) == set("ACGT")
    for item in [
        args.n_seqs,
        args.length,
        args.steps,
        args.stride,
        args.gc_constrlen,
        args.batch_size,
        args.mut_batch_size,
        args.chunk_size,
        args.period,
        args.periodlen,
        args.insertstart,
    ]:
        assert item is None or item >= 1
    assert args.temperature > 0
    assert args.nmut_step >= 0
    assert args.target_gc < 1 and args.target_gc > 0
    # Check starting sequences
    if args.start_seqs is not None:
        seqs = np.load(args.start_seqs)
        assert seqs.ndim == 2
        args.n_seqs, args.length = seqs.shape
    # Extract target profile
    if args.target_file is not None:
        with np.load(args.target_file) as f:
            if isinstance(f, np.ndarray):
                args.target = f
                args.target_rev = f
            else:
                args.target = f["forward"]
                args.target_rev = f["reverse"]
        if args.start_seqs is None:
            args.length = len(args.target)
    elif args.insertlen is not None:
        args.background = tuple(args.background)
        args.target = make_target(
            args.length,
            args.insertlen,
            args.amplitude,
            ishape=args.insertshape,
            insertstart=args.insertstart,
            background=args.background,
            period=args.period,
            pinsertlen=args.periodlen,
            pshape=args.periodshape,
            std_factor=args.std_factor,
            sig_spread=args.sig_spread,
        )
        args.target_rev = args.target
    else:
        # Forward target
        if len(args.target) == 1:
            args.target = np.full(args.length, args.target[0], dtype=float)
        else:
            args.target = np.array(args.target, dtype=float)
        # Reverse target
        if args.target_rev is None:
            args.target_rev = args.target
        elif len(args.target_rev) == 1:
            args.target_rev = np.full(args.length, args.target_rev[0], dtype=float)
        else:
            args.target_rev = np.array(args.target_rev, dtype=float)
    assert len(args.target) == len(args.target_rev) and len(args.target) == args.length
    return args


def maxstep_linspace(start: int, stop: int, max_step: int) -> np.ndarray:
    """Return pseudo-evenly spaced integers over a specified interval.

    Start and stop are included. Spacings can differ by at most 1 and are smaller than max_step.
    Spacings are the largest possible value satisfying these conditions.
    This is similar to np.linspace only on integers, and by specifying the maximum step size
    instead of the number of values.

    Parameters
    ----------
    start, stop: int
        Start and stop values of the interval, both included
    max_step: int
        Maximum spacing between two consecutive integers

    Returns
    -------
    ndarray
        Array of pseudo-evenly spaced integers

    Examples
    --------
    >>> maxstep_linspace(0, 8, 2)  # all spacings equal
    array([0, 2, 4, 6, 8])
    >>> maxstep_linspace(0, -8, 3)  # uneven adjusted spacings
    array([0, -3, -5, -8])
    >>> maxstep_linspace(-4.0, 4.0, 7.0)  # spacings much smaller than max_step
    array([-4, 0, 4])
    >>> maxstep_linspace(0, 0, 2)  # start same as stop
    array([0])
    """
    for arg in [start, stop, max_step]:
        if not isinstance(arg, (int, float)):
            raise ValueError(f"Arguments must be integers, not {type(arg).__name__}")
        elif not int(arg) == arg:  # accept float typed integers
            raise ValueError(f"Float arguments must be integer valued, not {arg}")
    if max_step <= 0:
        raise ValueError(f"max_step must be positive but has value {max_step}")
    if stop == start:
        return np.array([start])
    n_steps = math.ceil(abs(stop - start) / max_step)
    return np.linspace(start, stop, n_steps + 1).round(0).astype(int)


def slicer_on_axis(
    arr: np.ndarray,
    slc: Union[slice, Iterable[slice]],
    axis: Union[None, int, Iterable[int]] = None,
) -> Tuple[slice]:
    """Build slice of array along specified axis.

    This function can be used to build slices from unknown axis parameter.

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
    >>> arr[slicer_on_axis(arr, slice(1, 3), axis=-1)]  # simple slice on last axis
    array([[[ 1,  2],
            [ 5,  6],
            [ 9, 10]],

           [[13, 14],
            [17, 18],
            [21, 22]]])
    >>> for axis, res in enumerate([arr[1:], arr[:, 1:], arr[:, :, 1:]]):  # unknown axis parameter
    >>>     print(np.all(arr[slicer_on_axis(arr, slice(1, None), axis=axis)] == res))
    True
    True
    True
    >>> arr[slicer_on_axis(arr, slice(None, -1))]  # no axis parameter
    array([[[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]]])
    >>> arr[slicer_on_axis(arr, [slice(None, -1), slice(1, 3)], axis=[0, 2])]  # multiple slices and axis
    array([[[ 1,  2],
            [ 5,  6],
            [ 9, 10]]])
    >>> arr[slicer_on_axis(arr, [slice(None, -1), slice(1, 3)])]  # multiple slices without axis parameter
    array([[[ 4,  5,  6,  7],
            [ 8,  9, 10, 11]]])
    >>> arr[slicer_on_axis(arr, slice(1, None), axis=[1, 2])]  # single slice on multiple axis
    array([[[ 5,  6,  7],
            [ 9, 10, 11]],

           [[17, 18, 19],
            [21, 22, 23]]])
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
            raise ValueError("if slc is an iterable, axis must be an iterable or None")
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
        GC_mask = seqs[..., [order.find("C"), order.find("G")]].sum(axis=-1)
        if n > seqs.shape[-1]:
            n = seqs.shape[-1]
        return moving_sum(GC_mask, n=n, axis=-1) / moving_sum(valid_mask, n=n, axis=-1)
    else:
        raise ValueError(f"form must be 'token' or 'one_hot', not {form}")


def GC_energy(
    seqs: np.ndarray,
    n: int,
    target_gc: float,
    tol: float = 0,
    order: str = "ACGT",
    form: str = "token",
) -> Union[int, np.ndarray]:
    """Compute GC energy of all sequences.

    GC energy of a sequence is defined as the rmse of all values of sliding GC along the sequence.

    Parameters
    ----------
    seqs: ndarray
        Input sequences. Dimensions must start with number of sequences if applicable,
        then length of sequences, and finally one-hot dimension if applicable.
    n: int
        Length of window to compute GC content on, must be greater than 0.
    target_gc: float
        Target gc content.
    tol: float, optional
        Tolerance for target_gc precision, any value closer than tol to the target_gc will be considered equal.
    order: str, optional
        Order of encoding, must contain each letter of ACGT exactly once.
        If `form` is 'token', then value i corresponds to base at index i in order.
        If `form` is 'one_hot', then vector of zeros with a 1 at position i corresponds to base at index i in order.
    form: {'token', 'one_hot'}, optional
        Form of input array. 'token' for indexes of bases and 'one_hot' for one-hot encoded sequences, with an extra dimension.

    Returns
    -------
    int or ndarray
        Array of energies for each sequence. If a single sequence is given without a number of sequences dimension, a scalar is returned.
    """
    sliding_gc = sliding_GC(seqs, n, axis=-1, order=order, form=form)
    if tol != 0:
        sliding_gc[np.abs(sliding_gc - target_gc) < tol] = target_gc
    return rmse(sliding_gc, target_gc)


def get_pred_start_and_stop(
    winsize: int, head_interval: Union[None, int] = None, middle: bool = False
) -> Tuple[int, int]:
    """Compute start and stop indices of predictions

    Parameters
    ----------
    winsize : int
        Input length of the model
    head_interval : int, optional
        Spacing between outputs of the model, for a model with multiple
        outputs starting at first window position and with regular spacing.
        If None, model must have a single output in the middle of the window.
    middle : bool, optional
        Whether to use only the middle half of output heads for deriving
        predictions. This results in no predictions on sequence edges.

    Returns
    -------
    pred_start : int
        Distance between first prediction and sequence start
    pred_stop : int
        Distance between last prediction and sequence end
    """
    if head_interval is not None:
        pred_start = 0
        pred_stop = head_interval - 1
        # Increase distances in case of middle predictions
        if middle:
            pred_start += winsize // 4
            pred_stop += winsize // 4
    else:
        pred_start = winsize // 2
        pred_stop = winsize // 2
    return pred_start, pred_stop


def pad_sequences(
    seqs: np.ndarray,
    pred_start: int,
    pred_stop: int,
    flanks: Union[None, str, Tuple[np.ndarray]],
    reverse: bool = False,
) -> Tuple[np.ndarray, int, int]:
    """Pad sequences of the appropriate length using flanking sequences.

    Parameters
    ----------
    seqs : ndarray
        Input array, dimension to be padded must on the last axis
    pred_start : int
        Distance between first prediction and sequence start
    pred_stop : int
        Distance between last prediction and sequence end
    flanks : str or tuple of 2 ndarrays
        Tuple of 2 arrays flank_left and flank_right to be added at the start
        and end respectively of each sequence to get prediction on the entire
        sequence.
    reverse : bool, optional
        If True, pad for prediction on reverse strand. Default is False for prediction on
        forward strand.

    Returns
    -------
    seqs_pad: np.ndarray
        Padded sequences, all but last dimension are the same as seqs.
    leftpad, rightpad : int
        Length of padding sequences added before and after the sequences respectively
    """
    if flanks is None:
        return seqs, 0, 0
    if reverse:
        leftpad, rightpad = pred_stop, pred_start
    else:
        leftpad, rightpad = pred_start, pred_stop
    if flanks == "self":
        flank_left = np.tile(seqs, leftpad // seqs.shape[-1] + 1)
        flank_left = flank_left[..., flank_left.shape[-1] - leftpad :]
        flank_right = np.tile(seqs, rightpad // seqs.shape[-1] + 1)
        flank_right = flank_right[..., :rightpad]
        seqs_pad = np.concatenate([flank_left, seqs, flank_right], axis=-1)
    else:
        flank_left, flank_right = flanks
        # Truncate flanks if they are longer
        if leftpad < len(flank_left):
            flank_left = flank_left[len(flank_left) - leftpad :]
        flank_right = flank_right[:rightpad]
        # Adjust actual pad values if flanks are shorter
        leftpad = len(flank_left)
        rightpad = len(flank_right)
        seqs_pad = np.concatenate(
            [
                np.tile(flank_left, seqs.shape[:-1] + (1,)),
                seqs,
                np.tile(flank_right, seqs.shape[:-1] + (1,)),
            ],
            axis=-1,
        )
    return seqs_pad, leftpad, rightpad


def all_mutations(seqs, positions, return_occs=False, bases_occs=None):
    """Perform all possible mutations at given positions.

    Parameters
    ----------
    seqs : ndarray, shape=(n, l)
        Array of indexes into 'ACGT', n is the number of sequences and l their
        length.
    positions : ndarray
        Indices of positions to perform mutations on.
    return_occs : bool, optional
        If True, return occurences of each mutation. Requires bases_occ to be set.
    bases_occs : ndarray, shape=(n, l, 4), optional
        Array of occurence of each base at each position during the
        optimization process of each sequence

    Returns
    -------
    seqs_mut : ndarray, shape=(n, 3*l, l)
        Array of all possible mutations of each sequence in seqs
    mut_occs : ndarray, shape=(n, 3*l)
        Array of previous occurences of each mutation

    Notes
    -----
    Mutations are performed in a circular rotation (0 -> 1 -> 2 -> 3 -> 0) by
    adding 1, 2 or 3 to the index and then taking the modulo 4.
    """
    n_seqs, length = seqs.shape
    # Create mutations array
    # Array of increments of 1, 2 or 3 at each position for a single sequence
    single_seq_increments = np.zeros((len(positions), length), dtype=seqs.dtype)
    single_seq_increments[np.arange(len(positions)), positions] = 1
    single_seq_increments = (
        np.expand_dims(single_seq_increments, 1) * np.arange(1, 4).reshape(1, -1, 1)
    ).reshape(-1, length)  # shape (3*len(batch), length)
    # Add increments to each sequence by broadcasting and take modulo 4
    seqs_mut = (
        np.expand_dims(single_seq_increments, 0) + np.expand_dims(seqs, 1)
    ) % 4  # shape (n_seqs, 3*len(batch), length)
    if return_occs:
        # Associate each mutation with its occurence
        # Array of resulting bases for each mutation
        mut_idx = (
            np.expand_dims(seqs[:, positions], axis=-1)
            + np.arange(1, 4).reshape(1, 1, -1)
        ) % 4  # shape (n_seqs, len(batch), 3)
        # Select values associated with each base in bases_occs
        mut_occs = np.take_along_axis(
            bases_occs[:, positions], mut_idx, axis=-1
        ).reshape(n_seqs, -1)  # shape (n_seqs, 3*len(batch))
        return seqs_mut, mut_occs
    else:
        return seqs_mut


def np_idx_to_one_hot(
    idx: np.ndarray,
    order: str = "ACGT",
    extradims: Union[None, int] = None,
    dtype: type = bool,
) -> np.ndarray:
    """Convert array of indexes into one-hot in np format.

    Parameters
    ----------
    idx : ndarray
        Array of indexes into 'ACGT'
    order : str, optional
        String representation of the order in which to encode bases. Default
        value of 'ACGT' means that A has the representation with 1 in first
        position, C with 1 in second position, etc...
    extradims : int or list of int, optional
        Extra dimensions to give to the one_hot, which by default is of shape
        idx.shape + (4,). If extradims is an array there will be an error.
    dtype : bool, optional
        Type to use for one-hot encoding

    Returns
    -------
    ndarray
        Array with same shape as idx, in one-hot format.
    """
    if not (len(order) == 4 and set(order) == set("ACGT")):
        raise ValueError("order must be a permutation of ACGT, not {order}")
    converter = np.zeros((4, 4), dtype=dtype)
    for i, c in enumerate("ACGT"):
        converter[i, order.find(c)] = 1
    one_hot = converter[idx]
    if extradims is not None:
        one_hot = np.expand_dims(one_hot, axis=extradims)
    return one_hot


def get_profile_chunk(
    seqs: np.ndarray,
    model: tf.keras.Model,
    winsize: int,
    head_interval: Union[None, int] = None,
    middle: bool = False,
    reverse: bool = False,
    flanks: Union[None, str, Tuple[np.ndarray]] = None,
    stride: int = 1,
    offset: Union[None, int] = None,
    batch_size: int = 1024,
    chunk_size: int = 128000,
    one_hot_converter: Callable[[np.ndarray], np.ndarray] = np_idx_to_one_hot,
    return_index: bool = False,
    rng: np.random.Generator = np.random.default_rng(),
    verbose: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray]]:
    """Predict profile.

    Parameters
    ----------
    seqs : ndarray
        Array of indexes into 'ACGT', sequences to predict on are read on the
        last axis
    model
        Tensorflow model instance supporting the predict method
    winsize : int
        Input length of the model
    head_interval : int, optional
        Spacing between outputs of the model, for a model with multiple
        outputs starting at first window position and with regular spacing.
        If None, model must have a single output in the middle of the window.
    middle : bool, optional
        Whether to use only the middle half of output heads for deriving
        predictions. This results in no predictions on sequence edges.
    reverse : bool, optional
        If True predict on reverse strand. Default is False for predicting on
        forward strand.
    flanks : tuple of array or 'self', optional
        Tuple of 2 arrays flank_left and flank_right to be added at the start
        and end respectively of each sequence to get prediction on the entire
        sequence.
    stride : int, optional
        Stride to use for prediction. Using a value other than 1 will result
        in bases being skipped and make prediction faster
    offset : int, optional
        Offset for start of prediciton, will be forced to be positive and
        smaller than stride by taking the modulo. Default value of None will
        result in a random offset being chosen.
    batch_size : int, optional
        Batch_size for model.predict().
    chunk_size : int, optional
        Number of full sequences to feed to model.predict() at once. This is
        different from the model batch_size. The constraint here is the total
        number of one-hot encoded sequence slides that can fit into memory.
    one_hot_converter : function, optional
        Function taking as input an array of indexes into 'ACGT' with shape
        (n_windows, window_length) and converts it to the required model input
        format.
    return_index : bool, optional
        If True, return indices corresponding to the predictions.
    rng : np.random.Generator, optional
        Random number generator.
    verbose : bool, optional
        If True, print information messages.

    Returns
    -------
    preds : ndarray
        Array of predictions with same shape as seqs, except on the last
        dimension, containing predictions for that sequence.
    indices : ndarray
        Array of indices of preds into seqs to be taken with
        np.take_along_axis, only provided if return_index is True.
    """
    # Copy sequence array and make 2D with sequence along the 2nd axis
    seqs2D = seqs.reshape(-1, seqs.shape[-1]).copy()
    # Determine offset for prediction
    if offset is None:
        # Randomize offset
        offset = rng.integers(0, stride)
    else:
        offset %= stride
    if verbose:
        print(
            f"Predicting {'reverse' if reverse else 'forward'} with stride {stride} and offset {offset}"
        )
    # pred_start: distance between first prediction and sequence start
    # pred_stop: distance between last prediction and sequence end
    pred_start, pred_stop = get_pred_start_and_stop(winsize, head_interval, middle)
    # Add flanking sequences to make prediction along the entire sequence, and
    # update distances
    seqs2D, leftpad, _ = pad_sequences(
        seqs2D, pred_start, pred_stop, flanks, reverse=reverse
    )
    # Determine indices of predictions along the sequence axis
    if return_index:
        indices = np.arange(pred_start + offset, seqs2D.shape[-1] - pred_stop, stride)
        if reverse:
            indices = np.flip(seqs2D.shape[-1] - indices - 1)
        indices -= leftpad
    # Maybe reverse complement the sequences
    if reverse:
        seqs2D[seqs2D == 0] = -1
        seqs2D[seqs2D == 3] = 0
        seqs2D[seqs2D == -1] = 3
        seqs2D[seqs2D == 1] = -1
        seqs2D[seqs2D == 2] = 1
        seqs2D[seqs2D == -1] = 2
        seqs2D = np.flip(seqs2D, axis=-1)
    # Get windows to predict on, and split them into chunks
    if head_interval is not None:
        # Determine number of windows in a slide, number of heads, of kept
        # heads and length to jump once a slide is done
        slide_length = head_interval // stride
        n_heads = winsize // head_interval
        if middle:
            jump_stride = winsize // 2
            n_kept_heads = n_heads // 2
        else:
            jump_stride = winsize
            n_kept_heads = n_heads
        windows = utils.strided_sliding_window_view(
            seqs2D[:, offset:],
            (len(seqs2D), winsize),
            stride=(jump_stride, stride),
            sliding_len=head_interval,
            axis=-1,
        ).reshape(-1, winsize)
        chunks = np.split(windows, np.arange(chunk_size, len(windows), chunk_size))

        # Deal with the last slide
        # Last position where a window can be taken
        last_valid_pos = seqs2D.shape[-1] - winsize
        # Last position where a full slide can be taken
        last_slide_start = last_valid_pos - head_interval + stride
        # Adjust last_slide_start to offset (for continuity)
        last_slide_start -= (last_slide_start - offset) % stride
        # Distance between last slide in `slides` and last_slide_start
        last_jump = (last_slide_start - offset) % jump_stride
        # Add extra windows if there is still a window to predict on
        if last_jump != 0:
            extra_windows = utils.strided_sliding_window_view(
                seqs2D[:, last_slide_start:],
                (len(seqs2D), winsize),
                stride=(jump_stride, stride),
                sliding_len=head_interval,
                axis=-1,
            ).reshape(-1, winsize)
            chunks += np.split(
                extra_windows, np.arange(chunk_size, len(extra_windows), chunk_size)
            )
    else:
        windows = (
            utils.strided_window_view(
                seqs2D[:, offset:], (len(seqs2D), winsize), stride=(1, stride)
            )
            .transpose([0, 2, 1, 3])
            .reshape(-1, winsize)
        )
        chunks = np.split(windows, np.arange(chunk_size, len(windows), chunk_size))
    # Make predictions
    preds = []
    for chunk in chunks:
        # Convert to one-hot and predict
        pred = model.predict(one_hot_converter(chunk), batch_size=batch_size).squeeze(
            axis=-1
        )
        # Collect garbage to prevent memory leak from model.predict()
        gc.collect()
        preds.append(pred)
    preds = np.concatenate(preds)
    # Reformat predictions
    if head_interval is not None:
        # Maybe extract middle prediction
        if middle:
            preds = preds[:, n_heads // 4 : 3 * n_heads // 4]
        # Transpose slide_length and n_kept_heads to get proper sequence order
        preds = np.transpose(preds.reshape(-1, slide_length, n_kept_heads), [0, 2, 1])
        # Seperate last slide prediction to truncate its beginning then put it
        # back
        if last_jump != 0:
            preds = preds.reshape(-1, len(seqs2D), slide_length * n_kept_heads)
            first_part = preds[:-1, :, :].reshape(len(seqs2D), -1)
            last_part = preds[-1, :, -(last_jump // stride) :]
            preds = np.concatenate([first_part, last_part], axis=-1)
    # Reshape as original sequence
    preds = preds.reshape(seqs.shape[:-1] + (-1,))
    # Maybe reverse predictions
    if reverse:
        preds = np.flip(preds, axis=-1)
    if return_index:
        return preds, indices
    else:
        return preds


def get_profile_mutations(
    seqs_mut: np.ndarray,
    mutpos: np.ndarray,
    cur_preds: np.ndarray,
    predicter: Callable[
        [np.ndarray, bool, Union[None, int], Union[None, str, Tuple[np.ndarray]]],
        np.ndarray,
    ],
    winsize: int,
    head_interval: Union[None, int] = None,
    middle: bool = False,
    reverse: bool = False,
    flanks: Union[None, str, Tuple[np.ndarray]] = None,
    stride: int = 1,
    offset: Union[None, int] = None,
    return_index: bool = False,
    cur_indices: Union[None, np.ndarray] = None,
) -> Union[np.ndarray, Tuple[np.ndarray]]:
    """Predict profile by only updating.
    Perform all mutations, in its prediction window and associate its mutation energy.

    The prediction window is the part of the sequence for which the model prediction
    will see the effect of the mutation, it is of length `winsize`*2-1,
    with the mutation generally in its center. Except for positions too far left or right
    of the sequence, for which the prediction window is incomplete, so we use the first
    or the last one, and the mutation isn't in the middle.

    Parameters
    ----------
    seqs_mut : ndarray, shape (n_seqs, n_mutations, length)
        Mutated sequences. Array of indexes into 'ACGT', sequences to predict on are read on the
        last axis. Can be the output of `all_mutations`.
    mutpos : ndarray, shape (n_seqs, n_mutations)
        1D-array of positions in length where mutations have been taken.
        It is a repeated/broadcasted version of mutfree_pos in `all_mutations`.
    cur_preds : ndarray, shape (n_seqs, pred_len)
        Array of current predictions on seqs
    predicter : callable
        Function used to predict profile of each sequence. Must admit seqs, reverse, offset and flanks arguments.
    winsize : int
        Input length of the model
    head_interval : int, optional
        Spacing between outputs of the model, for a model with multiple
        outputs starting at first window position and with regular spacing.
        If None, model must have a single output in the middle of the window.
    middle : bool, optional
        Whether to use only the middle half of output heads for deriving
        predictions. This results in no predictions on sequence edges.
    reverse : bool, optional
        If True predict on reverse strand. Default is False for predicting on
        forward strand.
    flanks : tuple of array or 'self', optional
        Tuple of 2 arrays flank_left and flank_right to be added at the start
        and end respectively of each sequence to get prediction on the entire
        sequence.
    stride : int, optional
        Stride to use for prediction. Using a value other than 1 will result
        in bases being skipped and make prediction faster
    offset : int, optional
        Offset for start of prediciton, will be forced to be positive and
        smaller than stride by taking the modulo. Default value of None will
        result in a random offset being chosen.
        format.
    return_index : bool, optional
        If True, return indices corresponding to the predictions, requires cur_indices to be set.
    cur_indices : ndarray, shape (pred_len,) or (n_seqs, pred_len), optional
        Array of indices for current predictions. Required if return_index is set to True.

    Returns
    -------
    preds : ndarray
        Array of predictions with same shape as seqs, except on the last
        dimension, containing predictions for that sequence.
    indices : ndarray
        Array of indices of preds into seqs to be taken with
        np.take_along_axis, only provided if return_index is True.
    """
    # pred_start: distance between first prediction and sequence start
    # pred_stop: distance between last prediction and sequence end
    pred_start, pred_stop = get_pred_start_and_stop(winsize, head_interval, middle)
    # Add flanking sequences to make prediction along the entire sequence, and
    # update distances
    seqs_mut, leftpad, _ = pad_sequences(
        seqs_mut, pred_start, pred_stop, flanks, reverse=reverse
    )
    left_missing = leftpad - (pred_stop if reverse else pred_start)
    # Extract changed windows from mutated sequences (i.e. where the model will see the effect of the mutation)
    changesize = 2 * winsize - 1  # changed window length
    # If sequence too small, revert to full sequence prediction
    if changesize > seqs_mut.shape[-1]:
        changesize = seqs_mut.shape[-1]
    # Starting index of each changed window
    start_idx = np.clip(
        mutpos - winsize + 1 + leftpad,
        left_missing,
        seqs_mut.shape[-1] - changesize + left_missing,
    )  # shape (n_seqs, n_mutations)
    # Complete indices of each window
    window_indices = np.expand_dims(start_idx, axis=-1) + np.arange(changesize).reshape(
        1, -1
    )  # shape (n_seqs, n_mutations, changesize)
    seqs_mut = np.take_along_axis(
        seqs_mut, window_indices, axis=2
    )  # shape (n_seqs, n_mutations, changesize)

    # Make predictions on sequences
    preds, indices = predicter(seqs_mut, reverse=reverse, offset=offset, flanks=None)

    # Adjust indices according to starting index of windows
    indices = np.expand_dims(start_idx - leftpad, axis=-1) + indices.reshape(1, -1)
    # Rebuild full predictions by updating current predictions with mutated predictions
    full_preds = np.repeat(np.expand_dims(cur_preds, axis=1), indices.shape[1], axis=1)
    modindices = indices // stride
    np.put_along_axis(full_preds, modindices, preds, axis=-1)
    if return_index:
        if cur_indices.ndim == 1:
            # Expand dimensions to match that of cur_preds
            cur_indices = np.tile(cur_indices, (len(cur_preds), 1))
            assert cur_indices.shape == cur_preds.shape
        # Rebuild full indices by updating current indices with new ones
        full_indices = np.repeat(
            np.expand_dims(cur_indices, axis=1), indices.shape[1], axis=1
        )
        np.put_along_axis(full_indices, modindices, indices, axis=-1)
        return full_preds, full_indices
    else:
        return full_preds


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> Union[int, np.ndarray]:
    """Compute RMSE between two arrays.

    RMSE(X, Y) is defined as sqrt(mean((X-Y)**2))

    Parameters
    ----------
    y_true, y_pred : ndarray
        Arrays of values. If multidimensional, computation is performed along
        the last axis.

    Returns
    -------
    int or ndarray
        Array of same shape as y_true and y_pred but with last axis removed. If arrays are 1D, a scalar is returned.
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2, axis=-1))


def make_shape(
    length: int,
    amplitude: float,
    shape: str = "linear",
    background: Tuple[str, str] = ("low", "low"),
    std_factor: float = 1 / 4,
    sig_spread: float = 6,
) -> np.ndarray:
    """TODO"""
    base_length = length
    if background[0] == background[1]:
        length = (length + 1) // 2
    if shape == "block":
        arr = np.full(length, amplitude)
    elif shape == "deplete":
        arr = np.zeros(length)
    elif shape == "linear":
        arr = np.linspace(0, amplitude, length)
    elif shape == "gaussian":
        if background[0] != background[1]:
            base_length = length * 2 + 1
        arr = np.exp(
            -((np.arange(length) - (base_length - 1) / 2) ** 2)
            / (2 * (base_length * std_factor) ** 2)
        )
        arr -= np.min(arr)
        arr *= amplitude / np.max(arr)
    elif shape == "sigmoid":
        x = np.linspace(-sig_spread, sig_spread, length)
        arr = 1 / (1 + np.exp(-x))
        arr -= np.min(arr)
        arr *= amplitude / np.max(arr)
    if background[0] == "high":
        arr = amplitude - arr
    if background[0] == background[1]:
        if base_length % 2 != 0:
            return np.concatenate([arr, arr[-2::-1]])
        else:
            return np.concatenate([arr, arr[::-1]])
    else:
        return arr


def make_target(
    length: int,
    insertlen: int,
    amplitude: float,
    ishape: str = "linear",
    insertstart: Union[None, int] = None,
    period: Union[None, int] = None,
    pinsertlen: int = 147,
    pshape: str = "gaussian",
    background: Tuple[str, str] = ("low", "low"),
    **kwargs,
):
    """TODO"""
    if insertstart is None:
        insertstart = (length - insertlen + 1) // 2
    rightlen = length - insertlen - insertstart
    if period is not None:
        if insertlen == 0:
            insertlen = period - pinsertlen
            insertstart = (length - insertlen + 1) // 2
            rightlen = length - insertlen - insertstart
        insert = make_shape(insertlen, amplitude, shape="deplete")
        pinsert = make_shape(pinsertlen, amplitude, shape=pshape, **kwargs)
        tile = np.concatenate([pinsert, np.zeros(period - len(pinsert))])
        backgr = np.tile(tile, length // (period) + 1)
        leftside = backgr[insertstart - 1 :: -1]
        rightside = backgr[:rightlen]
    else:
        insert = make_shape(
            insertlen, amplitude, shape=ishape, background=background, **kwargs
        )
        backgr_dict = {
            "low": lambda x: np.zeros(x),
            "high": lambda x: np.full(x, amplitude),
        }
        leftside = backgr_dict[background[0]](insertstart)
        rightside = backgr_dict[background[1]](rightlen)
    target = np.concatenate([leftside, insert, rightside])
    return target[:length]


def select(
    energies: Iterable[np.ndarray],
    weights: Iterable[float],
    cur_energy: np.ndarray,
    temperature: float,
    step: Union[None, int] = None,
    rng: np.random.Generator = np.random.default_rng(),
) -> Tuple[np.ndarray, np.ndarray]:
    """Choose a mutation with low energy based on kMC method.

    Parameters
    ----------
    energies : iterable of ndarrays, len=n_energies
        Iterable of energy components of the total energy. Energy components must
        be of shape (n_seqs, n_mutations)
    weights : iterable of floats, len=n_energies
        Value of weights for each energy component.
    cur_energy : ndarray, shape=(n_seqs,)
        Energies of the sequences before mutation.
    temperature : float
        Temperature to use for deriving probabilities from energies in the kMC
    step : int, optional
        Step index in the optimisation process. If set, the computed
        probabilities will be saved to a file.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    sel_idx : ndarray, shape=(n_seqs,)
        Array of indices of selected mutation for each sequence
    sel_energy : ndarray, shape=(n_seqs, n_energies+1)
        Array of energies associated to selected mutations for each sequence.
        First is total energy, followed by each component.

    Notes
    -----
    The kinetic Monte-Carlo (kMC) method consists in deriving energy values
    for all options and selecting a low energy option in a probabilistic way
    by computing exp(-E_i/T) / sum_i(exp(-E_i/T)) where T is the temperature.
    """
    # Compute energy and probability
    tot_energy = sum(
        w * e for w, e in zip(weights, energies)
    )  # shape (n_seqs, n_mutations)
    prob = utils.exp_normalize(
        (cur_energy.reshape(-1, 1) - tot_energy) / temperature
    )  # shape (n_seqs, n_mutations)
    # Maybe save probabilities
    if step is not None:
        np.save(Path(args.output_dir, "probs", f"prob_step{step}.npy"), prob)
    # Select by the position of a random number in the cumulative sum
    cumprob = np.cumsum(prob, axis=-1)  # shape (n_seqs, n_mutations)
    r = rng.random((len(prob), 1))  # shape (n_seqs,)
    sel_idx = np.argmax(r <= cumprob, axis=-1)  # shape (n_seqs,)
    # Associate energy to selected sequences
    sel_energies = np.stack(
        [en[np.arange(len(en)), sel_idx] for en in [tot_energy] + energies], axis=1
    )  # shape (n_seqs, n_energies+1)
    return sel_idx, sel_energies


def get_rows_and_cols(n_seqs: int) -> Tuple[int, int]:
    """Determine number of rows and columns to place a number of subplots.

    Parameters
    ----------
    n_seqs : int
        Number of subplots to place

    Returns
    -------
    nrow, ncol
        Number of rows and columns to use
    """
    if n_seqs > 25:
        nrow, ncol = 5, 5  # plot max 25 sequences
    else:
        ncol = 5  # max number of columns
        while n_seqs % ncol != 0:
            ncol -= 1
        nrow = n_seqs // ncol
    return nrow, ncol


def main(args):
    # Load model and derive model information
    if args.distribute:
        with tf.distribute.MirroredStrategy().scope():
            model = tf.keras.models.load_model(
                args.model_file,
                custom_objects={"correlate": correlate, "mae_cor": mae_cor},
            )
    else:
        model = tf.keras.models.load_model(
            args.model_file, custom_objects={"correlate": correlate, "mae_cor": mae_cor}
        )
    shape_array = np.array(model.input_shape)
    extradims = list(np.where(shape_array == 1)[0])
    squeezed_shape = shape_array[shape_array != 1]
    winsize = squeezed_shape[1]
    assert len(squeezed_shape) == 3 and squeezed_shape[2] == 4

    # Tune predicting functions according to args
    if model.output_shape[1] == 1:
        head_interval = None
    else:
        head_interval = winsize // model.output_shape[1]
        assert head_interval % args.stride == 0

    def one_hot_converter(idx):
        return np_idx_to_one_hot(idx, order=args.one_hot_order, extradims=extradims)

    def predicter(seqs, reverse=False, offset=None, flanks=None):
        return get_profile_chunk(
            seqs,
            model,
            winsize,
            head_interval=head_interval,
            middle=args.middle_pred,
            reverse=reverse,
            flanks=flanks,
            stride=args.stride,
            offset=offset,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size,
            one_hot_converter=one_hot_converter,
            verbose=False,
            return_index=True,
        )

    def change_predicter(
        seqs_mut,
        mutpos,
        cur_preds,
        cur_indices,
        reverse=False,
        offset=None,
        flanks=None,
    ):
        return get_profile_mutations(
            seqs_mut,
            mutpos,
            cur_preds,
            predicter,
            winsize,
            head_interval=head_interval,
            middle=args.middle_pred,
            reverse=reverse,
            flanks=flanks,
            stride=args.stride,
            offset=offset,
            return_index=True,
            cur_indices=cur_indices,
        )

    # Extract flanking sequences
    if args.flanks in ["random", "self"]:
        if head_interval is not None:
            pad = head_interval - 1
            # Increase distances in case of middle predictions
            if args.middle_pred:
                pad += winsize // 4
        else:
            pad = winsize // 2
        flanks = args.flanks
    elif args.flanks is not None:
        with np.load(args.flanks) as f:
            flank_left = f["left"]
            flank_right = f["right"]
            assert flank_left.ndim == flank_right.ndim
            if flank_left.ndim == 2:
                assert len(flank_left) == len(flank_right)
                flanks = "choose_idx"
            else:
                assert flank_left.ndim == 1
                flanks = (flank_left, flank_right)
    else:
        flanks = None
    # Extract kmer distribution
    if args.kmer_file is None:
        # Use target gc
        freq_kmer = pd.Series(
            [
                (1 - args.target_gc) / 2,
                args.target_gc / 2,
                args.target_gc / 2,
                (1 - args.target_gc) / 2,
            ],
            index=list("ACGT"),
        )
    else:
        # Infer value of k
        with open(args.kmer_file) as f:
            for k, c in enumerate(f.readline()):
                if c != ",":
                    break
        freq_kmer = pd.read_csv(args.kmer_file, index_col=np.arange(k)).iloc[:, 0]
    # Generate and save start sequences
    rng = np.random.default_rng(args.seed)
    if args.start_seqs:
        seqs = np.load(args.start_seqs)
    else:
        seqs = utils.random_sequences(args.n_seqs, args.length, freq_kmer, out="idx")
    np.save(Path(args.output_dir, "designed_seqs", "start_seqs.npy"), seqs)
    # Extract positions where mutations can be performed
    if args.mutfree_pos_file is not None:
        mutfree_pos = np.load(args.mutfree_pos_file)
    else:
        mutfree_pos = np.arange(seqs.shape[-1])
    # Maximum number of positions per sequence to mutate per mutation batch
    pos_batch_size = args.mut_batch_size // (3 * args.n_seqs * (args.length // 1000))
    if args.verbose:
        print(pos_batch_size)
    # Compute energy of start sequences
    # Predict on forward and reverse strands
    if flanks == "random":
        randseqs = utils.random_sequences(2, pad, freq_kmer, out="idx")
        flanks = (randseqs[0], randseqs[1])
    elif flanks == "choose_idx":
        flank_idx = rng.integers(0, len(flank_left))
        flanks = (flank_left[flank_idx], flank_right[flank_idx])
        if args.verbose:
            print(f"Using flank_idx {flank_idx}")
    cur_preds, cur_indices = predicter(
        seqs, offset=0, flanks=flanks
    )  # shape (n_seqs, pred_len) both
    cur_preds_rev, cur_indices_rev = predicter(
        seqs, offset=0, flanks=flanks, reverse=True
    )  # shape (n_seqs, pred_len) both
    # Compute energy
    gc_energy = GC_energy(seqs, args.gc_constrlen, args.target_gc, args.gctol)
    for_energy = args.loss(args.target[cur_indices], cur_preds)
    rev_energy = args.loss(args.target_rev[cur_indices_rev], cur_preds_rev)
    energy_list = [gc_energy, for_energy, rev_energy, np.zeros(args.n_seqs)]
    cur_energy = sum(w * e for w, e in zip(args.weights, energy_list))
    # Initialise best energy and lag
    best_energy = cur_energy
    best_lag = np.zeros(args.n_seqs, dtype=int)
    with open(Path(args.output_dir, "energy.txt"), "a") as f:
        np.savetxt(
            f,
            np.stack([cur_energy] + energy_list, axis=1),
            fmt="%-8e",
            delimiter="\t",
            header="start",
        )
    # Initialize array of already seen bases for each position
    bases_occs = np.eye(4, dtype=int)[seqs]
    # Determine figure parameters
    nrow, ncol = get_rows_and_cols(args.n_seqs)
    target_by_strand = not np.all(args.target == args.target_rev)
    for step in range(args.steps):
        if args.verbose:
            print(datetime.datetime.now() - tmstmp)
            print(f"Step {step}")
        # Get flanks and offset
        if flanks == "random":
            randseqs = utils.random_sequences(2, pad, freq_kmer, out="idx")
            flanks = (randseqs[0], randseqs[1])
        elif flanks == "choose_idx":
            flank_idx = rng.integers(0, len(flank_left))
            flanks = (flank_left[flank_idx], flank_right[flank_idx])
            if args.verbose:
                print(f"Using flank_idx {flank_idx}")
        offset_for = rng.integers(0, args.stride)
        offset_rev = rng.integers(0, args.stride)
        if args.verbose:
            print(
                f"Predicting with stride {args.stride} and offset {offset_for} "
                f"on forward and {offset_rev} on reverse"
            )
        # Subsample mutation positions
        if args.nmut_step != 0 and args.nmut_step < len(mutfree_pos):
            # Choose a position to start scanning from, with discretized possibility using a step
            max_start_mut = len(mutfree_pos) - args.nmut_step
            step_start_mut = args.nmut_step // 4  # step of discretization
            start_mut = rng.choice(maxstep_linspace(0, max_start_mut, step_start_mut))
            mutfree_pos_step = mutfree_pos[start_mut : start_mut + args.nmut_step]
            if args.verbose:
                print(
                    f"Testing mutations between positions "
                    f"{mutfree_pos_step[0]} and {mutfree_pos_step[-1]}"
                )
        else:
            mutfree_pos_step = mutfree_pos
        # Generate all mutations, and associated mutation energy
        mutpos_batches = np.split(
            mutfree_pos_step,
            np.arange(pos_batch_size, len(mutfree_pos_step), pos_batch_size),
        )
        if args.verbose:
            print(
                f"Scanning mutations on {len(mutfree_pos_step)} positions "
                f"in {len(mutpos_batches)} batches"
            )
        energy_list = [[] for i in range(4)]
        for mutpos_batch in mutpos_batches:
            # Perform mutations on positions of this batch, and get associated energy (occurences)
            seqs_mut, mut_energy = all_mutations(
                seqs, mutpos_batch, return_occs=True, bases_occs=bases_occs
            )  # shape (n_seqs, 3*len(mutpos_batch), length) and (n_seqs, 3*len(mutpos_batch))
            mutpos_batch_ext = np.tile(
                np.repeat(mutpos_batch, 3), (args.n_seqs, 1)
            )  # shape (n_seqs, 3*len(mutpos_batch))
            # Get gc_energy
            gc_energy = GC_energy(
                seqs_mut, args.gc_constrlen, target_gc=args.target_gc
            )  # shape (n_seqs, 3*len(mutpos_batch))
            # Predict on forward and compute forward energy
            preds, indices = change_predicter(
                seqs_mut,
                mutpos_batch_ext,
                cur_preds,
                cur_indices,
                offset=offset_for,
                flanks=flanks,
            )  # shape (n_seqs, 3*len(mutpos_batch), pred_len) both
            for_energy = args.loss(
                args.target[indices], preds
            )  # shape (n_seqs, 3*len(mutpos_batch))
            # Predict on reverse and compute reverse energy
            preds_rev, indices_rev = change_predicter(
                seqs_mut,
                mutpos_batch_ext,
                cur_preds_rev,
                cur_indices_rev,
                offset=offset_rev,
                flanks=flanks,
                reverse=True,
            )  # shape (n_seqs, 3*len(mutpos_batch), pred_len) both
            rev_energy = args.loss(
                args.target_rev[indices_rev], preds_rev
            )  # shape (n_seqs, 3*len(mutpos_batch))
            # Append all energy components to appropriate component in energy_list
            for batch_energy_comp, energy_comp in zip(
                [gc_energy, for_energy, rev_energy, mut_energy], energy_list
            ):
                energy_comp.append(batch_energy_comp)
        # Concatenate batches of energy
        for i, energy_comp in enumerate(energy_list):
            energy_list[i] = np.concatenate(
                energy_comp, axis=1
            )  # shape (n_seqs, 3*mutfree_pos_step)
        # Choose best mutation by kMC method
        sel_idx, sel_energies = select(
            energy_list, args.weights, cur_energy, args.temperature, step=step, rng=rng
        )  # shape (n_seqs,) and (n_seqs, n_energies+1)
        # Extract current energy
        cur_energy = sel_energies[:, 0]  # shape (n_seqs,)
        # Extract mutation position and increment
        sel_mutpos = mutfree_pos_step[sel_idx // 3]  # shape (n_seqs,)
        sel_incr = 1 + sel_idx % 3  # shape (n_seqs,)
        # Perform mutation on seqs and increment bases_occs
        seqs[np.arange(args.n_seqs), sel_mutpos] = (
            seqs[np.arange(args.n_seqs), sel_mutpos] + sel_incr
        ) % 4
        bases_occs[
            np.arange(args.n_seqs), sel_mutpos, seqs[np.arange(args.n_seqs), sel_mutpos]
        ] += 1
        # Recompute current predictions and indices
        cur_preds, cur_indices = change_predicter(
            np.expand_dims(seqs, axis=1),
            np.expand_dims(sel_mutpos, axis=1),
            cur_preds,
            cur_indices,
            offset=offset_for,
            flanks=flanks,
        )  # shape (n_seqs, 1, pred_len) both
        cur_preds = cur_preds.reshape(args.n_seqs, -1)
        cur_indices = cur_indices.reshape(args.n_seqs, -1)
        cur_preds_rev, cur_indices_rev = change_predicter(
            np.expand_dims(seqs, axis=1),
            np.expand_dims(sel_mutpos, axis=1),
            cur_preds_rev,
            cur_indices_rev,
            offset=offset_rev,
            flanks=flanks,
            reverse=True,
        )  # shape (n_seqs, 1, pred_len) both
        cur_preds_rev = cur_preds_rev.reshape(args.n_seqs, -1)
        cur_indices_rev = cur_indices_rev.reshape(args.n_seqs, -1)
        # Save sequence, energy and plot profile
        np.save(
            Path(args.output_dir, "designed_seqs", f"mut_seqs_step{step}.npy"), seqs
        )
        with open(Path(args.output_dir, "energy.txt"), "a") as f:
            np.savetxt(
                f, sel_energies, fmt="%-8e", delimiter="\t", header=f"step{step}"
            )
        fig, axes = plt.subplots(
            nrow,
            ncol,
            figsize=(2 + 3 * ncol, 1 + 2 * nrow),
            facecolor="w",
            layout="tight",
            sharey=True,
        )
        if args.n_seqs == 1:
            ax_list = [axes]
        else:
            ax_list = axes.flatten()
        for ax, predfor, indfor, predrev, indrev in zip(
            ax_list,
            cur_preds,
            cur_indices,
            cur_preds_rev,
            cur_indices_rev,
        ):
            ax.plot(args.target, color="k", label="target")
            if target_by_strand:
                ax.plot(-args.target_rev, color="k")
                predrev = -predrev
            ax.plot(indfor, predfor, label="forward")
            ax.plot(indrev, predrev, label="reverse", alpha=0.8)
            ax.legend()
        fig.savefig(
            Path(args.output_dir, "pred_figs", f"mut_preds_step{step}.png"),
            bbox_inches="tight",
        )
        plt.close()
        # Compute best energy and perform earlystopping
        best_cur_mask = cur_energy < best_energy
        best_lag += 1
        best_lag[best_cur_mask] = 0
        best_energy[best_cur_mask] = cur_energy[best_cur_mask]
        if args.patience > 0 and np.all(best_lag > args.patience):
            break
    if args.verbose:
        print(datetime.datetime.now() - tmstmp)


if __name__ == "__main__":
    tmstmp = datetime.datetime.now()
    # Get arguments
    args = parsing()
    if args.verbose:
        print("Initialization")
    # Build output directory and initialize energy file
    Path(args.output_dir).mkdir(parents=True, exist_ok=False)
    Path(args.output_dir, "pred_figs").mkdir()
    Path(args.output_dir, "designed_seqs").mkdir()
    Path(args.output_dir, "probs").mkdir()
    with open(Path(args.output_dir, "energy.txt"), "w") as f:
        f.write(
            "# total_energy\t"
            "gc_energy\t"
            "for_energy\t"
            "rev_energy\t"
            "mut_energy\n"
        )
    # Store arguments in config file
    to_serealize = {
        k: v for k, v in vars(args).items() if k not in ["target", "target_rev"]
    }
    with open(Path(args.output_dir, "config.txt"), "w") as f:
        json.dump(
            {
                **to_serealize,
                **{
                    "timestamp": str(tmstmp),
                    "machine": socket.gethostname(),
                },
            },
            f,
            indent=4,
        )
        f.write("\n")
    # Convert to non json serializable objects
    losses = {"rmse": rmse, "mae_cor": np_mae_cor}
    args.loss = losses[args.loss]
    args.weights = np.array(args.weights, dtype=float)
    # Save target
    np.savez(
        Path(args.output_dir, "target.npz"),
        forward=args.target,
        reverse=args.target_rev,
    )
    # Start computations, save total time even if there was a failure
    try:
        main(args)
    except:
        with open(Path(args.output_dir, "config.txt"), "a") as f:
            f.write("Aborted\n")
            f.write(f"total time: {datetime.datetime.now() - tmstmp}\n")
        raise
    with open(Path(args.output_dir, "config.txt"), "a") as f:
        f.write(f"total time: {datetime.datetime.now() - tmstmp}\n")
