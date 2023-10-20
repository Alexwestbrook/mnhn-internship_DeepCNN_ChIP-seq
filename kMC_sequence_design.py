from pathlib import Path
import time
import datetime
import argparse
import json
import socket
import gc
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf

from Modules import utils, tf_utils
from Modules.tf_utils import mae_cor, correlate, np_mae_cor


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
        "-o", "--output_dir", type=str, required=True,
        help="output directory")
    parser.add_argument(
        "-m", "--model_file", type=str,
        default='/home/alex/shared_folder/SCerevisiae/Trainedmodels/'
                'model_myco_pol_17/model',
        help="trained model file as hdf5 or in tf2 format")
    parser.add_argument(
        "-ord", "--one_hot_order", type=str, default='ACGT',
        help="order of the one-hot encoding for the model")
    parser.add_argument(
        "-kfile", "--kmer_file", type=str,
        default='/home/alex/shared_folder/SCerevisiae/genome/W303/'
                'W303_3mer_freq.csv',
        help="file with kmer distribution to use for initializing sequences")
    parser.add_argument(
        "-k", type=int, default=3,
        help="value of k for kmer distribution, must be provided to read the "
             "kmer_file correctly")
    parser.add_argument(
        "-n", "--n_seqs", type=int, default=1,
        help="number of sequences to generate simultaneously")
    parser.add_argument(
        "-l", "--length", type=int, default=2175,
        help="length of the sequences to generate")
    parser.add_argument(
        "-start", "--start_seqs", type=str,
        help="numpy binary file containing starting sequences as indexes with "
             "shape (n_seqs, length). If set overrides n_seqs and length.")
    parser.add_argument(
        "--steps", type=int, default=100,
        help="number of steps to perform")
    parser.add_argument(
        "-dist", "--distribute", action='store_true',
        help="indicates to use MirrorStrategy for prediction")
    parser.add_argument(
        "-s", "--stride", type=int, default=1,
        help="specifies a stride in predictions to go faster")
    parser.add_argument(
        "-mid", "--middle_pred", action='store_true',
        help="specifies to predict only on middle window")
    parser.add_argument(
        "--flanks", type=str,
        help="file with flanking sequences to use for prediction, or 'random' "
             "to get random flanks with specified kmer distribution")
    parser.add_argument(
        "-targ", "--target", type=float, nargs='+', default=[0],
        help="target profile for the designed sequences. If a single value, "
             "consider a flat target for the entire sequence, otherwise must "
             "be of same length as the sequences.")
    parser.add_argument(
        "-targ_rev", "--target_rev", type=float, nargs='+',
        help="target profile for the designed sequences on the reverse "
             "strand, if unspecified, consider the same target for forward "
             "and reverse. If a single value, consider a flat target for the "
             "entire sequence, otherwise must be of same length as the "
             "sequences.")
    parser.add_argument(
        "--target_file", type=str,
        help="numpy binary file containing the target profile for the "
             "designed sequences. If set, overrides target, target_rev and "
             "also length, unless start_seqs is set, in which case it must be "
             "of same length as start_seqs. Can also be an npz archive with 2 "
             "different targets for each strand, with keys named 'forward' "
             "and 'reverse'.")
    parser.add_argument(
        "-targ_gc", "--target_gc", type=float, default=0.3834,
        help="target GC content for the designed sequences")
    parser.add_argument(
        "--loss", type=str, default='rmse',
        help="loss to use for comparing prediction and target")
    parser.add_argument(
        "-w", "--weights", type=float, nargs=4, default=[1, 1, 1, 1],
        help="weights for the energy terms in order E_gc, E_for, E_rev, E_mut")
    parser.add_argument(
        "-t", "--temperature", type=float, default=0.1,
        help="temperature for kMC")
    parser.add_argument(
        "-maxopt", "--max_options", type=int,
        help="maximum number of options to apply nonzero probability to. Only "
             "best options are kept for selection. Set to -1 for no maximum.")
    parser.add_argument(
        "-b", "--batch_size", type=int, default=1024,
        help="number of mutations to predict on at once")
    parser.add_argument(
        "-c", "--chunk_size", type=int, default=128000,
        help="number of sequence slides that can fit into memory")
    parser.add_argument(
        "--seed", type=int, default=-1,
        help="seed to use for random generations")
    parser.add_argument(
        "-v", "--verbose", action='store_true',
        help="whether to print information messages")
    args = parser.parse_args()
    # Basic checks
    assert (len(args.one_hot_order) == 4
            and set(args.one_hot_order) == set('ACGT'))
    for item in [args.k, args.n_seqs, args.length, args.steps,
                 args.stride, args.max_options, args.batch_size,
                 args.chunk_size]:
        assert item is None or item >= 1
    assert args.temperature > 0
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
                args.target = f['forward']
                args.target_rev = f['reverse']
        if args.start_seqs is None:
            args.length = len(args.target)
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
            args.target_rev = np.full(args.length, args.target_rev[0],
                                      dtype=float)
        else:
            args.target_rev = np.array(args.target_rev, dtype=float)
    assert (len(args.target) == len(args.target_rev)
            and len(args.target) == args.length)
    return args


def GC_energy(seqs, target_gc):
    """Compute GC energy for each sequence in an array.

    The GC_energy is computed as `abs(gc - target_gc)`

    Parameters
    ----------
    seqs : ndarray
        Array of indexes into 'ACGT', with the sequences on the last axis
    target_gc : float
        Value of the target gc content

    Returns
    -------
    ndarray
        Array of absolute difference between sequence gc and target gc,
        with same shape as seqs but last dimension removed
    """
    return np.abs(np.sum((seqs == 1) | (seqs == 2), axis=-1) / seqs.shape[-1]
                  - target_gc)


def all_mutations(seqs, seen_bases):
    """Perform all possible mutations and associate each its mutation energy.

    Mutation energy of a mutation X -> Y is defined as the number of times Y
    has been seen at this position during the optimization of this particular
    sequence.

    Parameters
    ----------
    seqs : ndarray, shape=(n, l)
        Array of indexes into 'ACGT', n is the number of sequences and l their
        length.
    seen_bases : ndarray, shape=(n, l, 4)
        Array of occurence of each base at each position during the
        optimization process of each sequence

    Returns
    -------
    mutated : ndarray, shape=(n, 3*l, l)
        Array of all possible mutations of each sequence in seqs
    mut_energy : ndarray, shape=(n, 3*l)
        Array of mutation energy associated with each mutation

    Notes
    -----
    Mutations are performed in a circular rotation (0 -> 1 -> 2 -> 3 -> 0) by
    adding 1, 2 or 3 to the index and then taking the modulo 4.
    """
    n_seqs, length = seqs.shape
    # Create mutations array
    # Array of increments of 1, 2 or 3 at each position for a single sequence
    single_seq_increments = utils.repeat_along_diag(
        np.arange(1, 4, dtype=seqs.dtype).reshape(-1, 1),
        length)  # shape (3*length, length)
    # Add increments to each sequence by broadcasting and take modulo 4
    mutated = (np.expand_dims(seqs, axis=1)
               + np.expand_dims(single_seq_increments, axis=0)
               ) % 4
    # Associate each mutation with its mutation energy
    # Array of resulting bases for each mutation
    mut_idx = (np.expand_dims(seqs, axis=-1)
               + np.arange(1, 4).reshape(1, -1)
               ) % 4  # shape (n_seqs, length, 3)
    # Select values associated with each base in seen_bases
    mut_energy = np.take_along_axis(seen_bases, mut_idx, axis=-1
                                    ).reshape(n_seqs, 3*length)
    return mutated, mut_energy


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


def get_profile_hint(seqs, model, winsize, head_interval, reverse=False,
                     middle=False, stride=1, batch_size=1024,
                     chunk_size=128000, one_hot_converter=np_idx_to_one_hot,
                     offset=None, seed=None, verbose=False,
                     return_index=False, flanks=None):
    """Predict profile.

    Parameters
    ----------
    seqs : ndarray
        Array of indexes into 'ACGT', sequences to predict on are read on the
        last axis
    model
        Tensorflow model instance supporting the predict method, with multiple
        outputs starting at first window position and with regular spacing
    winsize : int
        Input length of the model
    head_interval : int
        Spacing beatween outputs of the model
    reverse : bool, default=False
        If True predict on reverse strand. Default is False for predicting on
        forward strand.
    middle : bool, default=False
        Whether to use only the middle half of output heads for deriving
        predictions. This results in no predictions on sequence edges.
    stride : int, default=1
        Stride to use for prediction. Using a value other than 1 will result
        in bases being skipped and make prediction faster
    batch_size : int, default=1024
        Batch_size for model.predict().
    chunk_size : int, default=128000
        Number of full sequences to feed to model.predict() at once. This is
        different from the model batch_size. The constraint here is the total
        number of one-hot encoded sequence slides that can fit into memory.
    one_hot_converter : function, default=np_idx_to_one_hot
        Function taking as input an array of indexes into 'ACGT' with shape
        (n_windows, window_length) and converts it to the required model input
        format.
    offset : int, default=None
        Offset for start of prediciton, will be forced to be positive and
        smaller than stride by taking the modulo. Default value of None will
        result in a random offset being chosen.
    seed : int, default=None
        Value of seed to use for choosing random offset.
    verbose : bool, default=False
        If True, print information messages.
    return_index : bool, default=False
        If True, return indices corresponding to the predictions.
    flanks : tuple of array, default=None
        Tuple of 2 arrays flank_keft and flank_right to be added at the start
        and end respectively of each sequence to get prediction on the entire
        sequence.

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
        if seed is not None:
            np.random.seed(seed)
        offset = np.random.randint(0, stride)
    else:
        offset %= stride
    if verbose:
        print(f'Predicting with stride {stride} and offset {offset}')
    # pred_start: distance between fisrt prediction and sequence start
    # pred_stop: distance between last prediction and sequence end
    pred_start = 0
    pred_stop = head_interval - 1
    # Increase distances in case of middle predictions
    if middle:
        pred_start += winsize // 4
        pred_stop += winsize // 4
    # Add flanking sequences to make prediction along the entire sequence, and
    # update distances
    if flanks is not None:
        flank_left, flank_right = flanks
        flank_left = flank_left[len(flank_left)-pred_start:]
        pred_start -= len(flank_left)
        flank_right = flank_right[:pred_stop]
        pred_stop -= len(flank_right)
        seqs2D = np.hstack([np.tile(flank_left, (len(seqs2D), 1)),
                            seqs2D,
                            np.tile(flank_right, (len(seqs2D), 1))])
    # Determine indices of predictions along the sequence axis
    if return_index:
        indices = np.arange(pred_start + offset,
                            seqs.shape[-1] - pred_stop,
                            stride)
        if reverse:
            indices = np.flip(seqs.shape[-1] - indices - 1)
    # Maybe reverse complement the sequences
    if reverse:
        seqs2D[seqs2D == 0] = -1
        seqs2D[seqs2D == 3] = 0
        seqs2D[seqs2D == -1] = 3
        seqs2D[seqs2D == 1] = -1
        seqs2D[seqs2D == 2] = 1
        seqs2D[seqs2D == -1] = 2
        seqs2D = np.flip(seqs2D, axis=-1)
    # Determine number of windows in a slide, number of heads, of kept heads
    # and length to jump once a slide is done
    slide_length = head_interval // stride
    n_heads = winsize // head_interval
    if middle:
        jump_stride = winsize // 2
        n_kept_heads = n_heads // 2
    else:
        jump_stride = winsize
        n_kept_heads = n_heads
    # Get windows to predict on, and split them into chunks
    windows = utils.strided_sliding_window_view(
        seqs2D[:, offset:],
        (len(seqs2D), winsize),
        stride=(jump_stride, stride),
        sliding_len=head_interval,
        axis=-1).reshape(-1, winsize)
    chunks = np.split(windows,
                      np.arange(chunk_size, len(windows), chunk_size))

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
            axis=-1).reshape(-1, winsize)
        chunks += np.split(
            extra_windows,
            np.arange(chunk_size, len(extra_windows), chunk_size))

    # Make predictions
    preds = []
    for chunk in chunks:
        # Convert to one-hot and predict
        pred = model.predict(one_hot_converter(chunk),
                             batch_size=batch_size).squeeze()
        # Collect garbage to prevent memory leak from model.predict()
        gc.collect()
        preds.append(pred)
    preds = np.concatenate(preds)
    # Maybe extract middle prediction
    if middle:
        preds = preds[:, n_heads//4:3*n_heads//4]
    # Transpose slide_length and n_kept_heads to get proper sequence order
    preds = np.transpose(preds.reshape(-1, slide_length, n_kept_heads),
                         [0, 2, 1])
    # Seperate last slide prediction to truncate its beginning then put it back
    if last_jump != 0:
        preds = preds.reshape(-1, len(seqs2D), slide_length*n_kept_heads)
        first_part = preds[:-1, :, :].reshape(len(seqs2D), -1)
        last_part = preds[-1, :, -(last_jump // stride):]
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


def get_profile_chunk(seqs, model, winsize, head_interval=None, middle=False,
                      reverse=False, stride=1, batch_size=1024,
                      chunk_size=128000, one_hot_converter=np_idx_to_one_hot,
                      offset=None, seed=None, verbose=False,
                      return_index=False, flanks=None):
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
    head_interval : int, default=None
        Spacing between outputs of the model, for a model with multiple
        outputs starting at first window position and with regular spacing.
        If None, model must have a single output in the middle of the window.
    middle : bool, default=False
        Whether to use only the middle half of output heads for deriving
        predictions. This results in no predictions on sequence edges.
    reverse : bool, default=False
        If True predict on reverse strand. Default is False for predicting on
        forward strand.
    stride : int, default=1
        Stride to use for prediction. Using a value other than 1 will result
        in bases being skipped and make prediction faster
    batch_size : int, default=1024
        Batch_size for model.predict().
    chunk_size : int, default=128000
        Number of full sequences to feed to model.predict() at once. This is
        different from the model batch_size. The constraint here is the total
        number of one-hot encoded sequence slides that can fit into memory.
    one_hot_converter : function, default=np_idx_to_one_hot
        Function taking as input an array of indexes into 'ACGT' with shape
        (n_windows, window_length) and converts it to the required model input
        format.
    offset : int, default=None
        Offset for start of prediciton, will be forced to be positive and
        smaller than stride by taking the modulo. Default value of None will
        result in a random offset being chosen.
    seed : int, default=None
        Value of seed to use for choosing random offset.
    verbose : bool, default=False
        If True, print information messages.
    return_index : bool, default=False
        If True, return indices corresponding to the predictions.
    flanks : tuple of array, default=None
        Tuple of 2 arrays flank_keft and flank_right to be added at the start
        and end respectively of each sequence to get prediction on the entire
        sequence.

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
        if seed is not None:
            np.random.seed(seed)
        offset = np.random.randint(0, stride)
    else:
        offset %= stride
    if verbose:
        print(f'Predicting with stride {stride} and offset {offset}')
    # pred_start: distance between fisrt prediction and sequence start
    # pred_stop: distance between last prediction and sequence end
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
    # Add flanking sequences to make prediction along the entire sequence, and
    # update distances
    if flanks is not None:
        flank_left, flank_right = flanks
        flank_left = flank_left[len(flank_left)-pred_start:]
        pred_start -= len(flank_left)
        flank_right = flank_right[:pred_stop]
        pred_stop -= len(flank_right)
        seqs2D = np.hstack([np.tile(flank_left, (len(seqs2D), 1)),
                            seqs2D,
                            np.tile(flank_right, (len(seqs2D), 1))])
    # Determine indices of predictions along the sequence axis
    if return_index:
        indices = np.arange(pred_start + offset,
                            seqs.shape[-1] - pred_stop,
                            stride)
        if reverse:
            indices = np.flip(seqs.shape[-1] - indices - 1)
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
            axis=-1).reshape(-1, winsize)
        chunks = np.split(windows,
                          np.arange(chunk_size, len(windows), chunk_size))

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
                axis=-1).reshape(-1, winsize)
            chunks += np.split(
                extra_windows,
                np.arange(chunk_size, len(extra_windows), chunk_size))
    else:
        windows = utils.strided_window_view(
            seqs2D[:, offset:],
            (len(seqs2D), winsize),
            stride=(1, stride)).transpose([0, 2, 1, 3]).reshape(-1, winsize)
        chunks = np.split(windows,
                          np.arange(chunk_size, len(windows), chunk_size))
    # Make predictions
    preds = []
    for chunk in chunks:
        # Convert to one-hot and predict
        pred = model.predict(one_hot_converter(chunk),
                             batch_size=batch_size).squeeze()
        # Collect garbage to prevent memory leak from model.predict()
        gc.collect()
        preds.append(pred)
    preds = np.concatenate(preds)
    # Reformat predictions
    if head_interval is not None:
        # Maybe extract middle prediction
        if middle:
            preds = preds[:, n_heads//4:3*n_heads//4]
        # Transpose slide_length and n_kept_heads to get proper sequence order
        preds = np.transpose(preds.reshape(-1, slide_length, n_kept_heads),
                             [0, 2, 1])
        # Seperate last slide prediction to truncate its beginning then put it
        # back
        if last_jump != 0:
            preds = preds.reshape(-1, len(seqs2D), slide_length*n_kept_heads)
            first_part = preds[:-1, :, :].reshape(len(seqs2D), -1)
            last_part = preds[-1, :, -(last_jump // stride):]
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


def rmse(y_true, y_pred):
    """Compute RMSE between two arrays.

    Parameters
    ----------
    y_true, y_pred : ndarray
        Arrays of values. If multidimensional, computation is performed along
        the last axis.

    Returns
    -------
    ndarray
        Array of same shape as y_true and y_pred but with last axis removed
    """
    return np.sqrt(np.mean((y_true - y_pred)**2, axis=-1))


def select(energies, weights, temperature, step=None, maxoptions=None):
    """Choose a mutation with low energy based on kMC method.

    Parameters
    ----------
    energies : list of arrays, len=n_energies
        List of energy components of the total energy. Energy components must
        be of shape (n_seqs, n_mutations)
    weights : array_like, len=n_energies
        Value of weights for each energy component.
    temperature : float
        Temperature to use for deriving probabilities from energies in the kMC
    step : int, default=None
        Step index in the optimisation process. If set, the computed
        probabilities will be save to a file.
    maxoptions : int, default=None
        Maximum number of options to give nonzero probabilities to.

    Returns
    -------
    sel_idx : ndarray, shape=(n_seqs)
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
    tot_energy = sum(w*e for w, e in zip(weights, energies))
    if maxoptions is not None:
        thres = np.partition(tot_energy, maxoptions, axis=-1)[:, maxoptions]
        tot_energy[tot_energy > np.expand_dims(thres, axis=-1)] = np.inf
    prob = utils.exp_normalize(-tot_energy / temperature)
    # Maybe save probabilities
    if step is not None:
        np.save(Path(args.output_dir, "probs", f'prob_step{step}.npy'), prob)
    # Select by the position of a random number in the cumulative sum
    cumprob = np.cumsum(prob, axis=-1)
    r = np.random.rand(len(prob), 1)
    sel_idx = np.argmax(r <= cumprob, axis=-1)
    # Associate energy to selected sequences
    sel_energies = np.stack(
        [en[np.arange(len(en)), sel_idx]
         for en in [tot_energy] + energies],
        axis=1)
    return sel_idx, sel_energies


def get_rows_and_cols(n_seqs):
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
                custom_objects={'correlate': correlate, 'mae_cor': mae_cor})
    else:
        model = tf.keras.models.load_model(
            args.model_file,
            custom_objects={'correlate': correlate, 'mae_cor': mae_cor})
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
        return np_idx_to_one_hot(
            idx, order=args.one_hot_order, extradims=extradims)

    def predicter(seqs, reverse=False, offset=None, flanks=None):
        return get_profile_chunk(
            seqs, model, winsize, head_interval=head_interval,
            middle=args.middle_pred, reverse=reverse, stride=args.stride,
            batch_size=args.batch_size, chunk_size=args.chunk_size,
            one_hot_converter=one_hot_converter, offset=offset,
            verbose=args.verbose, return_index=True, flanks=flanks)
    # Extract flanking sequences
    if args.flanks == 'random':
        if head_interval is not None:
            leftpad = 0
            rightpad = head_interval - 1
            # Increase distances in case of middle predictions
            if args.middle_pred:
                leftpad += winsize // 4
                rightpad += winsize // 4
        else:
            leftpad = winsize // 2
            rightpad = winsize // 2
    elif args.flanks is not None:
        with np.load(args.flanks) as f:
            flank_left = f['left']
            flank_right = f['right']
            assert flank_left.ndim == flank_right.ndim
            if flank_left.ndim == 2:
                assert len(flank_left) == len(flank_right)
                flanks = 'choose_idx'
            else:
                assert flank_left.ndim == 1
                flanks = (flank_left, flank_right)
    else:
        flanks = None
    # Extract kmer distribution
    freq_kmer = pd.read_csv(args.kmer_file,
                            index_col=[i for i in range(args.k)])
    # Generate and save start sequences
    if args.seed != -1:
        np.random.seed(args.seed)
    if args.start_seqs:
        seqs = np.load(args.start_seqs)
    else:
        seqs = utils.random_sequences(args.n_seqs, args.length,
                                      freq_kmer.iloc[:, 0], out='idx')
        np.save(Path(args.output_dir, "designed_seqs", "start_seqs.npy"), seqs)
    # Initialize array of already seen bases for each position
    seen_bases = np.eye(4, dtype=int)[seqs]
    # Determine figure parameters
    nrow, ncol = get_rows_and_cols(args.n_seqs)
    target_by_strand = not np.all(args.target == args.target_rev)
    for step in range(args.steps):
        if args.verbose:
            print(time.time() - t0)
            print(f'Step {step}')
        # Generate all mutations, and associated mutation energy
        seqs, mut_energy = all_mutations(seqs, seen_bases)
        # Predict on forward and reverse strands
        if args.flanks == 'random':
            flanks = (utils.random_sequences(1, leftpad, freq_kmer.iloc[:, 0],
                                             out='idx').ravel(),
                      utils.random_sequences(1, rightpad, freq_kmer.iloc[:, 0],
                                             out='idx').ravel())
        elif flanks == 'choose_idx':
            flank_idx = np.random.randint(0, len(flank_left))
            flanks = (flank_left[flank_idx], flank_right[flank_idx])
            if args.verbose:
                print(f"Using flank_idx {flank_idx}")
        preds, indices = predicter(
            seqs, offset=np.random.randint(0, args.stride), flanks=flanks)
        preds_rev, indices_rev = predicter(
            seqs, offset=np.random.randint(0, args.stride), flanks=flanks,
            reverse=True)
        # Compute energy components
        gc_energy = GC_energy(seqs, args.target_gc)
        for_energy = args.loss(args.target[indices], preds)
        rev_energy = args.loss(args.target_rev[indices_rev], preds_rev)
        energy_list = [gc_energy, for_energy, rev_energy, mut_energy]
        # Choose best mutation by kMC method
        sel_idx, sel_energies = select(energy_list, args.weights,
                                       args.temperature, step=step,
                                       maxoptions=args.max_options)
        # Keep selected sequence and increment seen_bases
        seqs = seqs[np.arange(len(seqs)), sel_idx]
        seen_bases[np.arange(len(seqs)),
                   sel_idx // 3,
                   seqs[np.arange(len(seqs)), sel_idx // 3]] += 1
        # Save sequence, energy and plot profile
        np.save(Path(args.output_dir, "designed_seqs",
                     f"mut_seqs_step{step}.npy"),
                seqs)
        with open(Path(args.output_dir, 'energy.txt'), 'a') as f:
            np.savetxt(f, sel_energies, fmt='%-8e',
                       delimiter='\t', header=f'step{step}')
        fig, axes = plt.subplots(nrow, ncol, figsize=(2+3*ncol, 1+2*nrow),
                                 facecolor='w', layout='tight', sharey=True)
        if args.n_seqs == 1:
            ax_list = [axes]
        else:
            ax_list = axes.flatten()
        for ax, pfor, prev in zip(ax_list,
                                  preds[np.arange(len(seqs)), sel_idx],
                                  preds_rev[np.arange(len(seqs)), sel_idx]
                                  ):
            ax.plot(args.target, color='k', label='target')
            if target_by_strand:
                ax.plot(-args.target_rev, color='k')
                prev = -prev
            ax.plot(indices, pfor, label='forward')
            ax.plot(indices_rev, prev, label='reverse', alpha=0.8)
            ax.legend()
        fig.savefig(Path(args.output_dir, "pred_figs",
                         f"mut_preds_step{step}.png"),
                    bbox_inches='tight')
        plt.close()
    if args.verbose:
        print(time.time() - t0)


if __name__ == "__main__":
    tmstmp = datetime.datetime.now()
    t0 = time.time()
    # Get arguments
    args = parsing()
    if args.verbose:
        print('Initialization')
    # Build output directory and initialize energy file
    Path(args.output_dir).mkdir(parents=True, exist_ok=False)
    Path(args.output_dir, 'pred_figs').mkdir()
    Path(args.output_dir, 'designed_seqs').mkdir()
    Path(args.output_dir, 'probs').mkdir()
    with open(Path(args.output_dir, 'energy.txt'), 'w') as f:
        f.write('# total_energy\t'
                'gc_energy\t'
                'for_energy\t'
                'rev_energy\t'
                'mut_energy\n')
    # Store arguments in config file
    to_serealize = {k: v for k, v in vars(args).items()
                    if k not in ['target', 'target_rev', 'weights']}
    with open(Path(args.output_dir, 'config.txt'), 'w') as f:
        json.dump(to_serealize, f, indent=4)
        f.write('\n')
        f.write(f'weights: {args.weights}\n')
        f.write(f'timestamp: {tmstmp}\n')
        f.write(f'machine: {socket.gethostname()}\n')
    # Convert to non json serializable objects
    losses = {'rmse': rmse, 'mae_cor': np_mae_cor}
    args.loss = losses[args.loss]
    args.weights = np.array(args.weights, dtype=float)
    # Save target
    np.savez(Path(args.output_dir, 'target.npz'),
             forward=args.target, reverse=args.target_rev)
    # Start computations, save total time even if there was a failure
    try:
        main(args)
    except KeyboardInterrupt:
        with open(Path(args.output_dir, 'config.txt'), 'a') as f:
            f.write('KeyboardInterrupt\n')
            f.write(f'total time: {time.time() - t0}\n')
        raise
    with open(Path(args.output_dir, 'config.txt'), 'a') as f:
        f.write(f'total time: {time.time() - t0}\n')
