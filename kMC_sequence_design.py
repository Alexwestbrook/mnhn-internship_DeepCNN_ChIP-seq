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

from Modules import utils
from Modules.tf_utils import mae_cor, correlate


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
        "--extradims", type=int, nargs='+',
        help="extra dimensions for the one-hot input of the model, which by "
             "default is of shape (batch_size, length, 4)")
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
        "-nseq", "--n_seqs", type=int, default=1,
        help="number of sequences to generate simultaneously")
    parser.add_argument(
        "-l", "--length", type=int, default=2175,
        help="length of the sequences to generate")
    parser.add_argument(
        "--start_seqs", type=str,
        help="numpy binary file containing starting sequences as indexes with "
             "shape (n_seqs, length). If set overrides kmer_file, k, n_seqs "
             "and length.")
    parser.add_argument(
        "--steps", type=int, default=100,
        help="number of steps to perform")
    parser.add_argument(
        "-dist", "--distribute", action='store_true',
        help="indicates to use MirrorStrategy for prediction")
    parser.add_argument(
        "--stride", type=int, default=1,
        help="specifies a stride in predictions to go faster")
    parser.add_argument(
        "-mid", "--middle_pred", action='store_true',
        help="specifies to predict only on middle window")
    parser.add_argument(
        "-targ", "--target", type=float, nargs='+',
        help="target profile for the designed sequences")
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
        "-temp", "--temperature", type=float, default=0.1,
        help="temperature for kMC")
    parser.add_argument(
        "-maxopt", "--max_options", type=float, default=1000,
        help="maximum number of options to apply nonzero probability to. Only "
             "best options are kept for selection. Set to -1 for no maximum.")
    parser.add_argument(
        "-b", "--batch_size", type=int, default=1000,
        help="number of mutations to predict on at once")
    parser.add_argument(
        "--seed", type=int, default=-1,
        help="seed to use for random generations")
    args = parser.parse_args()
    assert (len(args.one_hot_order) == 4
            and set(args.one_hot_order) == set('ACGT'))
    for item in [args.k, args.n_seqs, args.length, args.steps,
                 args.stride, args.batch_size]:
        assert item >= 1
    assert args.temperature > 0
    if args.target is None:
        args.target = np.zeros(args.length)
    else:
        args.target = np.array(args.target, dtype=float)
    if args.start_seqs is not None:
        seqs = np.load(args.start_seqs)
        assert seqs.ndim == 2
        args.n_seqs, args.length = seqs.shape
        args.k, args.kmer_file = None, None
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


def tf_idx_to_one_hot(idx, order='ACGT', extradims=None):
    """Convert array of indexes into one-hot in tf format.

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
        idx.shape + (4,)

    Returns
    -------
    Tensor
        Tensor with same shape as idx, in one-hot format.
    """
    assert (len(order) == 4 and set(order) == set('ACGT'))
    converter = np.zeros((4, 4))
    for i, c in enumerate('ACGT'):
        converter[i, order.find(c)] = 1
    one_hot = converter[idx]
    if extradims is not None:
        one_hot = np.expand_dims(one_hot, axis=extradims)
    return tf.convert_to_tensor(one_hot, dtype=tf.float32)


def get_profile_hint(seqs, model, winsize, head_interval, reverse=False,
                     middle=True, stride=1, batch_size=1000,
                     one_hot_converter=tf_idx_to_one_hot):
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
    middle : bool, default=True
        Whether to use only the middle half of output heads for deriving
        predictions. This results in no predictions on sequence edges.
    stride : int, default=1
        Stride to use for prediction. Using a value other than 1 will result
        in bases being skipped and make prediction faster
    batch_size : int, default=1000
        Number of full sequences to feed to model.predict() at once. This is
        different from the model batch_size, which is derived inside
        model.predict(). The constraint here is the total number of one-hot
        encoded sequence slides that can fit into memory.
    one_hot_converter : function
        Function taking as input an array of indexes into 'ACGT' with shape
        (n_windows, window_length) and converts it to the required model input
        format.

    Returns
    -------
    ndarray
        Array of predictions with same shape as seqs, except on the last
        dimension, containing predictions for that sequence.
    """
    # Copy sequence array and make 2D with sequence along the 2nd axis
    seqs2D = seqs.reshape(-1, seqs.shape[-1]).copy()
    # Maybe reverse complement the sequences
    if reverse:
        seqs2D[seqs2D == 0] = -1
        seqs2D[seqs2D == 3] = 0
        seqs2D[seqs2D == -1] = 3
        seqs2D[seqs2D == 1] = -1
        seqs2D[seqs2D == 2] = 1
        seqs2D[seqs2D == -1] = 2
        seqs2D = np.flip(seqs2D, axis=-1)
    # Determine number of heads, of kept heads and length to jump once sliding
    # prediction is done
    n_heads = winsize // head_interval
    if middle:
        jump_stride = winsize // 2
        kept_heads = n_heads // 2
    else:
        jump_stride = winsize
        kept_heads = n_heads
    if stride != 1:
        # Do not predict all bases, but randomize which ones are predicted
        # each time
        start = np.random.randint(0, stride)
        print(f'Predicting with stride {stride} and offset {start}')
        # Allow higher batch_size
        batch_size *= stride
    else:
        start = 0
    # Determine number of slides for reshaping
    slides_per_seq = head_interval // stride
    # Get windows to predict on
    slides = utils.strided_sliding_window_view(
        seqs2D[:, start:],
        (len(seqs2D), winsize),
        stride=(jump_stride, stride),
        sliding_len=head_interval,
        axis=-1).reshape(-1, slides_per_seq, winsize)
    # Initialise predictions list
    preds = []
    # Process slides by batch because it may use to much memory, batching is
    # done only along first dimension because all slides_per_seq must be
    # processed together
    for split in np.split(slides,
                          np.arange(batch_size, len(slides), batch_size)):
        # Convert to one-hot and predict
        pred = model.predict(
            one_hot_converter(split.reshape(-1, winsize))
        ).squeeze()
        # Collect garbage to prevent memory leak from model.predict()
        gc.collect()
        # Maybe extract middle prediction
        if middle:
            pred = pred[:, n_heads//4:3*n_heads//4]
        # Reorder and reshape predictions, then add to list
        pred = np.transpose(pred.reshape(-1, slides_per_seq, kept_heads),
                            [0, 2, 1])
        pred = pred.reshape(-1, slides_per_seq*kept_heads)
        preds.append(pred)
    # Concatenate predictions and reshape as original sequences
    preds = np.concatenate(preds).reshape(seqs.shape[:-1] + (-1,))
    # Maybe reverse predictions
    if reverse:
        preds = np.flip(preds, axis=-1)
    return preds


def get_profile_mid1(seqs, model, winsize, reverse=False, stride=1,
                     batch_size=1000, one_hot_converter=tf_idx_to_one_hot):
    """Predict profile.

    Parameters
    ----------
    seqs : ndarray
        Array of indexes into 'ACGT', sequences to predict on are read on the
        last axis
    model
        Tensorflow model instance supporting the predict method, with a single
        output in the middle of the window.
    winsize : int
        Input length of the model
    reverse : bool, default=False
        If True predict on reverse strand. Default is False for predicting on
        forward strand.
    stride : int, default=1
        Stride to use for prediction. Using a value other than 1 will result
        in bases being skipped and make prediction faster
    batch_size : int, default=1000
        Number of full sequences to feed to model.predict() at once. This is
        different from the model batch_size, which is derived inside
        model.predict(). The constraint here is the total number of one-hot
        encoded sequence slides that can fit into memory.
    one_hot_converter : function
        Function taking as input an array of indexes into 'ACGT' with shape
        (n_seqs, seq_length) and converts it to the required model input
        format.

    Returns
    -------
    ndarray
        Array of predictions with same shape as seqs, except on the last
        dimension, containing predictions for that sequence.
    """
    # Copy sequence array and make 2D with sequence along the 2nd axis
    seqs2D = seqs.reshape(-1, seqs.shape[-1]).copy()
    # Maybe reverse complement the sequences
    if reverse:
        seqs2D[seqs2D == 0] = -1
        seqs2D[seqs2D == 3] = 0
        seqs2D[seqs2D == -1] = 3
        seqs2D[seqs2D == 1] = -1
        seqs2D[seqs2D == 2] = 1
        seqs2D[seqs2D == -1] = 2
        seqs2D = np.flip(seqs2D, axis=-1)
    if stride != 1:
        # Do not predict all bases, but randomize which ones are predicted
        # each time
        start = np.random.randint(0, stride)
        print(f'Predicting with stride {stride} and offset {start}')
        # Allow higher batch_size
        batch_size *= stride
    else:
        start = 0
    # Get windows to predict on
    slides = utils.strided_window_view(
        seqs2D[:, start:],
        (len(seqs2D), winsize),
        (1, stride)).transpose([0, 2, 1, 3]).reshape(-1, winsize)
    # Initialise predictions list
    preds = []
    # Process slides by batch because it may use to much memory, batching is
    # done only along first dimension because all slides_per_seq must be
    # processed together
    for split in np.split(slides,
                          np.arange(batch_size, len(slides), batch_size)):
        # Convert to one-hot and predict
        pred = model.predict(
            one_hot_converter(split.reshape(-1, winsize))
        ).squeeze()
        # Collect garbage to prevent memory leak from model.predict()
        gc.collect()
        preds.append(pred)
    # Concatenate predictions and reshape as original sequences
    preds = np.concatenate(preds).reshape(seqs.shape[:-1] + (-1,))
    # Maybe reverse predictions
    if reverse:
        preds = np.flip(preds, axis=-1)
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


def select(energies, weights, temperature, step=None, maxoptions=1000):
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
    maxoptions : int, default=1000
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
    if maxoptions != -1:
        thres = np.partition(tot_energy, maxoptions, axis=-1)[:, maxoptions]
        tot_energy[tot_energy > np.expand_dims(thres, axis=-1)] = np.inf
    prob = utils.exp_normalize(-tot_energy / temperature)
    # Maybe save probabilities
    if step is not None:
        np.save(Path(args.output_dir, "probs", f'prob_step{step}.npy'), prob)
    # Select by the position of a random number in the cumulative sum
    cumprob = np.cumsum(prob, axis=-1)
    r = np.random.rand(len(prob)).reshape(-1, 1)
    sel_idx = np.argmax(r <= cumprob, axis=-1)
    # Associate energy to selected sequences
    sel_energies = np.stack(
        [en[np.arange(len(en)), sel_idx]
         for en in [tot_energy] + energies],
        axis=1)
    return sel_idx, sel_energies


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
    winsize = model.input_shape[1]

    # Tune predicting functions according to args

    def one_hot_converter(idx):
        return tf_idx_to_one_hot(
            idx, order=args.one_hot_order, extradims=args.extradims)

    if model.output_shape[1] == 1:

        def get_profile(seqs, reverse=False):
            return get_profile_mid1(
                seqs, model, winsize, reverse=reverse, stride=args.stride,
                batch_size=args.batch_size,
                one_hot_converter=one_hot_converter)
    else:
        head_interval = winsize // model.output_shape[1]
        assert head_interval % args.stride == 0

        def get_profile(seqs, reverse=False):
            return get_profile_hint(
                seqs, model, winsize, head_interval, reverse=reverse,
                middle=args.middle, stride=args.stride,
                batch_size=args.batch_size,
                one_hot_converter=one_hot_converter)
    # Generate and save start sequences
    if args.seed != -1:
        np.random.seed(args.seed)
    if args.start_seqs:
        seqs = np.load(args.start_seqs)
    else:
        freq_kmer = pd.read_csv(args.kmer_file,
                                index_col=[i for i in range(args.k)])
        seqs = utils.random_sequences(args.n_seqs, args.length,
                                      freq_kmer['W303'], out='idx')
        np.save(Path(args.output_dir, "designed_seqs", "start_seqs.npy"), seqs)
    # Initialize array of already seen bases for each position
    seen_bases = np.eye(4, dtype=int)[seqs]
    # Determine rows and cols to place n_seqs subplots
    ncol = 5  # max number of columns
    while args.n_seqs % ncol != 0:
        ncol -= 1
    nrow = args.n_seqs // ncol
    if args.n_seqs > 25:
        nrow, ncol = 5, 5  # plot max 25 sequences
    print(time.time() - t0)
    for step in range(args.steps):
        print(f'Step {step}')
        # Generate all mutations, and associated mutation energy
        seqs, mut_energy = all_mutations(seqs, seen_bases)
        # Predict on forward and reverse strands
        preds = get_profile(seqs)
        preds_rev = get_profile(seqs, reverse=True)
        # Compute energy components
        gc_energy = GC_energy(seqs, args.target_gc)
        for_energy = args.loss(args.target[:preds.shape[-1]], preds)
        rev_energy = args.loss(args.target[-preds_rev.shape[-1]:], preds_rev)
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
        offset = head_interval - args.stride
        if args.n_seqs == 1:
            pfor = np.repeat(preds[0, sel_idx].squeeze(), args.stride)
            prev = np.repeat(preds_rev[0, sel_idx].squeeze(), args.stride)
            axes.plot(pfor, label='forward')
            axes.plot(np.arange(offset, len(prev)+offset),
                      prev, label='reverse', alpha=0.8)
            axes.legend()
        else:
            for ax, pfor, prev in zip(axes.flatten(),
                                      preds[np.arange(len(seqs)), sel_idx],
                                      preds_rev[np.arange(len(seqs)), sel_idx]
                                      ):
                pfor = np.repeat(pfor, args.stride)
                prev = np.repeat(prev, args.stride)
                ax.plot(pfor, label='forward')
                ax.plot(np.arange(offset, len(prev)+offset),
                        prev, label='reverse', alpha=0.8)
                ax.legend()
        fig.savefig(Path(args.output_dir, "pred_figs",
                         f"mut_preds_step{step}.png"),
                    bbox_inches='tight')
        plt.close()
        print(time.time() - t0)


if __name__ == "__main__":
    tmstmp = datetime.datetime.now()
    print('Initialization')
    t0 = time.time()
    # Get arguments
    args = parsing()
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
                    if k not in ['target', 'weights']}
    with open(Path(args.output_dir, 'config.txt'), 'w') as f:
        json.dump(to_serealize, f, indent=4)
        f.write('\n')
        f.write(f'weights: {args.weights}\n')
        f.write(f'timestamp: {tmstmp}\n')
        f.write(f'machine: {socket.gethostname()}\n')
    # Convert to non json serializable objects
    losses = {'rmse': rmse}
    args.loss = losses[args.loss]
    args.weights = np.array(args.weights, dtype=float)
    # Save and plot target
    np.save(Path(args.output_dir, 'target.npy'),
            args.target)
    fig, ax = plt.subplots(1, 1, facecolor='w', layout='tight')
    ax.plot(args.target)
    fig.savefig(Path(args.output_dir, f"target.png"),
                bbox_inches='tight')
    plt.close()
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
