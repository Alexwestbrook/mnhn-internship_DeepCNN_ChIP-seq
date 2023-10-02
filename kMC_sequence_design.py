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
        "--steps", type=int, default=100,
        help="number of steps to perform")
    parser.add_argument(
        "--stride", type=int, default=1,
        help="specifies a stride in predictions to go faster")
    parser.add_argument(
        "-mid", "--middle_pred", action='store_true',
        help="specifies to predict only on middle window")
    parser.add_argument(
        "-targ", "--target", type=float, nargs='+', default=[],
        help="target profile for the designed sequences")
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
    if args.target == []:
        args.target = [0] * args.length
    return args


def GC_energy(seqs, target_gc=0.3834):
    return np.abs(np.sum((seqs == 1) | (seqs == 2), axis=-1) / seqs.shape[-1]
                  - target_gc)


def all_mutations(seqs, seen_bases):
    """mute 1 bp per line in a circular rotation 0 -> 1 -> 2 -> 3 -> 0"""
    n_seqs, length = seqs.shape
    mutated = np.tile(seqs, (1, 3*length)).reshape(-1, length)
    for i in range(3):
        mutated[np.arange(i, len(mutated), 3),
                np.repeat(np.arange(length), n_seqs)] += i + 1
    mutated = mutated.reshape(n_seqs, 3*length, length) % 4
    mut_energy = seen_bases[~np.eye(4, dtype=bool)[seqs]
                            ].reshape(n_seqs, 3*length)
    return mutated, mut_energy


def tf_idx_to_one_hot(idx, order='ACGT'):
    converter = np.zeros((4, 4))
    for i, c in enumerate('ACGT'):
        converter[i, order.find(c)] = 1
    return tf.convert_to_tensor(converter[idx], dtype=tf.float32)


def get_profile(seqs, model, winsize, head_interval, reverse=False,
                middle=True, batch_size=1000, one_hot_order='ACGT',
                stride=1):
    seqs = seqs.copy()
    n_heads = winsize // head_interval
    if reverse:
        seqs[seqs == 0] = -1
        seqs[seqs == 3] = 0
        seqs[seqs == -1] = 3
        seqs[seqs == 1] = -1
        seqs[seqs == 2] = 1
        seqs[seqs == -1] = 2
        seqs = np.flip(seqs, axis=-1)
    if middle:
        strides = winsize // 2
        out_heads = n_heads // 2
    else:
        strides = winsize
        out_heads = n_heads
    if stride != 1:
        # Do not predict all bases, but randomize which ones are predicted
        # each time
        start = np.random.randint(0, stride)
        stop = start - stride + 1
        if stop == 0:
            stop = seqs.shape[-1]
        slides_per_seq = head_interval // stride
        print(f'Predicting with stride {stride} and offset {start}')
        slides = utils.strided_sliding_window_view(
            seqs[:, :, start:stop],
            seqs.shape[:-1] + (winsize,),
            stride=(strides, stride),
            sliding_len=head_interval,
            axis=-1).reshape(-1, slides_per_seq, winsize)
        batch_size *= stride
    else:
        slides = utils.strided_sliding_window_view(
            seqs,
            seqs.shape[:-1] + (winsize,),
            stride=strides,
            sliding_len=head_interval,
            axis=-1).reshape(-1, head_interval, winsize)
        slides_per_seq = head_interval
    preds = []
    for split in np.split(slides,
                          np.arange(batch_size, len(slides), batch_size)):
        pred = model.predict(
            tf_idx_to_one_hot(split.reshape(-1, winsize), order=one_hot_order)
        ).squeeze()
        gc.collect()
        if middle:
            pred = pred[:, n_heads//4:3*n_heads//4]
        pred = np.transpose(pred.reshape(-1, slides_per_seq, out_heads),
                            [0, 2, 1])
        pred = pred.reshape(-1, slides_per_seq*out_heads)
        preds.append(pred)
    preds = np.concatenate(preds).reshape(seqs.shape[:-1] + (-1,))
    if reverse:
        preds = np.flip(preds, axis=-1)
    return preds


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2, axis=-1))


def select(energies, weights, temperature):
    tot_energy = sum(w*e for w, e in zip(weights, energies))
    prob = np.exp(-tot_energy / temperature)
    prob /= np.sum(prob, axis=-1, keepdims=True)
    cumprob = np.cumsum(prob, axis=-1)
    r = np.random.rand(len(prob)).reshape(-1, 1)
    sel_idx = np.argmax(r <= cumprob, axis=-1)
    sel_energies = np.stack(
        [en[np.arange(len(en)), sel_idx]
         for en in [tot_energy] + energies],
        axis=1)
    return sel_idx, sel_energies


if __name__ == "__main__":
    tmstmp = datetime.datetime.now()
    print('Initialization')
    t0 = time.time()
    # Get arguments
    args = parsing()
    # Build output directory and initialize energy file
    Path(args.output_dir).mkdir(parents=True, exist_ok=False)
    with open(Path(args.output_dir, 'energy.txt'), 'w') as f:
        f.write('total_energy\t'
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
        f.write(f'target: {args.target}\n')
        f.write(f'weights: {args.weights}\n')
        f.write(f'timestamp: {tmstmp}\n')
        f.write(f'machine: {socket.gethostname()}\n')
    # Convert to non json serializable objects
    args.target = np.array(args.target, dtype=float)
    losses = {'rmse': rmse}
    args.loss = losses[args.loss]
    args.weights = np.array(args.weights, dtype=float)
    # Load model
    # with tf.distribute.MirroredStrategy().scope():
    model = tf.keras.models.load_model(
        args.model_file,
        custom_objects={'correlate': correlate, 'mae_cor': mae_cor})
    winsize = model.input_shape[1]
    head_interval = winsize // model.output_shape[1]
    assert head_interval % args.stride == 0
    # Generate and save start sequences
    if args.seed != -1:
        np.random.seed(args.seed)
    freq_kmer = pd.read_csv(args.kmer_file,
                            index_col=[i for i in range(args.k)])
    seqs = utils.random_sequences(args.n_seqs, args.length, freq_kmer['W303'],
                                  out='idx')
    np.save(Path(args.output_dir, "start_seqs.npy"), seqs)
    # Initialize array of already seen bases for each position
    seen_bases = np.eye(4, dtype=int)[seqs]
    # Determine rows and cols to place n_seqs subplots
    ncol = 5  # max number of columns
    while args.n_seqs % ncol != 0:
        ncol -= 1
    nrow = args.n_seqs // ncol
    print(time.time() - t0)
    for i in range(args.steps):
        print(f'Step {i}')
        # Generate all mutations, and associated mutation energy
        seqs, mut_energy = all_mutations(seqs, seen_bases)
        # Predict on forward and reverse strands
        preds = get_profile(
            seqs, model, winsize, head_interval, batch_size=args.batch_size,
            stride=args.stride, one_hot_order=args.one_hot_order,
            middle=args.middle_pred)
        preds_rev = get_profile(
            seqs, model, winsize, head_interval, batch_size=args.batch_size,
            stride=args.stride, one_hot_order=args.one_hot_order,
            middle=args.middle_pred, reverse=True)
        # Compute energy
        gc_energy = GC_energy(seqs)
        for_energy = args.loss(args.target[:preds.shape[-1]], preds)
        rev_energy = args.loss(args.target[-preds_rev.shape[-1]:], preds_rev)
        energy_list = [gc_energy, for_energy, rev_energy, mut_energy]
        # Choose best mutation by kMC method
        sel_idx, sel_energies = select(energy_list, args.weights,
                                       args.temperature)
        # Keep selected sequence and increment seen_bases
        seqs = seqs[np.arange(len(seqs)), sel_idx]
        seen_bases[np.arange(len(seqs)),
                   sel_idx // 3,
                   seqs[np.arange(len(seqs)), sel_idx // 3]] += 1
        # Save sequence, energy and plot profile
        np.save(Path(args.output_dir, f"mut_seqs_step{i}.npy"), seqs)
        with open(Path(args.output_dir, 'energy.txt'), 'a') as f:
            f.write(f'{sel_energies}\n')
        fig, axes = plt.subplots(nrow, ncol, figsize=(10, 3),
                                 facecolor='w', layout='tight', sharey=True)
        if args.n_seqs == 1:
            pfor = preds[0, sel_idx].squeeze()
            prev = preds_rev[0, sel_idx].squeeze()
            axes.plot(pfor, label='forward')
            axes.plot(np.arange(head_interval, len(prev)+head_interval),
                      prev, label='reverse', alpha=0.8)
            axes.legend()
        else:
            for ax, pfor, prev in zip(axes.flatten(),
                                      preds[np.arange(len(seqs)), sel_idx],
                                      preds_rev[np.arange(len(seqs)), sel_idx]
                                      ):
                ax.plot(pfor, label='forward')
                ax.plot(np.arange(head_interval, len(prev)+head_interval),
                        prev, label='reverse', alpha=0.8)
                ax.legend()
        fig.savefig(Path(args.output_dir, f"mut_preds_step{i}.png"),
                    bbox_inches='tight')
        plt.close()
        print(time.time() - t0)
    with open(Path(args.output_dir, 'config.txt'), 'a') as f:
        f.write(f'total time: {time.time() - t0}\n')
