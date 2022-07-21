#!/usr/bin/env python

import numpy as np
import tensorflow as tf
from keras.engine import data_adapter
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.utils import Sequence
from tensorflow.keras import Model
from tensorflow.keras.metrics import binary_crossentropy
from tensorflow.python.eager import backprop
import os
from sklearn.preprocessing import OneHotEncoder
import time


class Eval_after_epoch(Callback):
    def __init__(self, model_dir, generator_eval, verbose=0):
        self.model_dir = model_dir
        self.generator_eval = generator_eval
        self.verbose = verbose
        self.preds = []
        self.epochs = 0

    def on_epoch_end(self, epoch, logs=None):
        pred = self.model.predict(
            self.generator_eval,
            verbose=self.verbose
        ).ravel()
        self.preds.append(pred)
        self.epochs += 1

    def on_train_end(self, logs=None):
        np.save(os.path.join(self.model_dir, 'eval_epochs'),
                np.reshape(self.preds, (self.epochs, -1)))


class ReweightingModel(Model):
    def __init__(self, *args, T=1, start_reweighting=2000, **kwargs):
        super().__init__(*args, **kwargs)
        self.T = T
        self.start_reweighting = start_reweighting

    @tf.function
    def maybe_reweight(self, x, y, sample_weight):
        if self._train_counter > self.start_reweighting:
            y_pred = self(x, training=False)
            sample_weight = self.reweight_positives(sample_weight, y, y_pred)
        return sample_weight

    @tf.function
    def reweight_positives(self, weights, y_true, y_pred):
        weights = tf.squeeze(weights)
        # compute loss
        loss = binary_crossentropy(y_true, y_pred)
        # compute new weights for each sample
        val = - loss / self.T
        max_val = tf.math.reduce_max(val)
        new_weights = tf.exp(val - max_val)
        # rescale positive weights to maintain total sum over batch
        mask_pos = (tf.squeeze(y_true) == 1)
        mask_pos.set_shape([None])
        old_sum_pos = tf.reduce_sum(weights[mask_pos])
        sum_pos = tf.reduce_sum(new_weights[mask_pos])
        coef_pos = old_sum_pos/sum_pos
        # change only positive weights
        new_weights = tf.where(mask_pos, new_weights*coef_pos, weights)
        return tf.expand_dims(new_weights, axis=1)

    def train_step(self, data):
        # These are the only transformations `Model.fit` applies to user-input
        # data when a `tf.data.Dataset` is provided.
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        # Reweight samples ####################################
        sample_weight = self.maybe_reweight(x, y, sample_weight)
        ########################################################

        # Run forward pass.
        with backprop.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(
                y, y_pred, sample_weight, regularization_losses=self.losses)
        # Run backwards pass.
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        # Collect metrics to return
        return_metrics = {}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result
        return return_metrics


class DataGenerator(Sequence):
    def __init__(self,
                 indexes,
                 data,
                 labels,
                 batch_size,
                 class_weights={0: 1, 1: 1},
                 sample_weights=None,
                 shuffle=True):
        self.dim = data[0].shape
        self.list_IDs = indexes
        self.batch_size = batch_size
        self.labels = labels
        if len(self.labels.shape) == 1:
            self.labels = np.expand_dims(self.labels, axis=1)
        self.data = data
        self.class_weights = class_weights
        self.sample_weights = sample_weights
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.list_IDs))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        start_batch = index*self.batch_size
        end_batch = min((index+1)*self.batch_size,
                        len(self.indexes))
        indexes = self.indexes[start_batch:end_batch]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, Y, weights = self.__data_generation(list_IDs_temp)
        return X, Y, weights

    def __data_generation(self, IDs):
        X = np.empty((len(IDs), *self.dim), dtype='float')
        Y = np.empty((len(IDs), 1), dtype='float')
        weights = np.empty((len(IDs), 1), dtype='float')
        for i, ID in enumerate(IDs):
            X[i, ] = self.data[ID]
            Y[i] = self.labels[ID]
            if self.sample_weights is None:
                weights[i] = self.class_weights[self.labels[ID, 0]]
            else:
                weights[i] = self.sample_weights[ID]
        # X = tf.convert_to_tensor(X, dtype=tf.float32)
        # Y = tf.convert_to_tensor(Y, dtype=tf.float32)
        # weights = tf.convert_to_tensor(weights, dtype=tf.float32)
        return X, Y, weights

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


def data_generation(IDs, reads, labels, class_weights):
    X = np.empty((len(IDs), *reads[0].shape), dtype='bool')
    Y = np.empty((len(IDs), 1), dtype='bool')
    weights = np.empty((len(IDs), 1), dtype='float')
    for i, ID in enumerate(IDs):
        X[i, ] = reads[ID]
        Y[i] = labels[ID]
        weights[i] = class_weights[labels[ID]]
    X = tf.convert_to_tensor(X, dtype=tf.float32)
    Y = tf.convert_to_tensor(Y, dtype=tf.float32)
    Y = tf.expand_dims(Y, axis=1)
    weights = tf.convert_to_tensor(weights, dtype=tf.float32)
    return X, Y, weights


def data_generator_from_files(files, batch_size, class_weights={0: 1, 1: 1},
                              shuffle=True):
    while True:
        for file in files:
            with np.load(file) as f:
                reads = f['x_train']
                labels = f['y_train']
            indexes = np.arange(len(labels))
            list_IDs = indexes
            n_batch = int(np.ceil(len(list_IDs) / batch_size))
            if shuffle:
                np.random.shuffle(indexes)

            for index in range(n_batch):
                start_batch = index*batch_size
                end_batch = min((index+1)*batch_size, len(indexes)-1)
                indexes_batch = indexes[start_batch:end_batch]
                list_IDs_batch = [list_IDs[k] for k in indexes]
                yield data_generation(list_IDs_batch, reads, labels,
                                      class_weights)


def create_weights(y):
    """
    Computes weights for negative and positive examples if they are unbalanced

    Arguments
    ---------
    y: array of labels to weight

    Returns
    -------
    weights: dictonary of class weights, 0 for neg, 1 for pos
    """
    n_pos = len(y[y == 1])
    n_neg = len(y[y == 0])
    pos_weight = 1/n_pos * (n_pos+n_neg)/2
    neg_weight = 1/n_neg * (n_pos+n_neg)/2
    return {0: neg_weight, 1: pos_weight}


def create_sample_weights(y):
    """
    Analog to `create_weights` returning an array of sample weights.

    Uses:
    create_weights
    """
    weights = create_weights(y)
    sample_weights = np.where(np.squeeze(y) == 1,
                              weights[1],
                              weights[0])
    return sample_weights


@tf.function
def change_sample_weights(loss, T=1):
    """
    Build weights from loss using softmax with temperature.
    """
    # exp normalize trick
    print('tracing c_s_w!')
    val = - loss / T
    max_val = tf.math.reduce_max(val)
    weights = tf.exp(val - max_val)
    weights *= tf.size(loss, out_type=weights.dtype) / tf.reduce_sum(weights)
    return weights


@tf.function
def balance_classes(weights, y):
    """
    Normalize array of weights to a mean 1.

    If an array of labels y is specified, balance each class
    """
    print('tracing b_c!')
    y = tf.squeeze(y)
    tot = tf.size(weights, out_type=weights.dtype)
    tot_pos = tf.reduce_sum(tf.where(y, weights, 0))
    tot_neg = tf.reduce_sum(tf.where(y, weights, 0))
    weights = tf.where(
        y,
        weights*tot / (2*tot_pos),
        weights*tot / (2*tot_neg)
    )
    return weights


def one_hot_encoding(array, read_length=101, one_hot_type=bool):
    return one_hot_encoding_v1(array,
                               read_length=read_length,
                               one_hot_type=one_hot_type)


def one_hot_encoding_v1(array, read_length=101, one_hot_type=bool):
    """
    Applies one hot encoding to every read sequence in an array.

    If a given read sequence is too short, it is filled with 0 which represent
    N values. If it is too long, the read is truncated to read_length

    Arguments
    ---------
    array: numpy array containing n reads sequences of same length l

    Returns
    -------
    new_array: numpy array with every letter replaced by a 4 dimensional
        vector containing a 1 in the position corresponding to that letter,
        and 0 elsewhere. Output shape is (n,l,4)
    """
    # warning raise in case sequences don't have the appropriate read_length
    new_array = np.zeros((len(array), read_length, 4), dtype=one_hot_type)
    unmatched_lengths = 0
    for i, seq in enumerate(array):
        if len(seq) != read_length:
            unmatched_lengths += 1
        for j in range(min(len(seq), read_length)):
            if seq[j] == 'A':
                new_array[i, j, 0] = 1
            elif seq[j] == 'C':
                new_array[i, j, 1] = 1
            elif seq[j] == 'G':
                new_array[i, j, 2] = 1
            elif seq[j] == 'T':
                new_array[i, j, 3] = 1
    if unmatched_lengths != 0:
        print(f"Warning: {unmatched_lengths} sequences don't have the "
              "appropriate read length")
    return new_array


def one_hot_encoding_v2(reads,
                        read_length=101,
                        one_hot_type=bool,
                        sparse=False):
    """
    Applies one hot encoding to every read sequence in an array.

    If a given read sequence is too short, it is filled with 0 which represent
    N values. If it is too long, the read is truncated to read_length.
    This implementation uses scikit-learn's OneHotEncoder

    Arguments
    ---------
    reads: numpy array containing n reads sequences of same length l

    Returns
    -------
    new_array: numpy array with every letter replaced by a 4 dimensional
        vector containing a 1 in the position corresponding to that letter,
        and 0 elsewhere. Output shape is (n,l,4)
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


def write_fasta(seqs, fasta_file, wrap=80):
    # from https://www.programcreek.com/python/?code=Ecogenomics%2FGTDBTk%
    # 2FGTDBTk-master%2Fgtdbtk%2Fbiolib_lite%2Fseq_io.py
    """Write sequences to a fasta file.

    Parameters
    ----------
    seqs : dict[seq_id] -> seq
        Sequences indexed by sequence id.
    fasta_file : str
        Path to write the sequences to.
    wrap: int
        Number of AA/NT before the line is wrapped.
    """
    with open(fasta_file, 'w') as f:
        for id, seq in enumerate(seqs):
            f.write('>{}\n'.format(id))
            for i in range(0, len(seq), wrap):
                f.write('{}\n'.format(seq[i:i + wrap]))


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
    """
    Return all unique reads and occurences.
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


def ram_usage():
    # https://www.geeksforgeeks.org/how-to-get-current-cpu-and-ram-usage-in-python/
    # Getting all memory using os.popen()
    total_memory, used_memory, free_memory = map(
        int, os.popen('free -t -m').readlines()[-1].split()[1:])
    # Memory usage
    print("RAM memory % used:", round((used_memory/total_memory) * 100, 2))


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
            if (read_length is not None and len(seq) != read_length):
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


def GC_content(reads):
    assert(len(reads.shape) == 3)
    content = np.sum(reads, axis=1)
    gc = (content[:, 1] + content[:, 2]) / np.sum(content, axis=1)
    return gc


def classify_1D(features, y, bins):
    bins = np.histogram(features, bins=bins, range=(0, 1))[1]
    gc_pos = features[y == 1]
    gc_neg = features[y == 0]
    pos_bins = np.digitize(gc_pos, bins).ravel()
    pos_count = np.bincount(pos_bins, minlength=len(bins)+1)[1:]
    cumul_pos_count = np.cumsum(pos_count)
    neg_bins = np.digitize(gc_neg, bins).ravel()
    neg_count = np.bincount(neg_bins, minlength=len(bins)+1)[1:]
    cumul_neg_count = np.cumsum(neg_count)
    cumul_diff = cumul_pos_count - cumul_neg_count
    bin_thres = np.argmax(np.abs(cumul_diff))
    if cumul_diff[bin_thres] < 0:
        accuracy = (len(gc_pos) - cumul_diff[bin_thres]) / len(features)
    else:
        accuracy = (len(gc_neg) + cumul_diff[bin_thres]) / len(features)
    assert(bin_thres != len(bins) - 1)
    thres = (bins[bin_thres] + bins[bin_thres+1]) / 2
    return accuracy, thres


def chunck_into_reads(long_reads, read_length=101):
    reads = []
    for i, long in enumerate(long_reads):
        chuncks = [long[i:i+read_length]
                   for i in range(0, len(long), read_length)]
        reads.extend(chuncks)
    return reads

# # sparse_one_hot encoding
# categories = np.array([['A'], ['C'], ['G'], ['T']])
# encoder = OneHotEncoder(dtype=bool, handle_unknown='ignore')
# encoder.fit(categories)


# def sparse_one_hot_encoding(array, read_length=read_length,
#                             one_hot_type=one_hot_type, encoder=encoder):
#     n_seq = 0
#     for seq in array:
#         seq_list = np.reshape(list(seq), (read_length, 1))
#         res = encoder.transform(seq_list)
#         cols = res.nonzero()[0]
#         letter = res.indices
#         if n_seq != 0:
#             rows = np.ones(len(cols), dtype=int) * n_seq
#             indices = np.concatenate(
#                 (indices, np.stack((rows, cols, letter), axis=1))
#             )
#         else:
#             rows = np.zeros(len(cols), dtype=int)
#             indices = np.stack((rows, cols, letter), axis=1)
#         n_seq += 1
#     return indices
