#!/usr/bin/env python
import os
from pathlib import Path
from collections import defaultdict
from typing import Optional, Union

import numpy as np
from numpy.lib.stride_tricks import as_strided
from numpy.core.numeric import normalize_axis_tuple

from sklearn.preprocessing import OneHotEncoder
from scipy.signal import gaussian, convolve

import tensorflow as tf
from keras.engine import data_adapter
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.utils import Sequence
from tensorflow.keras import Model
from tensorflow.keras.metrics import binary_crossentropy
from tensorflow.python.eager import backprop

import pyBigWig


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


# Generators
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


class DataGeneratorFromFiles(Sequence):
    """
    Build callable generator for Tensorflow model from multiple files
    """
    def __init__(self,
                 dataset_dir: Path,
                 batch_size: int,
                 split: str = 'train',
                 use_labels: bool = False,
                 class_weights: dict = {0: 1, 1: 1},
                 use_sample_weights: np.ndarray = False,
                 shuffle: bool = True,
                 file_suffix: str = None):
        """
        Build callable generator for Tensorflow model from multiple files

        Parameters
        ----------
        dataset_dir : Path
            Path to dataset directory, containing files in numpy binary format
            npy or npz. Filenames must be '[split]_[index].[npy or npz]'. For
            example, the first train file (in npy format) must be named
            'train_0.npy'.
            The directory must not contain unrelated files or globbing might
            include more files in the dataset.
            If files are provided in npz format, they will be copied in npy
            format for faster access during training and prediction. The copy
            will have same name as the original but with the .npy extension
            instead of .npz. It will raise an Exception if the file already
            exists in npy format.
        batch_size : int
            Number of samples per batch.
        split : str, default='train'
            Name of the split to generate, must match the dataset filenames
        use_labels: bool, default=False
            If False, the even index samples are assumed of label 1 and odd
            index samples of label 0. If True, indicates to use labels in
            label files.
            The files must be in the same directory as their data files and
            with the name labels_[split]_[index].npy. The labels must be an
            array of same shape as the data.
        class_weight: dict[label] -> float, default={0: 1, 1: 1}
            Dictionary mapping labels to class weights values.
        use_sample_weight: bool, default=False
            If True, overides class_weight argument and indicates to use weight
            files. The sample_weight files must be in the same directory as
            their data files and with the name weights_[split]_[index].npy.
            The weights must be an array of same shape as the data.
        shuffle: bool, default=True
            If True, indicates to shuffle file order and content before each
            iteration, recommended for training. During prediction, set to
            False for easy matching between data and prediction.
        file_suffix: {'npy', 'npz', 'None'}, default=None
            Indicates the suffix of the files to use in the dataset. If None,
            the suffix is inferred from the presence of a [split]_0.npy file.
            If not, the file_suffix is assumed to be npz. Due to conversion of
            npz into npy, an Error will be raised in case of conflicting names.
        """
        self.batch_size = batch_size
        self.use_labels = use_labels
        self.class_weights = class_weights
        self.use_sample_weights = use_sample_weights
        # Initializers for file parsing
        if use_labels:
            self.label_files = []
        if use_sample_weights:
            self.weight_files = []
        self.data_files = []
        self.files_idx = []
        self.contents_idx = []
        self.dim = None
        # Check if first .npy file exists, if not assume data is in .npz
        if file_suffix is None:
            if Path(dataset_dir, split + '_0.npy').exists():
                file_suffix = '.npy'
            else:
                file_suffix = '.npz'
        for file in Path(dataset_dir).glob(split + '_*' + file_suffix):
            # Get file information, if npz, make a npy copy for faster reading
            if file_suffix == '.npy':
                data = np.load(file)
                self.data_files.append(file)
            elif file_suffix == '.npz':
                with np.load(file) as f:
                    data = f['one_hots']
                new_file = Path(file.parent, file.stem + '.npy')
                if new_file.exists():
                    raise FileExistsError(
                        f"{file} could not be converted to npy because "
                        f"{new_file} already exists"
                    )
                np.save(new_file, data)
                self.data_files.append(new_file)
            # File content indices, used for for shuffling file contents
            self.contents_idx.append(np.arange(len(data)))
            # Get label files
            if use_labels:
                label_file = Path(file.parent, 'labels_' + file.stem + '.npy')
                if not label_file.exists():
                    raise FileNotFoundError(f"No label file found for {file}")
                self.label_files.append(label_file)
            # Get weight files
            if use_sample_weights:
                weight_file = Path(file.parent, 'weights_'+file.stem+'.npy')
                if not weight_file.exists():
                    raise FileNotFoundError(f"No weight file found for {file}")
                self.weight_files.append(weight_file)
            # Get shape of a data sample
            if self.dim is None:
                self.dim = data[0].shape
        # Free space from data
        del data

        if self.dim is None:
            raise FileNotFoundError(
                f"No {split} files found in {dataset_dir}. Files must be in "
                "numpy binary format, named as [split]_[number].[npy or npz]")
        # File indices and number of samples, used for shuffling file order
        self.files_idx = np.vstack((
            np.arange(len(self.data_files)),
            [len(content) for content in self.contents_idx]
        )).T
        # File seperators for finding which file an index falls into
        self.file_seperators = np.zeros(len(self.files_idx) + 1,
                                        dtype=np.int64)
        self.file_seperators[1:] = np.cumsum(self.files_idx[:, 1])
        # Shuffle handling
        self.shuffle = shuffle
        if shuffle:
            self.rng = np.random.default_rng()
            self.on_epoch_end()

    def __len__(self):
        total_samples = self.file_seperators[-1]
        return int(np.ceil(total_samples / self.batch_size))

    def __getitem__(self, index):
        # Initialize batch
        print(f"getitem {index}")
        print("file order")
        print(self.files_idx)
        X = np.empty((self.batch_size, *self.dim), dtype='float')
        Y = np.empty((self.batch_size, 1), dtype='float')
        weights = np.empty((self.batch_size, 1), dtype='float')
        # Keep track of next index to fill
        filled_idx = 0
        # Fill batch sequentially from all files required to be opened
        # Hopefully most cases require only one file
        for file_idx, start, stop in self._file_access(index):
            if start == 0:
                # Open a new data file
                self.cur_data = np.load(self.data_files[file_idx])
                # Get labels
                if self.use_labels:
                    self.cur_labels = np.load(self.label_files[file_idx])
                else:
                    self.cur_labels = np.zeros(len(self.cur_data), dtype=bool)
                    self.cur_labels[::2] = 1
                # Get weights
                if self.use_sample_weights:
                    self.cur_weights = np.load(self.weight_files[file_idx])
                else:
                    self.cur_weights = np.empty(len(self.cur_data), dtype=bool)
                    for label, weight in self.class_weights.items():
                        self.cur_weights[self.cur_labels == label] = weight
            content_indices = self.contents_idx[file_idx][start:stop]
            to_fill = slice(filled_idx, filled_idx + stop - start)
            print(to_fill)
            print(X.shape)
            print(X[to_fill].shape)
            X[to_fill] = self.cur_data[content_indices]
            Y[to_fill, 0] = self.cur_labels[content_indices]
            weights[to_fill, 0] = self.cur_weights[content_indices]
            filled_idx += stop - start
        return X[:filled_idx], Y[:filled_idx], weights[:filled_idx]

    def _file_access(self, index):
        """Determine which file or files to get next batch from"""
        offset = index * self.batch_size - self.file_seperators
        start_idx = argmax_last(offset >= 0)
        end_idx = min(argmax_last(offset > - self.batch_size),
                      len(self.files_idx) - 1)
        print("start file", start_idx, "| end file", end_idx)
        starts = np.maximum(offset, 0)
        for idx in range(start_idx, end_idx + 1):
            file_idx, file_len = self.files_idx[idx]
            start = starts[idx]
            if idx == end_idx:
                stop = min(offset[idx] + self.batch_size, file_len)
            else:
                stop = file_len
            yield file_idx, start, stop

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.files_idx)
            print(self.files_idx)
            for file_idx in self.files_idx[:, 0]:
                print(file_idx)
                self.rng.shuffle(self.contents_idx[file_idx])
            self.file_seperators[1:] = np.cumsum(self.files_idx[:, 1])


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
                     one_hot_type: type = bool) -> np.ndarray:
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
                               one_hot_type=one_hot_type)


def one_hot_encoding_v1(array: np.ndarray,
                        read_length: int = 101,
                        one_hot_type: type = bool) -> np.ndarray:
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
            if base == 'A':
                new_array[i, j, 0] = 1
            elif base == 'C':
                new_array[i, j, 1] = 1
            elif base == 'G':
                new_array[i, j, 2] = 1
            elif base == 'T':
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
            chr_id = chr_id[3:]
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
        meaning [1 2] and [4 5] actually contain a gap of one base, even
        but the distance is 2 (4-2). The dafault value of 1 means that no
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


# Other utils
def s_plural(value: float) -> str:
    """Return s if scalar value induces plural"""
    if value > 1:
        return 's'
    else:
        return ''


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
