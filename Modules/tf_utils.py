#!/usr/bin/env python
from pathlib import Path
import tempfile
import numpy as np

import tensorflow as tf
from keras.engine import data_adapter
import keras.backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.utils import Sequence
from tensorflow.keras import Model
from tensorflow.keras.metrics import binary_crossentropy, mse
from tensorflow.python.eager import backprop

import Modules.utils as utils


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
        np.save(Path(self.model_dir, 'eval_epochs'),
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


class ConfidenceLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        return tf.reduce_mean(
            binary_crossentropy(y_true, y_pred[:, :1])
            + mse(tf.abs(y_true - y_pred[:, :1]), y_pred[:, 1:]),
            axis=-1)


class DummyLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        return tf.reduce_mean(
            binary_crossentropy(y_true, y_pred[:, :1]),
            axis=-1)


def confidence_loss(y_true, y_pred):
    return tf.reduce_mean(
        binary_crossentropy(y_true, y_pred[:, 0])
        + mse(y_true - y_pred[:, 0], y_pred[:, 1]))


def mae_cor(y_true, y_pred):
    """Compute loss with Mean absolute error and correlation.
        :Example:
        >>> model.compile(optimizer = 'adam', losses = mae_cor)
        >>> load_model('file', custom_objects = {'mae_cor : mae_cor})
    """
    X = y_true - K.mean(y_true)
    Y = y_pred - K.mean(y_pred)

    sigma_XY = K.sum(X*Y)
    sigma_X = K.sqrt(K.sum(X*X))
    sigma_Y = K.sqrt(K.sum(Y*Y))

    cor = sigma_XY/(sigma_X*sigma_Y + K.epsilon())
    mae = K.mean(K.abs(y_true - y_pred))

    return (1 - cor) + mae


def correlate(y_true, y_pred):
    """Calculate the correlation between the predictions and the labels.
        :Example:
        >>> model.compile(optimizer = 'adam', losses = correlate)
        >>> load_model('file', custom_objects = {'correlate : correlate})
    """
    X = y_true - K.mean(y_true)
    Y = y_pred - K.mean(y_pred)

    sigma_XY = K.sum(X*Y)
    sigma_X = K.sqrt(K.sum(X*X))
    sigma_Y = K.sqrt(K.sum(Y*Y))

    return sigma_XY/(sigma_X*sigma_Y + K.epsilon())


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
    def __init__(self,
                 dataset_dir: Path,
                 batch_size: int,
                 temp_dir: Path = None,
                 split: str = 'train',
                 use_labels: bool = False,
                 class_weights: dict = {0: 1, 1: 1},
                 use_sample_weights: np.ndarray = False,
                 shuffle: bool = True,
                 file_suffix: str = None):
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
        self.temp_dir = temp_dir
        # Check if first .npy file exists, if not assume data is in .npz
        if file_suffix is None:
            if Path(dataset_dir, split + '_0.npy').exists():
                file_suffix = '.npy'
            else:
                file_suffix = '.npz'
                if self.temp_dir is None:
                    self.temp_dir = tempfile.TemporaryDirectory()
        for file in Path(dataset_dir).glob(split + '_*' + file_suffix):
            # Get file information, if npz, make a npy copy for faster reading
            if file_suffix == '.npy':
                data = np.load(file)
                self.data_files.append(file)
            elif file_suffix == '.npz':
                with np.load(file) as f:
                    data = f['one_hots']
                new_file = Path(self.temp_dir.name, file.stem + '.npy')
                if new_file.exists():
                    raise FileExistsError(
                        f"{file} could not be converted to npy because "
                        f"{new_file} already exists"
                    )
                np.save(new_file, data)
                self.data_files.append(new_file)
            # File content indices, used for shuffling file contents
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
        # print(f"getitem {index}")
        # print("file order")
        # print(self.files_idx)
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
            # print(to_fill)
            # print(X.shape)
            # print(X[to_fill].shape)
            X[to_fill] = self.cur_data[content_indices]
            Y[to_fill, 0] = self.cur_labels[content_indices]
            weights[to_fill, 0] = self.cur_weights[content_indices]
            filled_idx += stop - start
        return X[:filled_idx], Y[:filled_idx], weights[:filled_idx]

    def _file_access(self, index):
        """Determine which file or files to get next batch from"""
        offset = index * self.batch_size - self.file_seperators
        start_idx = utils.argmax_last(offset >= 0)
        end_idx = min(utils.argmax_last(offset > - self.batch_size),
                      len(self.files_idx) - 1)
        # print("start file", start_idx, "| end file", end_idx)
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


class WindowGenerator(Sequence):
    """
    Build a Generator for training Tensorflow model from chromosome windows.

    Parameters
    ----------
    data : ndarray, shape=(n, 4)
        2D-array of one-hot encoded chromosome.
    labels : ndarray, shape=(n,)
        array of labels for each base of the chromosome.
    winsize : int
        length of windows to send as input to the model
    batch_size : int
        number of windows to send per batch
    max_data : int
        maximum number of windows per epoch (before evaluating on the
        validation set). If there are more windows, they will be used in a
        further epoch. There may be multiple epochs without reusing data.
    shuffle : bool, default=True
        If True, indicates to shuffle windows before training and once all
        windows have been used (not necessarily after each epoch).
    same_samples : bool, default=False
        If True, indicates to use the same sample at each epoch. Otherwise the
        sample is changed at each epoch to use all available data.
    balance : {None, "batch", "global"}, default=None
        "batch" indicates to balance weights among classes inside each batch,
        while "global" indicates to balance on the entire data. Default value
        None indicates not to balance weights.
    n_classes : int, default=500
        If `balance` is set, indicates number of bins to divide the signal
        range in, for determining classes.
    strand : {'for', 'rev', 'both'}, default='both'
        Indicates which strand to use for training. If 'both', half the
        windows of each batch are reversed.

    Attributes
    ----------
    data : ndarray, shape=(n, 4)
        same as in Parameters
    labels : ndarray, shape=(n, 1)
        same as in Parameters, but with dimension expanded
    winsize : int
        same as in Parameters
    batch_size : int
        same as in Parameters
    max_data : int
        same as in Parameters. But `max_data` is set to the number of windows
        if `max_data` is larger.
    shuffle : bool
        same as in Parameters
    same_samples : bool, default=False
        same as in Parameters
    balance : {None, "batch", "global"}, default=None
        same as in Parameters
    n_classes : int, default=500
        same as in Parameters
    strand : {'for', 'rev', 'both'}, default='for'
        same as in Parameters
    indexes : ndarray
        1D-array of valid window indexes for training. Valid windows exclude
        windows with Ns or null labels. Indexescorrespond to the center of the
        window in `data`.
    weights : ndarray
        global weights to use when `balance` is set to 'global'.
    start_idx : int
        Index of the starting window to use for next epoch when `same_samples`
        is set to False. It is used and updated in the method `on_epoch_end`.
    """
    def __init__(self,
                 data,
                 labels,
                 winsize,
                 batch_size,
                 max_data,
                 shuffle=True,
                 same_samples=False,
                 balance=None,
                 n_classes=500,
                 strand='both'):
        self.data = data
        self.labels = labels
        self.winsize = winsize
        self.batch_size = batch_size
        self.max_data = max_data
        self.shuffle = shuffle
        self.same_samples = same_samples
        self.balance = balance
        self.n_classes = n_classes
        self.strand = strand
        if len(self.labels.shape) == 1:
            self.labels = np.expand_dims(self.labels, axis=1)
        # Select indices of windows without Ns or null labels
        N_mask = (np.sum(data, axis=1) == 0)
        N_window_mask = np.asarray(
            np.convolve(N_mask, np.ones(self.winsize), mode='same'),
            dtype=int)
        valid_window_mask = (N_window_mask == 0) & (labels != 0).ravel()
        self.indexes = np.arange(len(data))[valid_window_mask]
        # Remove indices of edge windows
        self.indexes = self.indexes[
            (self.indexes >= self.winsize // 2)
            & (self.indexes < len(data) - (self.winsize // 2))]
        # Set max_data to only take less than all the indexes
        if max_data > len(self.indexes):
            self.max_data = len(self.indexes)
        if self.shuffle:
            np.random.shuffle(self.indexes)
        # Build first sample
        self.sample = self.indexes[0:self.max_data]
        if not self.same_samples:
            self.start_idx = self.max_data
        if self.balance == "global":
            # Compute effective labels that will be used for training
            if self.same_samples:
                y_eff = self.labels[self.sample]
            else:
                y_eff = self.labels[self.indexes]
            # Determine weights with effective labels
            bin_values, bin_edges = np.histogram(
                y_eff, bins=self.n_classes, range=(0, 1))
            # Weight all labels for convinience
            bin_idx = np.digitize(self.labels, bin_edges)
            bin_idx[bin_idx == self.n_classes+1] = self.n_classes
            bin_idx -= 1
            self.weights = len(y_eff) / (
                self.n_classes * bin_values[bin_idx])

    def __len__(self):
        """Return length of generator.

        The length displayed is the length for the current epoch. Not the
        number of available windows accross multiple epochs.
        """
        return int(np.ceil(len(self.sample) / self.batch_size))

    def __getitem__(self, idx):
        """Get a batch of data.

        Parameters
        ----------
        idx : int
            Index of the batch to extract
        """
        # Get window center idxes
        batch_idxes = self.sample[idx*self.batch_size:(idx+1)*self.batch_size]
        # Get full window idxes
        window_indices = (
            batch_idxes.reshape(-1, 1)
            + np.arange(-(self.winsize//2), self.winsize//2 + 1).reshape(1, -1)
        )
        batch_x = self.data[window_indices]
        if self.strand == 'rev':
            batch_x = batch_x[:, ::-1, ::-1]
        elif self.strand == 'both':
            half_size = self.batch_size // 2
            batch_x[:half_size] = batch_x[:half_size, ::-1, ::-1]
        batch_y = self.labels[batch_idxes]
        # Divide continuous labels into classes and balance weights
        if self.balance == 'batch':
            bin_values, bin_edges = np.histogram(
                batch_y, bins=self.n_classes, range=(0, 1))
            bin_idx = np.digitize(batch_y, bin_edges)
            bin_idx[bin_idx == self.n_classes+1] = self.n_classes
            bin_idx -= 1
            batch_weights = self.batch_size / (
                self.n_classes * bin_values[bin_idx])
        elif self.balance == 'global':
            batch_weights = self.weights[batch_idxes]
        else:
            return batch_x, batch_y
        return batch_x, batch_y, batch_weights

    def on_epoch_end(self):
        """Update the sample to use for next epoch.

        The sample can be different from the one used in the previous epoch if
        there are enough windows. If all windows have been seen, a shuffle may
        be applied and additional windows are extracted from the start.
        """
        if self.same_samples:
            if self.shuffle:
                np.random.shuffle(self.sample)
        else:
            stop_idx = self.start_idx + self.max_data
            self.sample = self.indexes[self.start_idx:stop_idx]
            if stop_idx >= len(self.indexes):
                # Complete sample by going back to the beginning of indexes
                if self.shuffle:
                    # Save current sample because shuffling will modify it
                    self.sample = self.sample.copy()
                    np.random.shuffle(self.indexes)
                stop_idx = stop_idx - len(self.indexes)
                if stop_idx != 0:
                    self.sample = np.concatenate(
                        (self.sample, self.indexes[:stop_idx]))
            # Update start_idx for next call to on_epoch_end
            self.start_idx = stop_idx


class PredGenerator(Sequence):
    def __init__(self,
                 data,
                 winsize,
                 batch_size):
        self.data = data
        self.winsize = winsize
        self.batch_size = batch_size
        self.indexes = np.arange(self.winsize // 2,
                                 len(data) - (self.winsize // 2))

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, idx):
        # Get window center idxes
        batch_idxes = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
        # Get full window idxes
        window_indices = (
            batch_idxes.reshape(-1, 1)
            + np.arange(-(self.winsize//2), self.winsize//2 + 1).reshape(1, -1)
        )
        batch_x = self.data[window_indices]
        batch_y = np.zeros((len(batch_x), 1))
        return batch_x, batch_y


def predict(model, one_hot_chr, winsize, reverse=False, batch_size=1024):
    if reverse:
        one_hot_chr = one_hot_chr[::-1, ::-1]
    X = PredGenerator(one_hot_chr, winsize, batch_size)
    pred = np.zeros(len(one_hot_chr), dtype='float32')
    pred[winsize//2:-(winsize//2)] = model.predict(X).ravel()
    if reverse:
        return pred[::-1]
    else:
        return pred


def predict_v1(model, one_hot_chr, winsize, reverse=False, batch_size=1024):
    if reverse:
        one_hot_chr = one_hot_chr[::-1, ::-1]
    indexes, data = utils.chunk_chr(one_hot_chr, winsize)
    labels = np.zeros(len(data), dtype=bool)
    X = DataGenerator(
        indexes,
        data,
        labels,
        batch_size,
        shuffle=False)
    pred = np.zeros(len(one_hot_chr))
    pred[winsize//2:-(winsize//2)] = model.predict(X).ravel()
    if reverse:
        return pred[::-1]
    else:
        return pred


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
