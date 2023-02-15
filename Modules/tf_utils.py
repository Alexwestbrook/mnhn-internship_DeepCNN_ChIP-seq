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
            binary_crossentropy(y_true, y_pred[0])
            + binary_crossentropy(y_true - y_pred[0], y_pred[1]),
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
                 temp_dir: Path = None,
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
