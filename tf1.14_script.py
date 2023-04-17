import tensorflow as tf
from tensorflow.keras import backend as K
from pathlib import Path
import numpy as np
import deeplift
from deeplift.conversion import kerasapi_conversion as kc
from deeplift.util import get_shuffle_seq_ref_function
from deeplift.dinuc_shuffle import dinuc_shuffle
print(tf.__version__)


def correlate(y_true, y_pred):
    """
        Calculate the correlation between the predictions and the labels.

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


def mae_cor(y_true, y_pred):
    """
        Calculate the mean absolute error minus the correlation between
        predictions and  labels.

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

    return 1 + mae - cor


def rolling_window(array, window=(0,), asteps=None, wsteps=None, axes=None,
                   toend=True):
    """
        Take a numpy array and return a view of this array after applying a
        rolling window.

        This takes a numpy and cut it in several pieces with the size, the
        stride and the axes asked as needed. You may want to use it in order
        to create a set of sequences from an array.

        :param array: The array we want to cut
        :param window: The length of the window
        :param asteps: The stride between two window applied
        :param wsteps: The stride whitin the window
        :param axes: The axe on which to apply the rolling window
        :param toend: Weither or not to finish the cut
        :type array: numpy array
        :type window: int or tuple
        :type asteps: int or tuple
        :type wsteps: int or tuple
        :type axes: int
        :type toend: boolean
        :return: The view of the array
        :rtype: numpy array

        :Example:

        >>> a = numpy.array([0,1,2,3,4,5])
        >>> rolling_window(a, window = 2, asteps = 2, wsteps = None)
        array([[0,1],
               [2,3],
               [4,5]])
        >>> rolling_window(a, window = 2, asteps = None, wsteps = 2)
        array([[0,2],
               [1,3],
               [2,4]
               [3,5]])
        >>> rolling_window(a, window = 5, asteps = 2, wsteps = None)
        array([[0,1,2,3,4]])

        .. warning:: Be carreful about the combination of window, wsteps and
        asteps that may raise  ValueError. This function forces the window to
        be of the asked size and thus may stop the application of the window
        before the end.
    """

    array = np.asarray(array)
    orig_shape = np.asarray(array.shape)
    window = np.atleast_1d(window).astype(int)  # maybe crude to cast to int...

    if axes is not None:
        axes = np.atleast_1d(axes)
        w = np.zeros(array.ndim, dtype=int)
        for axis, size in zip(axes, window):
            w[axis] = size
        window = w

    # Check if window is legal:
    if window.ndim > 1:
        raise ValueError("`window` must be one-dimensional.")
    if np.any(window < 0):
        raise ValueError("All elements of `window` must be larger than 1.")
    if len(array.shape) < len(window):
        raise ValueError(
            "`window` length must be less or equal `array` dimension.")

    _asteps = np.ones_like(orig_shape)
    if asteps is not None:
        asteps = np.atleast_1d(asteps)
        if asteps.ndim != 1:
            raise ValueError(
                "`asteps` must be either a scalar or one dimensional.")
        if len(asteps) > array.ndim:
            raise ValueError(
                "`asteps` cannot be longer then the `array` dimension.")
        # does not enforce alignment, so that steps can be same as window too.
        _asteps[-len(asteps):] = asteps

        if np.any(asteps < 1):
            raise ValueError("All elements of `asteps` must be larger then 1.")
    asteps = _asteps

    _wsteps = np.ones_like(window)
    if wsteps is not None:
        wsteps = np.atleast_1d(wsteps)
        if wsteps.shape != window.shape:
            raise ValueError("`wsteps` must have the same shape as `window`.")
        if np.any(wsteps < 0):
            raise ValueError("All elements of `wsteps` must be larger then 0.")

        _wsteps[:] = wsteps
        # make sure that steps are 1 for non-existing dims.
        _wsteps[window == 0] = 1
    wsteps = _wsteps

    # Check that the window would not be larger than the original:
    if np.any(orig_shape[-len(window):] < window * wsteps):
        raise ValueError(
            "`window` * `wsteps` larger then `array` in at least one "
            "dimension.")

    new_shape = orig_shape  # just renaming...

    # For calculating the new shape 0s must act like 1s:
    _window = window.copy()
    _window[_window == 0] = 1

    new_shape[-len(window):] += wsteps - _window * wsteps
    new_shape = (new_shape + asteps - 1) // asteps
    # make sure the new_shape is at least 1 in any \"old\" dimension (ie. steps
    # is (too) large, but we do not care.
    new_shape[new_shape < 1] = 1
    shape = new_shape

    strides = np.asarray(array.strides)
    strides *= asteps
    new_strides = array.strides[-len(window):] * wsteps

    # The full new shape and strides:
    if toend:
        new_shape = np.concatenate((shape, window))
        new_strides = np.concatenate((strides, new_strides))
    else:
        _ = np.zeros_like(shape)
        _[-len(window):] = window
        _window = _.copy()
        _[-len(window):] = new_strides
        _new_strides = _

        new_shape = np.zeros(len(shape)*2, dtype=int)
        new_strides = np.zeros(len(shape)*2, dtypenucleotid=int)

        new_shape[::2] = shape
        new_strides[::2] = strides
        new_shape[1::2] = _window
        new_strides[1::2] = _new_strides

    new_strides = new_strides[new_shape != 0]
    new_shape = new_shape[new_shape != 0]

    return np.lib.stride_tricks.as_strided(
        array, shape=new_shape, strides=new_strides)


def nuc_shuffle(seq, num_shufs=1, rng=None):
    return seq[rng.rand(num_shufs, len(seq)).argsort(axis=1)]


deeplift_model = kc.convert_model_from_saved_files(
    Path('shared_folder', 'SCerevisiae', 'models_etienne',
         'weights_myco_rep1.hdf5'),
    nonlinear_mxts_mode=deeplift.layers.NonlinearMxtsMode.DeepLIFT_GenomicsDefault
    # nonlinear_mxts_mode=deeplift.layers.NonlinearMxtsMode.RevealCancel
    )
find_scores_layer_idx = 0
# with np.load(Path('shared_folder', 'SCerevisiae', 'genome', 'W303',
#                   'W303_ATGC.npz')) as f:
#     one_hot = f['chr16']
# seq_start = 114_050
# X = rolling_window(
#     one_hot, window=(2001, 4))[seq_start:seq_start+1001, 0, :, None, :]
# X = rolling_window(
#     one_hot, window=(2001, 4))[:, 0, :, None, :]
with np.load(Path('shared_folder', 'SCerevisiae', 'results', 'models_etienne',
                  'modisco', 'onehot_slidemaxchr16.npz')) as f:
    X = np.transpose(f['arr_0'], [0, 2, 1]).reshape(-1, 2001, 1, 4)
deeplift_contribs_func = deeplift_model.get_target_contribs_func(
    find_scores_layer_idx=find_scores_layer_idx,
    target_layer_idx=-2)
# scores = np.array(deeplift_contribs_func(task_idx=0,
#                                          input_data_list=[X],
#                                          input_references_list=[
#                                           np.array([0.31,
#                                                     0.19,
#                                                     0.19,
#                                                     0.38])[None, None, :]],
#                                          batch_size=10,
#                                          progress_update=1000))
func_many_refs = get_shuffle_seq_ref_function(
    score_computation_function=deeplift_contribs_func,
    shuffle_func=dinuc_shuffle)
scores = func_many_refs(task_idx=0,
                        input_data_sequences=X,
                        num_refs_per_seq=20,
                        batch_size=200,
                        progress_update=1000)
print(scores.shape)
np.save(Path('shared_folder', 'SCerevisiae', 'results',
             'models_etienne', 'DeepLIFT',
             'score_nuc1_slidemaxchr16_vs_20dinucshuffle.npy'),
        scores.squeeze())

# model = tf.keras.models.load_model(
#     Path('shared_folder', 'SCerevisiae', 'models_etienne',
#          'weights_myco_rep1.hdf5'),
#     custom_objects={'correlate': correlate, 'mae_cor': mae_cor}
#     )
# with np.load(Path('shared_folder', 'SCerevisiae', 'genome', 'W303',
#                   'W303_ATGC.npz')) as f:
#     one_hot = f['chr16']
# print(one_hot.shape)
# pred = model.predict(one_hot[:2001].reshape(1, 2001, 1, 4))[:, 0]
# print(pred)
# del model
