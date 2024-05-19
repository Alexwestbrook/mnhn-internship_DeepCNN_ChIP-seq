import tensorflow as tf
import numpy as np
from pathlib import Path
from Modules import utils
from tensorflow.keras import backend as K


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


def get_gradients(model, one_hots, batch_size=1024, predict=False):
    grads = np.empty(one_hots.shape, dtype='float32')
    n_batches = int(np.ceil((len(one_hots) / batch_size)))
    for i in range(n_batches):
        batch_start, batch_stop = i*batch_size, (i+1)*batch_size
        X = tf.Variable(one_hots[batch_start:batch_stop], dtype=tf.float32)
        with tf.GradientTape() as tape:
            Y = model(X, training=False)
        grads[batch_start:batch_stop] = np.array(tape.gradient(Y, X))
        if predict:
            Y = np.array(Y).squeeze()
            if i == 0:
                preds = np.empty((len(one_hots),) + Y.shape[1:],
                                 dtype='float32')
            preds[batch_start:batch_stop] = Y
    if predict:
        return grads.squeeze(), preds
    else:
        return grads.squeeze()


model_batch_size = 1024
chr_chunk_size = 200000
WINDOW = 2001

genome_file = Path('..', 'shared_folder', 'SCerevisiae', 'genome', 'W303',
                   'W303_ATCG.npz')
model_name = 'weights_myco_rep1'
model_file = Path('..', 'shared_folder', 'SCerevisiae', 'models_etienne',
                  f'{model_name}.hdf5')
custom_objects = {'correlate': correlate, 'mae_cor': mae_cor}
output_dir = Path(
    '..', 'shared_folder', 'SCerevisiae', 'results', 'models_etienne',
    'saliency')

model = tf.keras.models.load_model(model_file, custom_objects=custom_objects)
with np.load(genome_file) as f:
    for chr_id in f.keys():
        if chr_id[:3] == 'chr':
            one_hot = f[chr_id]
        else:
            continue
        print(f'processing {chr_id}')
        output_file = Path(output_dir, f'grads_{model_name}_{chr_id}.npz')
        output_file = utils.safe_filename(output_file)
        one_hots = utils.sliding_window_view(
            one_hot, (WINDOW, 4)
        ).reshape(-1, 2001, 1, 4)
        n_chunks = int(np.ceil((len(one_hots) / chr_chunk_size)))
        grads = []
        for i in range(n_chunks):
            X = one_hots[chr_chunk_size*i:chr_chunk_size*(i+1)]
            print(f'\tprocessing chunk {i} of size {len(X)}')
            grads.append(get_gradients(model, X, model_batch_size))
        print('\tsaving output file')
        np.savez_compressed(output_file, *grads)
