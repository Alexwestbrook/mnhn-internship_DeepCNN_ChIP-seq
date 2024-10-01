from pathlib import Path

import numpy as np
import tensorflow as tf
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

    sigma_XY = K.sum(X * Y)
    sigma_X = K.sqrt(K.sum(X * X))
    sigma_Y = K.sqrt(K.sum(Y * Y))

    return sigma_XY / (sigma_X * sigma_Y + K.epsilon())


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

    sigma_XY = K.sum(X * Y)
    sigma_X = K.sqrt(K.sum(X * X))
    sigma_Y = K.sqrt(K.sum(Y * Y))

    cor = sigma_XY / (sigma_X * sigma_Y + K.epsilon())
    mae = K.mean(K.abs(y_true - y_pred))

    return 1 + mae - cor


def get_gradients(
    model, one_hots, batch_size=1024, predict=False, output_idx=0, proj=["simplex"]
):
    grads = np.empty(one_hots.shape, dtype="float32")
    n_batches = int(np.ceil((len(one_hots) / batch_size)))
    for i in range(n_batches):
        batch_start, batch_stop = i * batch_size, (i + 1) * batch_size
        X = tf.Variable(one_hots[batch_start:batch_stop], dtype=tf.float32)
        with tf.GradientTape() as tape:
            Y = model(X, training=False)
            if output_idx is not None:
                Y = Y[..., output_idx]
        grads[batch_start:batch_stop] = np.array(tape.gradient(Y, X))
        if predict:
            Y = np.array(Y)
            if i == 0:
                preds = np.empty((len(one_hots),) + Y.shape[1:], dtype="float32")
            preds[batch_start:batch_stop] = Y
    if "simplex" in proj:
        grads -= grads.sum(axis=-1, keepdims=True)
    if "sequence" in proj:
        grads = (grads * one_hots).sum(axis=-1)
    if predict:
        return grads, preds
    else:
        return grads


custom_objects = {"correlate": correlate, "mae_cor": mae_cor}
for config in [
    # ("model_myco_nuc_2", 2001, 0, 200000, 1024)
    # ("model_myco_pol_17", 2048, 8, 200000, 1024),
    # ("model_myco_coh_14", 32768, 128, 20000, 64),  # not enough memory, see compute_grads_onseq_sumabs.py
]:
    model_name, WINDOW, output_idx, chr_chunk_size, model_batch_size = config

    genome_file = Path(
        "..", "shared_folder", "SCerevisiae", "genome", "W303_Mmmyco.npz"
    )
    model_file = Path(
        "..", "shared_folder", "SCerevisiae", "Trainedmodels", model_name, "model"
    )

    model = tf.keras.models.load_model(model_file, custom_objects=custom_objects)
    with np.load(genome_file) as f:
        for chr_id in f.keys():
            if chr_id in ["chrI"]:
                continue
            one_hot = f[chr_id]
            print(f"processing {chr_id}")
            one_hots = utils.sliding_window_view(one_hot, (WINDOW, 4)).reshape(
                -1, WINDOW, 4
            )
            n_chunks = int(np.ceil((len(one_hots) / chr_chunk_size)))
            grads = []
            for i in range(n_chunks):
                X = one_hots[chr_chunk_size * i : chr_chunk_size * (i + 1)]
                print(f"\tprocessing chunk {i} of size {len(X)}")
                grads.append(
                    get_gradients(
                        model, X, model_batch_size, proj=["simplex", "sequence"]
                    )
                )
            full_grads = np.concatenate(grads)
            del grads
            print("\tsaving output file")
            output_file = utils.safe_filename(
                f"/home/alex/shared_folder/SCerevisiae/results/{model_name}/saliency_{model_name}_{chr_id}_onseq.npz"
            )
            np.savez_compressed(output_file, full_grads)
            del full_grads

        print("Computing sum of absolute values by position")
        sumabs_grads_dict = {}
        for chr_id in f.keys():
            print(chr_id)
            with np.load(
                f"/home/alex/shared_folder/SCerevisiae/results/{model_name}/saliency_{model_name}_{chr_id}_onseq.npz"
            ) as f:
                grads = f["arr_0"]
            sumabs_grads = np.zeros(len(grads) + WINDOW - 1)
            for j in range(WINDOW):
                sumabs_grads[j : len(sumabs_grads) - WINDOW + j + 1] += np.abs(
                    grads[:, j]
                )
            del grads
            sumabs_grads /= WINDOW
            sumabs_grads_dict[chr_id] = sumabs_grads
            del sumabs_grads
        print("Saving output file")
        output_file = utils.safe_filename(
            f"/home/alex/shared_folder/SCerevisiae/results/{model_name}/saliency_{model_name}_W303_Mmmyco_onseq_sumabs.npz"
        )
        np.savez_compressed(output_file, **sumabs_grads_dict)
