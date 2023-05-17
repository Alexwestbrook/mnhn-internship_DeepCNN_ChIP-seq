import numpy as np
from pathlib import Path

from Modules import utils


def compute_seq_contrib(grad_file, one_hot_chr):
    one_hots = utils.sliding_window_view(one_hot_chr, (2001, 4)).squeeze()
    grads = []
    with np.load(grad_file) as f:
        for i, key in enumerate(f.keys()):
            chunk_grad = f[key]
            # project gradient to simplex
            chunk_grad -= np.mean(chunk_grad, keepdims=True, axis=-1)
            # Get chunk size
            if i == 0:
                chr_chunk_size = len(chunk_grad)
            # project gradient on sequence
            X = one_hots[i*chr_chunk_size:(i+1)*chr_chunk_size]
            grads.append((chunk_grad * X).sum(axis=-1))
    # Compute sum over positions in window
    for i, chunk_grad in enumerate(grads):
        if i == 0:
            pos_sum = np.sum(np.abs(chunk_grad), axis=0)
        else:
            pos_sum += np.sum(np.abs(chunk_grad), axis=0)
    # Sum over samples
    seq_contrib = np.zeros(len(one_hot_chr), dtype='float32')
    chunk_start = 0
    for chunk_grad in grads:
        for i in range(len(chunk_grad)):
            seq_contrib[chunk_start+i:chunk_start+i+2001] += np.abs(
                chunk_grad[i])
        chunk_start += len(chunk_grad)
    return seq_contrib, grads, pos_sum


data_dir = '../shared_folder'
species = 'SCerevisiae'

with np.load(Path(data_dir, species, 'genome', 'W303', 'W303_ATCG.npz')) as f:
    one_hots_yeast_ATCG = {k: f[k] for k in f.keys() if k[:3] == 'chr'}

with np.load(Path(data_dir, species, 'results', 'models_etienne',
                  'preds_weights_myco_rep1_on_W303.npz')) as f:
    preds_nuc = {k: f[k] for k in f.keys() if k[:3] == 'chr'}

preds_nuc_yeast = np.concatenate(
    [v for k, v in preds_nuc.items() if k[:3] == 'chr'])
q04, q06 = np.quantile(preds_nuc_yeast, [0.4, 0.6])

seq_contrib, pos_sum = {}, {}
sign_seq_contrib = {}
for chr_id, one_hot in one_hots_yeast_ATCG.items():
    if chr_id[:3] == 'chr':
        print(f'processing {chr_id}')
        grad_file = Path(data_dir, species, 'results', 'models_etienne',
                         'saliency', f'grads_weights_myco_rep1_{chr_id}.npz')
        seq_contrib[chr_id], grads, pos_sum[chr_id] = compute_seq_contrib(
            grad_file, one_hot)
        # compute signed contribution
        mask = np.zeros(len(one_hot), dtype='int8')
        mask[preds_nuc[chr_id] > q06] = 1
        mask[preds_nuc[chr_id] < q04] = -1
        sign_seq_contrib[chr_id] = np.zeros(len(one_hot), dtype='float32')
        chunk_start = 0
        for chunk_grad in grads:
            for i in range(len(chunk_grad)):
                sign_seq_contrib[chr_id][chunk_start+i:chunk_start+i+2001] += (
                    chunk_grad[i] * mask[chunk_start+i+1000])
            chunk_start += len(chunk_grad)
        del grads
np.savez(Path(data_dir, species, 'results', 'models_etienne', 'saliency',
              'seq_contrib_weights_myco_rep1.npz'),
         **seq_contrib)
np.savez(Path(data_dir, species, 'results', 'models_etienne', 'saliency',
              'seq_contrib_weights_myco_rep1.npz'),
         **sign_seq_contrib)
np.savez(Path(data_dir, species, 'results', 'models_etienne', 'saliency',
              'grads_abs_pos_sum_weights_myco_rep1.npz'),
         **pos_sum)
# for chr_id, array in sign_seq_contrib.items():
#     tmp_contrib = utils.strided_window_view(
#         array.reshape(-1, 1) * one_hots_yeast_ATCG[chr_id], (2000, 4), 1600
#     ).squeeze()
#     sign_seq_contrib[chr_id] = np.transpose(tmp_contrib, [0, 2, 1])
# np.savez(Path(data_dir, species, 'results', 'models_etienne', 'saliency',
#               'sign_seq_contrib_weights_myco_rep1.npz'),
#          np.concatenate(list(sign_seq_contrib.values()), axis=0))
