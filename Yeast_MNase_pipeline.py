from pathlib import Path
import numpy as np
from Modules import utils

data_dir = Path('/home/alex/shared_folder/SCerevisiae/data')


def clipnorm(signals, q=0.99):
    """Clip max signal to a given quantile and normalize between 0 and 1."""
    full = np.concatenate(list(signals.values()))
    quant = np.quantile(full, q)
    return {k: np.clip(v, None, quant) / quant for k, v in signals.items()}


mnase_myco = utils.load_bw(Path(data_dir, 'mnase_myco.bw'))
mnase_pneumo = utils.load_bw(Path(data_dir, 'mnase_pneumo.bw'))

myco_nuc = clipnorm(mnase_myco)
pneu_nuc = clipnorm({k: v for k, v in mnase_pneumo.items() if k != 'Mito'})

np.savez_compressed(Path(data_dir, 'labels_myco_nuc.npz'), **myco_nuc)
np.savez_compressed(Path(data_dir, 'labels_pneu_nuc.npz'), **pneu_nuc)

utils.write_bw(Path(data_dir, 'labels_myco_nuc.bw'), myco_nuc)
utils.write_bw(Path(data_dir, 'labels_pneu_nuc.bw'), pneu_nuc)
