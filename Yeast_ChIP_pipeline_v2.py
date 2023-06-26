from pathlib import Path
import numpy as np
from Modules import utils

data_dir = Path('/home/alex/shared_folder/SCerevisiae/data/GSE217022')


def remove_artifacts(signals, strand):
    """Exclude regions with abnormally high control signal.

    Regions are excluded by being set to 0.
    There is a slight variation between strands.
    """
    if strand == 'W303':
        signals['chrXII'][482_664:499_263] = 0
    elif strand == 'S288c':
        signals['VIII'][213_000:215_500] = 0
        signals['XII'][458_944:461_406] = 0
    else:
        raise ValueError("strand must be either W303 or S288c")
    return signals


def load_clean_bw(filename, strand):
    """Load file, remove artifacts and renormalize"""
    signals = utils.load_bw(filename)
    signals = remove_artifacts(signals, strand)
    # Normalize by Count Per Million Counts to have equal overall signal
    # and remove Mitochondrion
    total = sum(np.sum(v) for k, v in signals.items() if k != 'Mito')
    return {k: v * 1_000_000 / total
            for k, v in signals.items() if k != 'Mito'}


def clean_ratio(ips, ctrls, func='ratio'):
    """Compute ratio between signals by excluding 0s.

    User may specify another function instead of ratio,
    either by name if implemented or as a callable.

    Anywhere ctrls or ips is 0 will be set to 0
    """
    assert set(ips.keys()) == set(ctrls.keys())
    if func == 'ratio':
        def func(x, y): return x / y
    elif func == 'log':
        def func(x, y): return np.log(x / y)
    elif func == 'KL':
        def func(x, y): return x * np.log(x / y)
    elif func == 'sqrtKL':
        def func(x, y): return np.sqrt(x) * np.log(x / y)
    else:
        assert callable(func)
    return {k: np.where((ctrls[k] == 0) | (ips[k] == 0),
                        0,
                        func(ips[k], ctrls[k]))
            for k in ips.keys()}


def clipnorm(signals, q=0.99):
    """Clip max signal to a given quantile and normalize between 0 and 1."""
    full = np.concatenate(list(signals.values()))
    quant = np.quantile(full, q)
    return {k: np.clip(v, 0, quant) / quant for k, v in signals.items()}


# Load data
myco_coh_ip = load_clean_bw(
    Path(data_dir, 'GSM6703630_Scc1_Mmyco_rep1.CPM.bw'), 'W303')
myco_coh_ctrl = load_clean_bw(
    Path(data_dir, 'GSM6703631_Scc1-inp_Mmyco_rep1.CPM.bw'), 'W303')
pneu_coh_ip = load_clean_bw(
    Path(data_dir, 'GSM6703640_Scc1_Mpneumo_rep1.CPM.bw'), 'S288c')
pneu_coh_ctrl = load_clean_bw(
    Path(data_dir, 'GSM6703641_Scc1-inp_Mpneumo_rep1.CPM.bw'), 'S288c')

myco_pol_ip = load_clean_bw(
    Path(data_dir, 'GSM6703624_PolII_Mmyco_rep1.CPM.bw'), 'W303')
myco_pol_ctrl = load_clean_bw(
    Path(data_dir, 'GSM6703625_PolII-inp_Mmyco_rep1.CPM.bw'), 'W303')
myco_pol_ip2 = load_clean_bw(
    Path(data_dir, 'GSM6703626_PolII_Mmyco_rep2.CPM.bw'), 'W303')
myco_pol_ctrl2 = load_clean_bw(
    Path(data_dir, 'GSM6703627_PolII-inp_Mmyco_rep2.CPM.bw'), 'W303')
pneu_pol_ip = load_clean_bw(
    Path(data_dir, 'GSM6703638_PolII_Mpneumo_rep1.CPM.bw'), 'S288c')
pneu_pol_ctrl = load_clean_bw(
    Path(data_dir, 'GSM6703639_PolII-inp_Mpneumo_rep1.CPM.bw'), 'S288c')

# Merge Control and IP replicates
myco_pol_ip_merge = {k: (myco_pol_ip[k] + myco_pol_ip2[k]) / 2
                     for k in myco_pol_ctrl.keys()}
myco_ctrl_merge = {k: (myco_coh_ctrl[k]
                       + myco_pol_ctrl[k]
                       + myco_pol_ctrl2[k]) / 3
                   for k in myco_coh_ctrl.keys()}
pneu_ctrl_merge = {k: (pneu_coh_ctrl[k] + pneu_pol_ctrl[k]) / 2
                   for k in pneu_coh_ctrl.keys()}

myco_coh = clipnorm(clean_ratio(myco_coh_ip, myco_ctrl_merge, func='log'))
myco_pol = clipnorm(
    clean_ratio(myco_pol_ip_merge, myco_ctrl_merge, func='log'))
pneu_coh = clipnorm(clean_ratio(pneu_coh_ip, pneu_ctrl_merge, func='log'))
pneu_pol = clipnorm(clean_ratio(pneu_pol_ip, pneu_ctrl_merge, func='log'))

np.savez(Path(data_dir, 'labels_myco_coh_log.npz'), **myco_coh)
np.savez(Path(data_dir, 'labels_myco_pol_log.npz'), **myco_pol)
np.savez(Path(data_dir, 'labels_pneu_coh_log.npz'), **pneu_coh)
np.savez(Path(data_dir, 'labels_pneu_pol_log.npz'), **pneu_pol)
