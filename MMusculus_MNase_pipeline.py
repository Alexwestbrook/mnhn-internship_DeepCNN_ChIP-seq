import numpy as np
from pathlib import Path
from scipy.signal import windows
import pyBigWig

input_dir = Path(
    '..', 'shared_folder', 'MMusculus', 'data', 'MNase', 'multimappers', 'A')
output_file = Path(input_dir, 'MNase_A_16_2medclip.npz')
log_file = Path(input_dir, output_file.stem + '_log.txt')
# Parameters
max_counts = 65
sigma = 15
winsize = None
q = 0.5
m = 2
if winsize is None:
    winsize = 6*sigma + 1
# Log parameters
with open(log_file, 'w') as f:
    f.write(f'input_dir: {input_dir}\n'
            f'Output_file: {output_file}\n'
            f'max_counts: {max_counts}\n'
            f'sigma: {sigma}\n'
            f'winsize: {winsize}\n'
            f'quantile: {q}\n'
            f'multiplier: {m}\n\n')

# Aggregate 3 replicas
all_chrs = {}
for i in range(3):
    # Load replica
    rep_file = Path(input_dir, f'MNase_{input_dir.name}_16_{i+1}.bw')
    bw = pyBigWig.open(str(rep_file))
    rep_chrs = {}
    for chr_id in bw.chroms():
        rep_chrs[chr_id] = np.nan_to_num(bw.values(chr_id, 0, -1, numpy=True))
    bw.close()
    # Compute step of replica
    step = min((np.min(v[v != 0]) for v in rep_chrs.values()))
    with open(log_file, 'a') as f:
        f.write(f"Step size in {rep_file} is {step}\n")

        for chr_id, value in rep_chrs.items():
            # Change values into counts then clip to max_counts
            clipped = np.clip(np.round(value / step), None, max_counts)
            f.write(f'{chr_id}: {np.sum(clipped)} counts\n')
            # Sum counts into all_chrs
            if chr_id in all_chrs.keys():
                all_chrs[chr_id] += clipped
            else:
                all_chrs[chr_id] = clipped
del clipped, rep_chrs

# Apply gaussian smoothing
for chr_id, val in all_chrs.items():
    all_chrs[chr_id] = np.convolve(
        val, windows.gaussian(winsize, sigma).astype('float32'), mode='same')

# Compute quantile of all data
quant = np.quantile(
    np.concatenate([val for val in all_chrs.values()]), q)
max_val = quant * m
with open(log_file, 'a') as f:
    f.write(f'max_val: {max_val}\n')

# Clip to given quantile
for chr_id, val in all_chrs.items():
    all_chrs[chr_id] = np.clip(val, None, max_val) / max_val

# Save in npz format
np.savez_compressed(output_file,
                    **all_chrs)
