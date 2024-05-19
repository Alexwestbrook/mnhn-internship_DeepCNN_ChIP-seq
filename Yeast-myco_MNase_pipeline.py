import numpy as np
from pathlib import Path
import pyBigWig
import scipy
import argparse
import json
from Modules import utils
data_dir = '../shared_folder'

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i", "--input",
    help='BigWig file to use as input',
    type=str,
    required=True)
parser.add_argument(
    "-o", "--output",
    help='output file prefix, to which will be appended npz suffix',
    type=str,
    required=True)
parser.add_argument(
    "-pc", "--postclip",
    help='indicate to clip after smoothing',
    action='store_true')
parser.add_argument(
    "-r0", "--remove_0",
    help='indicate to remove 0s when computing quantiles for clipping',
    action='store_true')

# Get arguments
args = parser.parse_args()
# Store arguments in file
out_file = utils.safe_filename(Path(args.output + '.npz'))
log_file = utils.safe_filename(Path(args.output + '_log.txt'))
args.output = str(out_file)
with open(log_file, 'w') as f:
    json.dump(vars(args), f, indent=4)
    f.write('\n')

# Load MNase signal by chromosome into a dictionnary
bw = pyBigWig.open(str(args.input))
chr_values = {}
for chr_id in bw.chroms().keys():
    chr_id_asint = 'chr' + format(utils.roman_to_int(chr_id[3:]), '02d')
    chr_values[chr_id_asint] = bw.values(chr_id, 0, -1, numpy=True)
    # chr_values[chr_id] = bw.values(chr_id, 0, -1, numpy=True)

# Get step corresponding to a single count
all_values = np.concatenate([val for val in chr_values.values()])
step = np.unique(all_values)[1]

# Transform signal into mid point counts
for key, val in chr_values.items():
    # Transform signal into counts
    val = np.round(val / step)
    # Remove first count of each pair
    val = np.insert(np.diff(np.cumsum(val) // 2), 0, 0)
    chr_values[key] = val


# Clip and smooth signal
all_values = np.concatenate([val for val in chr_values.values()])
if args.remove_0:
    all_values = all_values[all_values != 0]
quant99 = np.quantile(all_values, 0.99)
for key, val in chr_values.items():
    # Clip to 99th percentile (remove artifacts)
    val = np.clip(val, 0, quant99)
    # Apply gaussian smoothing
    chr_values[key] = np.convolve(
        val, scipy.signal.windows.gaussian(91, 15), mode='same')

# Clip to 99th percentile again (bump up the signal)
if args.postclip:
    all_values = np.concatenate([val for val in chr_values.values()])
    if args.remove_0:
        all_values = all_values[all_values != 0]
    quant99 = np.quantile(all_values, 0.99)
    for key, val in chr_values.items():
        chr_values[key] = np.clip(val, 0, quant99)

# Normalize in [0, 1]
for key, val in chr_values.items():
    chr_values[key] = (val - np.min(val))/(np.max(val) - np.min(val))

# Save in npz format
np.savez_compressed(out_file, **chr_values)


# # Legacy pipeline
# file = Path(data_dir, 'SCerevisiae', 'data', 'Yeast-myco_MNase_exp.bw')
# # Load MNase signal by chromosome into a dictionnary
# bw = pyBigWig.open(str(file))
# chr_values = {}
# for chr_id in bw.chroms().keys():
#     chr_id_asint = 'chr' + format(utils.roman_to_int(chr_id[3:]), '02d')
#     chr_values[chr_id_asint] = bw.values(chr_id, 0, -1, numpy=True)
#     # chr_values[chr_id] = bw.values(chr_id, 0, -1, numpy=True)

# # Get step corresponding to a single count
# all_values = np.concatenate([val for val in chr_values.values()])
# step = np.unique(all_values)[1]

# for key, val in chr_values.items():
#     # Transform signal into counts
#     val = np.round(val / step)
#     # Remove first count of each pair
#     val = np.insert(np.diff(np.cumsum(val) // 2), 0, 0)
#     # Clip to a maximum of 65 counts
#     val = np.clip(val, a_min=None, a_max=65)
#     # Apply gaussian smoothing
#     chr_values[key] = np.convolve(
#         val, scipy.signal.windows.gaussian(100, 15), mode='same')

# # Clip to 99th percentile and normalize in [0, 1]
# all_values = np.concatenate([val for val in chr_values.values()])
# quant99 = np.quantile(all_values, 0.99)
# for key, value in chr_values.items():
#     chr_values[key] = np.clip(value, None, quant99) / quant99

# # Save in npz format
# save_file = utils.safe_filename(
#     Path(data_dir, 'SCerevisiae', 'data', 'processed_MNase.npz'))
# np.savez_compressed(save_file, **chr_values)
