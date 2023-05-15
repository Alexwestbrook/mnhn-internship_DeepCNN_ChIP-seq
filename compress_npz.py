import numpy as np
from pathlib import Path

for file in Path('..', 'shared_folder', 'SCerevisiae', 'results',
                 'models_etienne', 'saliency').glob('grads*.npz'):
    print(f"processing {file.name}")
    with np.load(file) as f:
        data = {k: f[k] for k in f.keys()}
    np.savez_compressed(file, **data)
