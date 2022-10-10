#!/usr/bin/env python
from pathlib import Path

import numpy as np

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from scipy.cluster.hierarchy import linkage
import Modules.utils as utils


def metaplot_over_indices(values,
                          indices,
                          window_half_size,
                          label='values',
                          compare=None,
                          anchor='center',
                          plot='simple',
                          res_dir=None,
                          data='unknown_data',
                          chr='unknown_chr'):
    if anchor == 'center':
        window = np.arange(-window_half_size, window_half_size + 1)
    elif anchor == 'start':
        window = np.arange(2*window_half_size + 1)
    elif anchor == 'end':
        window = np.arange(-2*window_half_size, 1)
    else:
        raise NameError("Invalid anchor")
    # broadcast column of indices and window line
    indices = np.expand_dims(indices, axis=1) + np.expand_dims(window, axis=0)
    # compute mean over original indices at each point of the window
    mean_values = np.mean(values[indices], axis=0)
    if compare:
        compare_values, compare_label = compare
        mean_compare = np.mean(compare_values[indices], axis=0)
    # Make desired plot
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    if plot == 'simple':
        fig, ax = plt.subplots()
        ax.plot(window, mean_values, label=label, color=colors[1])
        handles, labels = ax.get_legend_handles_labels()
        if compare:
            ax2 = ax.twinx()
            ax2.plot(window, mean_compare, label=compare_label)
            handles2, labels2 = ax2.get_legend_handles_labels()
            handles += handles2
            labels += labels2
        ax.legend(handles, labels)
        return_values = tuple()
    elif plot == 'heatmap':
        corrs = utils.lineWiseCorrcoef(values[indices], mean_values)
        new_indices = indices[np.argsort(corrs)[::-1], :]
        fig, axs = plt.subplots(2, 2, sharex='col', figsize=(9, 11),
                                gridspec_kw={'width_ratios': [20, 1],
                                             'height_ratios': [6, 20]})
        # Average plot
        ax00 = axs[0, 0]
        ax00.plot(mean_values, label=label, color=colors[1])
        ax00.axvline(x=window_half_size+1, color='black', linestyle='--',
                     label='ENCODE peak center')
        handles, labels = ax00.get_legend_handles_labels()
        if compare:
            ax2 = ax00.twinx()
            ax2.plot(mean_compare, label=compare_label)
            handles2, labels2 = ax2.get_legend_handles_labels()
            handles += handles2
            labels += labels2
        ax00.xaxis.set_tick_params(which='both', labelbottom=True)
        ax00.legend(handles, labels)
        # remove upper right ax
        axs[0, 1].remove()
        # Heatmap
        df = pd.DataFrame(values[new_indices], columns=window)
        sns.heatmap(df, center=0, ax=axs[1, 0], cbar_ax=axs[1, 1],
                    xticklabels=window_half_size, yticklabels=1000,
                    robust=True, vmin=0, vmax=1)
        return_values = (corrs,)
    elif plot == 'clustermap':
        corr_matrix = np.corrcoef(values[indices])
        link_matrix = linkage(corr_matrix, method='ward')
        df = pd.DataFrame(values[indices], columns=window)
        clust = sns.clustermap(values[indices], center=0,
                               row_linkage=link_matrix, col_cluster=False,
                               xticklabels=window_half_size,
                               yticklabels=1000, robust=True)
        return_values = (clust,)
    if res_dir:
        plt.savefig(Path(res_dir, f'metaplot_{data}_{plot}_{chr}_peaks.png'),
                    bbox_inches='tight')
    plt.show()
    plt.close()
    return return_values + (mean_values, window)
