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
                          chr='unknown_chr',
                          vmin=None,
                          vmax=None):
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
                     label='peak center')
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
                    robust=False, vmin=vmin, vmax=vmax)
        return_values = (corrs,)
    elif plot == 'clustermap':
        corr_matrix = np.corrcoef(values[indices])
        link_matrix = linkage(corr_matrix, method='ward')
        df = pd.DataFrame(values[indices], columns=window)
        clust = sns.clustermap(values[indices], center=0,
                               row_linkage=link_matrix, col_cluster=False,
                               xticklabels=window_half_size,
                               yticklabels=1000, robust=False)
        return_values = (clust,)
    if res_dir:
        plt.savefig(Path(res_dir, f'metaplot_{data}_{plot}_{chr}_peaks.png'),
                    bbox_inches='tight')
    plt.show()
    plt.close()
    return return_values + (mean_values, window)


def add_legend(axes):
    handles, labels = [], []
    for ax in axes:
        handle, label = ax.get_legend_handles_labels()
        handles += handle
        labels += label
    return handles, labels


def binned_plot(values, ax, start, end, bins, **kwargs):
    bin_start = (start // bins) * bins + bins // 2
    bin_end = (end // bins) * bins + bins // 2
    ax.plot(
        np.arange(bin_start, bin_end, bins),
        values[start//bins:end//bins],
        **kwargs)


def compare_binned_plots(values_list, label_list, start, end, bins, **kwargs):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    axes = []
    for i, values in enumerate(values_list):
        if i == 0:
            axes.append(plt.subplot())
        else:
            axes.append(axes[i-1].twinx())
        binned_plot(values, axes[i], start, end, bins,
                    label=label_list[i], color=colors[i], **kwargs)
    handles, labels = add_legend(axes)
    leg = plt.legend(handles, labels, fontsize=16)
    for line in leg.get_lines():
        line.set_linewidth(2)
    return axes
