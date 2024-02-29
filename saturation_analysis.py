#!/usr/bin/env python
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import pyBigWig
import scipy
import scipy.stats
from statsmodels.stats import multitest


def parsing():
    """
    Parse the command-line arguments.

    Arguments
    ---------
    python command-line

    Returns
    -------
    """
    # Declaration of expexted arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--output_prefix", type=str, required=True, help="output file prefix"
    )
    parser.add_argument(
        "-i",
        "--ip_file",
        type=str,
        required=True,
        help="bigwig file of mid point counts in treatment (ip)",
    )
    parser.add_argument(
        "-c",
        "--control_file",
        type=str,
        required=True,
        help="bigwig file of mid point counts in control (input)",
    )
    parser.add_argument(
        "-b",
        "--binsizes",
        type=int,
        nargs="+",
        required=True,
        help="bigwig file of mid point counts in control (input)",
    )
    args = parser.parse_args()
    return args


def bin_values(array: np.ndarray, binsize: int, func=np.mean) -> np.ndarray:
    """Compute summary statistics on bins of an array

    If array length isn't divisible by binsize, the last bin will be smaller

    Parameters
    ----------
    array: np.ndarray
        1D input data
    binsize: int
        length of bins, must be greater than 0
    func: callable
        function computing summary statistic, must support axis parameter
        (ex: np.mean, np.sum)

    Returns
    -------
    res, np.ndarray
        Binned array
    """
    if binsize <= 0:
        raise ValueError("binsize must be greater than 0")
    nbins, r = divmod(len(array), binsize)
    res = func(array[: nbins * binsize].reshape(nbins, binsize), axis=1)
    if r != 0:
        res = np.append(res, func(array[-r:]))
    return res


def nb_boolean_true_clusters(array: np.ndarray) -> int:
    """Compute the number of clusters of True values in array.

    Parameters
    ----------
    array : array_like
        1D-array of boolean values.

    Returns
    -------
    int
        number of clusters of True values
    """
    # Clusters ~ switches in value / 2
    res = np.sum(np.diff(array)) // 2
    # Add one if at least a cluster is on the edge
    if array[0] or array[-1]:
        res += 1
    return res


def zero_to_epsilon(array: np.ndarray, copy: bool = True) -> np.ndarray:
    """Change all zero values to the minimal non-zero value.

    Parameters
    ----------
    array : np.ndarray
        Input array

    Returns
    -------
    np.ndarray
        Array with zeros replaced
    """
    if copy:
        array = array.copy()
    array[array == 0] = np.finfo(array.dtype).eps
    return array


def integer_histogram_sample(array: np.ndarray, frac: float) -> np.ndarray:
    """Sample a random fraction of a histogram with integer-only values.

    The sampled histogram is a an array of integers of same shape as the
    original histogram, with all values smaller of equal to original histogram
    values.

    Parameters
    ----------
    array : array_like
        1D-array of integer values.
    frac : float
        fraction of the histogram to sample, the cumulative sum of the sampled
        histogram will be the rounded fraction of the original one

    Returns
    -------
    np.ndarray
        1D-array of same length as `array`, containing the sampled histogram
        values
    """

    def get_subhistogram(frac):
        sampled_pos = rng.choice(
            positions, size=round(len(positions) * frac), replace=False
        )
        histogram = (
            scipy.sparse.coo_matrix(
                (
                    np.ones(len(sampled_pos), dtype=int),
                    (sampled_pos, np.zeros(len(sampled_pos), dtype=int)),
                ),
                shape=(len(array), 1),
            )
            .toarray()
            .ravel()
        )
        return histogram

    positions = np.repeat(np.arange(len(array), dtype=int), array)
    rng = np.random.default_rng()
    # get_subhistogram complexity is linear in frac, computing complement may save time
    if frac <= 0.5:
        return get_subhistogram(frac)
    else:
        return array - get_subhistogram(1 - frac)


def get_binned_counts(file, binsize):
    """Extract sum of counts by bin in file on all contigs

    Parameters
    ----------
    file: str
        bigwig file of counts
    binsize: int
        length of bins, must be greater than 0

    Returns
    -------
    np.ndarray
        concatenation of binned counts per chromosome
    """
    with pyBigWig.open(file) as bw:
        counts = [
            bin_values(bw.values(chr_id, 0, -1, numpy=True), binsize, func=np.sum)
            for chr_id in bw.chroms()
        ]
    return np.concatenate(counts)


def binom_enrichment(
    k_array: np.ndarray,
    n_array: np.ndarray,
    n_sum: int,
    use_fdr: bool = False,
    signif_thres: float = 0.05,
) -> Tuple[int, int]:
    p_binom = np.sum(k_array) / n_sum
    pval = zero_to_epsilon(
        1 - scipy.stats.binom.cdf(k_array - 1, n_array, p_binom), copy=False
    )
    # Extract significant bins
    if use_fdr:
        # correct with q-value on non-empty bins
        valid_bins = n_array != 0
        signif = np.zeros(len(k_array), dtype=bool)
        signif[valid_bins], *_ = multitest.multipletests(
            pval[valid_bins], method="fdr_bh"
        )
    else:
        signif = np.array(pval < signif_thres)
    n_signif = np.sum(signif)
    n_signif_clust = nb_boolean_true_clusters(signif)
    return n_signif, n_signif_clust


def downsample_enrichment_analysis(
    ip_file: str,
    ctrl_file: str,
    binsizes: List[int] = [1000],
    fracs=[1],
    divs=None,
    use_fdr=False,
    signif_thres=0.05,
):
    # Convert divs to fracs
    if divs is not None:
        fracs = 1 / np.array(divs)
    # Build resulting DataFrame
    mindex = pd.MultiIndex.from_product([binsizes, fracs])
    res = pd.DataFrame(
        index=mindex,
        columns=[
            "IP",
            "IP_clust",
            "Undetermined",
            "Ctrl",
            "Ctrl_clust",
            "total_cov",
        ],
    )
    # Start analysis
    for binsize in binsizes:
        # Load alignment data
        ip_counts = get_binned_counts(ip_file, binsize)
        ctrl_counts = get_binned_counts(ctrl_file, binsize)
        for frac in fracs:
            # Randomly sample alignment histogram
            ip_subcounts = integer_histogram_sample(ip_counts, frac)
            ctrl_subcounts = integer_histogram_sample(ctrl_counts, frac)
            # Compute number of significant bins
            total_subcounts = ip_subcounts + ctrl_subcounts
            total_sum = np.sum(total_subcounts)
            n_signif_ip, n_signif_ip_clust = binom_enrichment(
                ip_subcounts,
                total_subcounts,
                total_sum,
                use_fdr=use_fdr,
                signif_thres=signif_thres,
            )
            n_signif_ctrl, n_signif_ctrl_clust = binom_enrichment(
                ctrl_subcounts,
                total_subcounts,
                total_sum,
                use_fdr=use_fdr,
                signif_thres=signif_thres,
            )
            # Save results
            res.loc[binsize, frac] = [
                n_signif_ip,
                n_signif_ip_clust,
                len(ip_counts) - n_signif_ip - n_signif_ctrl,
                n_signif_ctrl,
                n_signif_ctrl_clust,
                total_sum,
            ]
    return res


def main():
    pass


if __name__ == "__main__":
    main()
