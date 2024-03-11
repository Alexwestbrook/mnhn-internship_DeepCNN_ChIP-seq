#!/usr/bin/env python
import argparse
import datetime
import json
import socket
from pathlib import Path
from typing import List, Tuple, Union

import matplotlib as mpl
import numpy as np
import pandas as pd
import pyBigWig
import scipy
import scipy.stats
from matplotlib import pyplot as plt
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
        default=[1000],
        help="binsizes at which to evaluate binomial enrichment",
    )
    parser.add_argument(
        "-f",
        "--fracs",
        type=float,
        nargs="+",
        default=[1],
        help="Fractions of counts to consider for downsampling",
    )
    parser.add_argument(
        "-d",
        "--divs",
        type=float,
        nargs="+",
        help="Alternative to fracs, to specify by which numbers the counts "
        "should be divided for downsampling. Will be converted to fractions.",
    )
    parser.add_argument(
        "-fdr",
        "--use_fdr",
        action="store_true",
        help="If set, correct pvalues to control false discovery rate "
        "with Benjamini-Hochberg correction",
    )
    parser.add_argument(
        "-thres",
        "--signif_thres",
        type=float,
        default=0.05,
        help="Threshold to consider pvalues as significant",
    )
    parser.add_argument(
        "-only_ip",
        "--only_downsample_ip",
        action="store_true",
        help="If set, only perform downsampling on ip file",
    )
    parser.add_argument(
        "--plot_on_x",
        type=str,
        choices=["Total_cov", "IP_cov", "Ctrl_cov"],
        help="Value to plot against on x-axis",
    )
    args = parser.parse_args()
    # Set default value for plot_on_x
    if args.plot_on_x is None:
        if args.only_downsample_ip:
            args.plot_on_x = "IP_cov"
        else:
            args.plot_on_x = "Total_cov"
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


def get_binned_counts(file: str, binsize: int) -> np.ndarray:
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
            bin_values(
                bw.values(chr_id, 0, -1, numpy=True).astype(int), binsize, func=np.sum
            )
            for chr_id in bw.chroms()
        ]
    return np.concatenate(counts)


def integer_histogram_sample(
    array: np.ndarray, frac: float, return_complement: bool = False
) -> np.ndarray:
    """Randomly sample a fraction of a histogram with integer-only values.

    The sampled histogram is a an array of integers of same shape as the
    original histogram, with all values smaller of equal to original histogram
    values.

    Parameters
    ----------
    array : np.ndarray
        1D-array of integer values.
    frac : float
        fraction of the histogram to sample, the cumulative sum of the sampled
        histogram will be the rounded fraction of the original one
    return_complement: bool
        If True, return the complement sample as well

    Returns
    -------
    np.ndarray or tuple of np.ndarray
        1D-array of same length as `array`, containing the sampled histogram
        values, or if return_complement is set, a tuple containing the sample
        and its complement
    """

    def get_subhistogram(frac: int) -> np.ndarray:
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
        res = get_subhistogram(frac)
        if return_complement:
            return res, array - res
        else:
            return res
    else:
        comp = get_subhistogram(1 - frac)
        if return_complement:
            return array - comp, comp
        else:
            return array - comp


def integer_histogram_serie_sample(
    counts: pd.Series,
    frac: float = 0.5,
    return_complement: bool = False,
    dtype=np.int32,
) -> pd.Series:
    """Randomly sample a fraction of a histogram with integer-only values.

    The sampled histogram is a an series of integers of same shape as the
    original histogram, with all values smaller of equal to original histogram
    values. This version is faster than integer_histogram_sample when there are
    not too many unique values relative to the total count of the histogram. It
    also uses less memory.

    Parameters
    ----------
    counts : pd.Series
        series of integer values.
    frac : float
        fraction of the histogram to sample, the cumulative sum of the sampled
        histogram will be the rounded fraction of the original one
    return_complement: bool
        If True, return the complement sample as well

    Returns
    -------
    pd.Series or tuple of pd.Series
        Series of same length as `counts`, containing the sampled histogram
        values, or if return_complement is set, a tuple containing the sample
        and its complement
    """
    s1 = pd.Series(data=0, index=counts.index, dtype=dtype)

    gb = counts[counts > 0].groupby(counts)
    for value, g in gb:
        # indices = g.index
        s1[g.index] = np.random.binomial(value, frac, size=len(g))

    if return_complement:
        s2 = counts - s1
        return s1.astype(dtype), s2.astype(dtype)
    else:
        return s1.astype(dtype)


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


def binom_pval(
    k_array: np.ndarray,
    n_array: np.ndarray,
    p_binom: float,
) -> Tuple[int, int]:
    """Compute arraywise pvalue according to binomial distribution.

    Parameters
    ----------
    k_array: np.ndarray
        1D-array of number of successes
    n_array: np.ndarray
        1D-array of sample sizes
    p_binom: float
        Probabilitity of success

    Returns
    -------
    np.ndarray
        1D-array of pvalues
    """
    # pval == P(X >= k) == 1-P(X <= k-1)
    return 1 - scipy.stats.binom.cdf(k_array - 1, n_array, p_binom)


def binom_enrichment(
    k_array: np.ndarray,
    n_array: np.ndarray,
    p_binom: int,
    use_fdr: bool = False,
    signif_thres: float = 0.05,
) -> Tuple[int, int]:
    """Determine number of enriched positions and of clusters of such positions.

    Parameters
    ----------
    k_array: np.ndarray
        1D-array of number of successes
    n_array: np.ndarray
        1D-array of sample sizes
    p_binom: float
        Probabilitity of success
    use_fdr: bool, optional
        If True, correct pvalues to control false discovery rate
        with Benjamini-Hochberg correction
    signif_thres: int, optional
        Threshold to consider pvalues as significant

    Returns
    -------
    n_signif, int
        number of enriched positions
    n_signif_clust, int
        number ofclusters of enriched positions
    """
    pval = binom_pval(k_array, n_array, p_binom)
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
    fracs: List[float] = [1],
    divs: List[float] = None,
    use_fdr: bool = False,
    signif_thres: float = 0.05,
    only_ip: bool = True,
) -> pd.DataFrame:
    """Compute pvalues of enrichment in IP vs INPUT genome-wide.

    Parameters
    ----------
    ip_file, ctrl_file: str
        Path to the BigWig files of midpoints for IP and INPUT.
    binsizes: list
        Binsizes at which to evaluate binomial enrichment
    fracs: list
        Fractions of counts to consider for downsampling
    divs: list, optional
        Alternative to fracs, to specify by which numbers the counts
        should be divided for downsampling. Will be converted to fractions.
    use_fdr: bool, optional
        If True, correct pvalues to control false discovery rate
        with Benjamini-Hochberg correction
    signif_thres: float, optional
        Threshold to consider pvalues as significant
    only_ip: bool, optional
        If True, only perform downsampling on ip file

    Returns
    -------
    res: pd.DataFrame
        Table of results for each binsize and fraction. Columns are:
        IP_cov: total counts in IP at this fraction
        Ctrl_cov: total counts in INPUT at this fraction
        IP: number of IP-enriched bins
        Ctrl: number of INPUT-enriched bins
        Undetermined: number of undetermined bins
        IP_clust: number of IP-enriched clusters
        Ctrl_clust: number of INPUT-enriched clusters

    """
    # Convert divs to fracs
    if divs is not None:
        fracs = 1 / np.array(divs)
    # Build resulting DataFrame
    mindex = pd.MultiIndex.from_product([binsizes, fracs])
    res = pd.DataFrame(
        index=mindex,
        columns=[
            "IP_cov",
            "Ctrl_cov",
            "IP",
            "Ctrl",
            "Undetermined",
            "IP_clust",
            "Ctrl_clust",
        ],
        dtype=float,
    )
    # Start analysis
    for binsize in binsizes:
        # Load alignment data
        ip_counts = get_binned_counts(ip_file, binsize)
        ctrl_counts = get_binned_counts(ctrl_file, binsize)
        if only_ip:
            ctrl_subcounts = ctrl_counts
            ctrl_sum = np.sum(ctrl_subcounts)
        for frac in fracs:
            # Randomly sample alignment histogram
            ip_subcounts = integer_histogram_sample(ip_counts, frac)
            ip_sum = np.sum(ip_subcounts)
            if not only_ip:
                ctrl_subcounts = integer_histogram_sample(ctrl_counts, frac)
                ctrl_sum = np.sum(ctrl_subcounts)
            # Compute number of significant bins
            total_subcounts = ip_subcounts + ctrl_subcounts
            p_ip = np.sum(ip_subcounts) / np.sum(total_subcounts)
            n_signif_ip, n_signif_ip_clust = binom_enrichment(
                ip_subcounts,
                total_subcounts,
                p_ip,
                use_fdr=use_fdr,
                signif_thres=signif_thres,
            )
            n_signif_ctrl, n_signif_ctrl_clust = binom_enrichment(
                ctrl_subcounts,
                total_subcounts,
                1 - p_ip,
                use_fdr=use_fdr,
                signif_thres=signif_thres,
            )
            # Save results
            res.loc[binsize, frac] = [
                ip_sum,
                ctrl_sum,
                n_signif_ip,
                n_signif_ctrl,
                len(ip_counts) - n_signif_ip - n_signif_ctrl,
                n_signif_ip_clust,
                n_signif_ctrl_clust,
            ]
    return res


def safe_filename(file: Union[str, Path]) -> Path:
    """Make sure file can be build without overriding an other.

    If file already exists, returns a new filename with a number in between
    parenthesis. If the parent of the file doesn't exist, it is created.

    Raises
    ------
    FileExistsError
        If one of the parents of the file to create is an existing file
    """
    file = Path(file)
    # Build parent directories if needed
    if not file.parent.is_dir():
        print("Building parent directories")
        file.parent.mkdir(parents=True)
    # Change filename if it already exists
    if file.exists():
        original_file = file
        file_dups = 0
        while file.exists():
            file_dups += 1
            file = Path(
                file.parent, original_file.stem + f"({file_dups})" + file.suffix
            )
            # python3.9: file.with_stem(original_file.stem + f'({file_dups})')
        print(f"{original_file} exists, changing filename to {file}")
    return file


def make_barplot(
    df: pd.DataFrame, x="Total_cov", log_scale: bool = False
) -> mpl.figure.Figure:
    """Make a barplot of percentage of enriched bins for each binsize and fraction.

    Parameters
    ----------
    df: pd.DataFrame
        Result table of the saturation analysis, names must coincide
    x: str, optional
        Column to consider for coverage on x-axis
    log_scale: bool, optional
        Whether to set the scale of the x-axis as log.

    Returns
    -------
    fig: mpl.figure.Figure
        Figure with the barplot
    """
    df = df.copy()
    if x not in df.columns:
        if x == "Total_cov":
            df["Total_cov"] = df["IP_cov"] + df["Ctrl_cov"]
        else:
            raise ValueError(f"{x} is not a column of the DataFrame")
    df_sum = df[["IP", "Undetermined", "Ctrl"]].sum(axis=1)
    df["IP_perc"] = 100 * (df["IP"]) / df_sum
    df["Ctrl_perc"] = 100 * (df["Ctrl"]) / df_sum
    df["Undetermined_perc"] = 100 * (df["Undetermined"]) / df_sum
    binsizes = list(df.index.levels[0])
    n_axes = len(binsizes)
    fig, axes = plt.subplots(
        n_axes, 1, figsize=(10, 2 + 1 * n_axes), facecolor="w", sharey=True, sharex=True
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    fig.tight_layout()
    for binsize, ax in zip(binsizes, axes.flatten()):
        subdf = df.loc[(binsize,), :]
        if log_scale:
            div = (subdf[x] / subdf[x].diff()).max()
            width = subdf[x] / div
        else:
            width = subdf[x].diff().min() - 1
        bottom = np.zeros(len(subdf))
        for var, color in zip(
            ["IP_perc", "Ctrl_perc", "Undetermined_perc"],
            ["#d62728", "#1f77b4", "silver"],
        ):
            ax.bar(
                subdf[x],
                subdf[var],
                width,
                bottom=bottom,
                align="center",
                label=var[:-5],
                color=color,
            )
            bottom += subdf[var]
        if log_scale:
            ax.set_xscale("log")
        else:
            ax.set_xlim(left=0)
        ax.set_title(f"binsize {binsize}")
    mid_ax = axes[len(axes) // 2]
    mid_ax.legend()
    mid_ax.set_ylabel("% of bins enriched")
    return fig


if __name__ == "__main__":
    tmstmp = datetime.datetime.now()
    # Get arguments
    args = parsing()
    # Maybe build output directory
    output_file = safe_filename(Path(args.output_prefix + ".csv"))
    log_file = safe_filename(Path(args.output_prefix + "_log.txt"))
    figure_file = safe_filename(Path(args.output_prefix + "_barplot.png"))
    logfigure_file = safe_filename(Path(args.output_prefix + "_barplot_logscale.png"))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    # Store arguments and other info in log file
    with open(log_file, "w") as f:
        json.dump(
            {
                **vars(args),
                **{
                    "timestamp": str(tmstmp),
                    "machine": socket.gethostname(),
                    "output_file": str(output_file),
                    "figure_file": str(figure_file),
                    "logfigure_file": str(logfigure_file),
                },
            },
            f,
            indent=4,
        )
    try:
        # Compute and save downsample enrichment tables
        res = downsample_enrichment_analysis(
            ip_file=args.ip_file,
            ctrl_file=args.control_file,
            binsizes=args.binsizes,
            fracs=args.fracs,
            divs=args.divs,
            use_fdr=args.use_fdr,
            signif_thres=args.signif_thres,
            only_ip=args.only_downsample_ip,
        )
        res.to_csv(output_file)
        # Plot percentage of bins enriched as barplot, with and without log scale
        fig = make_barplot(res, x=args.plot_on_x, log_scale=False)
        fig.savefig(figure_file, bbox_inches="tight")
        fig = make_barplot(res, x=args.plot_on_x, log_scale=True)
        fig.savefig(logfigure_file, bbox_inches="tight")
    except:
        with open(log_file, "a") as f:
            f.write("\nAborted")
            f.write(f"\ntime: {datetime.datetime.now() - tmstmp}")
        raise
    with open(log_file, "a") as f:
        f.write(f"\ntime: {datetime.datetime.now() - tmstmp}")
