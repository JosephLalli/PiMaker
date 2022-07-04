"""
This module contains functions related to calculating pi diversity from read counts.

Given numpy arrays of read counts at variable sites, this module implements the
formulas described in `Nelson and Hughes (2015)`_ to calculate pi diversity.

Nelson C.W., Hughes A.L. (2015) Within-host nucleotide diversity of virus
populations: Insights from next-generation sequencing. Infect. Genet. Evol., 30, 1–7.

"""

import numpy as np


def calculate_pi_math(read_cts):
    """
    Calculates 'pi math' from read counts.

    Per `Nelson and Hughes (2015)`_, pi is the sum of the product of the read
    counts of all possible nucleotide combinations (AC, AG, AT, CG, CT, GT).
    This function performs this math for all sites in the read count array.

    Args:
        read_cts:
            A (number of samples x n x 4) array of read counts. n is an
            arbitrary number of nucleotide sites.
    Returns:
        A (number of samples x n x 7) array containing the total number of
        reads at each site for each sample in the third axis' 0 index,
        and then the (AC, AG, AT, CG, CT, GT) products of the number of reads
        for each site and each sample at nucleotides at indices 1-6.
    """
    pi_math = np.zeros(shape=(read_cts.shape[0:2] + (7,)), dtype=np.int32)
    pi_math[:, :, 0] = np.sum(read_cts, axis=2)
    pi_math[:, :, 1] = read_cts[:, :, 0] * read_cts[:, :, 1]
    pi_math[:, :, 2] = read_cts[:, :, 0] * read_cts[:, :, 2]
    pi_math[:, :, 3] = read_cts[:, :, 0] * read_cts[:, :, 3]
    pi_math[:, :, 4] = read_cts[:, :, 1] * read_cts[:, :, 2]
    pi_math[:, :, 5] = read_cts[:, :, 1] * read_cts[:, :, 3]
    pi_math[:, :, 6] = read_cts[:, :, 2] * read_cts[:, :, 3]
    return pi_math


def per_site_pi(pi_math):
    """
    Calculates pi at each site of each sample.

    Uses the formula in `Nelson and Hughes (2015)`_.
    pi = sum(pi_math)/((depth**2 - depth)/2) for each site and each sample.

    Args:
        pi_math: A (number of samples x n x 7) array of pi math. See
        :func:`calculate_pi_math` for a description of this array.
    Returns:
        A (number of samples x n) array of pi values.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.sum(pi_math[:, :, 1:], axis=2) / ((pi_math[:, :, 0]**2 - pi_math[:, :, 0]) / 2)
    return np.nan_to_num(result)


def avg_pi_per_sample(site_pi, length=None):
    """
    Calculates average pi per sample from an array of pi values.

    Args:
        site_pi:
            A (number of samples x n) array of pi values.
        length:
            Optional, default None. The length of the sequence(s) for which
            we are calculating pi. Can be array of lengths, or scalar value.
            If not specified, avg_pi_per_sample will use the length of the
            second axis (the nucleotide axis) as the length of the sequence.
    Returns:
        A (number of samples) shaped array of average pi values per sample.
    """
    if length is None:
        length = site_pi.shape[1]
    return np.nansum(site_pi, axis=1) / length
