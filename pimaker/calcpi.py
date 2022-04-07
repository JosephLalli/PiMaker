"""
This module contains functions related to calculating pi diversity from read counts.

Given numpy arrays of read counts at variable sites, this module implements the
formulas described in `Nelson and Hughes (2015)`_ to calculate pi diversity.

Nelson C.W., Hughes A.L. (2015) Within-host nucleotide diversity of virus
populations: Insights from next-generation sequencing. Infect. Genet. Evol., 30, 1–7.

"""

import numpy as np
import dask.array as da
import sys
sys.path.append('/mnt/d/Projects/PiMaker/pimaker')
import utils
import numba as nb

# @nb.njit(parallel=True)
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
    pi_math = np.zeros(shape=(read_cts.shape[0:2] + (6,)), dtype=np.int64)
    # pi_math[:, :, 0] = np.sum(read_cts, axis=2)
    pi_math[:, :, 1] = read_cts[:, :, 0] * read_cts[:, :, 1]
    pi_math[:, :, 2] = read_cts[:, :, 0] * read_cts[:, :, 2]
    pi_math[:, :, 3] = read_cts[:, :, 0] * read_cts[:, :, 3]
    pi_math[:, :, 4] = read_cts[:, :, 1] * read_cts[:, :, 2]
    pi_math[:, :, 5] = read_cts[:, :, 1] * read_cts[:, :, 3]
    pi_math[:, :, 0] = read_cts[:, :, 2] * read_cts[:, :, 3]
    return pi_math


# #Calc pimath with numba stencil?
# @nb.stencil
# def _calc_pi_math(read_cts):
#     """read_cts is (4,) len of nuc counts at given site"""
#     pi_math = np.zeros(7)
#     pi_math[0] = read_cts.sum()
#     pi_math[1] = read_cts[0] * read_cts[1]
#     pi_math[2] = read_cts[0] * read_cts[2]
#     pi_math[3] = read_cts[0] * read_cts[3]
#     pi_math[4] = read_cts[1] * read_cts[2]
#     pi_math[5] = read_cts[1] * read_cts[3]
#     pi_math[6] = read_cts[2] * read_cts[3]
#     return pi_math

triu_calc_pi_ix = np.triu_indices(4, k=1)
# @nb.guvectorize([(nb.int64[:], nb.int64[:])], '(i)->(j)')
@nb.njit(parallel=True)
def _triu_calc_pi_math(rc2):
    """read_cts is (4,) len of nuc counts at given site"""
    return rc2[triu_calc_pi_ix[0]] * rc2[triu_calc_pi_ix[1]]

@nb.njit(parallel=True)
def _calc_depth(x):
    return x.sum(2)

@nb.guvectorize([(nb.int64[:], nb.int64[:], nb.int64[:])], '(i),(j)->(j)', target_backend='parallel',nopython=True)
def _triu_calc_pi_math(read_cts, zeros, res):
    """read_cts is (4,) len of nuc counts at given site
        always call with np.zeros(7)"""
    res[0] = read_cts.sum()
    res[1:] = read_cts[triu_calc_pi_ix[0]] * read_cts[triu_calc_pi_ix[1]]


@nb.guvectorize([(nb.int64[:], nb.int64[:])], '(n)->()')
def _calc_depth(x, res):
    res[0] = x.sum()






@nb.njit(parallel=True)
def calc_per_site_pi(pi_math, depth):
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
        result = da.sum(pi_math[:, :, 1:], axis=2) / ((pi_math[:, :, 0]**2 - pi_math[:, :, 0]) / 2)
    return da.nan_to_num(result)


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
    return da.nansum(site_pi, axis=1) / length


def calculate_pi_with_dask(read_cts, var_pos, transcript_coords, ref_array):
    # Create lists to track results in this chunk:
    # chunk_sample_pi, chunk_transcript_pi, chunk_site_pi = list(), list(), list()

    # The number of columns in the transcript data is unwieldly to do anything but a numpy array that I fill in
    # chunk_len = bin_end - bin_start
    # basic_data = (contig, chunk_id, chunk_len)

    # print ('Calculating Pi...')
    # piMath = calcpi.calculate_pi_math(read_cts)
    piMath = da.map_blocks(calculate_pi_math, read_cts, dtype=da.Array)
    with np.errstate(divide='ignore', invalid='ignore'):
        freq_cts = np.where(read_cts.sum(axis=2, keepdims=True) == 0, ref_array[:, var_pos, :], read_cts)
        freq_cts = freq_cts / freq_cts.sum(axis=2, keepdims=True)

    per_site_pi = da.map_blocks(calc_per_site_pi, piMath, dtype=da.Array)
    per_sample_pi = da.map_blocks(avg_pi_per_sample, per_site_pi, dtype=da.Array, length=ref_array.shape[1])

    # # before processing transcript results, record pi data
    # sample_pi.extend([(sample_id,) + basic_data + ('pi',) + (pi,) for sample_id, pi in zip(sample_list, per_sample_pi)])
    # site_pi.extend([(sample_id, contig, 'pi') + tuple(pi) for sample_id, pi in zip(sample_list, per_site_pi)])
    transcript_slices, transcript_var_slices, idx_of_var_sites = utils.coordinates_to_slices(var_pos, transcript_coords)
    
