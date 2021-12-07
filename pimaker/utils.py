"""
Contains miscellaneous utilities to perform advanced numpy indexing.

This module is a grab-bag of various tools that are handy, especially when
using advanced indexing to create new subsets of sequences or read counts.

"""

import numpy as np
from itertools import chain


def get_read_frame(ref_start, exon):
    """
    Determines the reading frame of a coding region.

    Given the starting index of the referece sequence and the coordinates of
    a coding region, determines the read frame of the coding region.

    Args:
        ref_start:
            Index of the reference sequence. Generally will be the start index
            of the chunk being processed.
        exon:
            Tuple of the start and end coordinates of a coding region.
    Returns:
        An integer between 0-6 defining the read frame of the coding region.
        Read frame notation: 0,1,2 = fwd1, fwd2, fwd3
                             3,4,5 = rev1, rev2, rev3
    """
    exon_start, exon_end = exon
    fwd_rev_adjust = 0
    if (exon_end - exon_start) < 0:
        fwd_rev_adjust = 3
    return (exon_start - ref_start) % 3 + fwd_rev_adjust


def calc_overlaps(gene_slices, gene_coords, bin_start, bin_end):
    """
    Determines the nucleotides that are in overlapping coding regions of
    different reading frames.

    Calculating the number of non/synon sites at a nucleotide that is in
    multiple reading frames is tricky. Eventually, we will implement the
    `OL Genie algorithm`_ for these sites, but for now we simply do not include
    these sites when calculating non/synon statistics.

    Args:
        gene_slices: dictionary of transcripts to numpy slice objects
            containing the reference sequence indices of the coding regions
            of that transcript.
        gene_coords: dictionary of transcripts to a list of (start, end) tuples
            containing the coordinates of the exons in that transcript.
        bin_start: Within-concatenated-reference start position of the chunk
            being processed.
        bin_end: Within-concatenated-reference end position of the chunk
            being processed.
    """
    reading_frames = np.zeros(shape=(bin_end - bin_start, 6), dtype=bool)
    for _, s in gene_coords.items():
        for exon in s:
            exon_read_frame = get_read_frame(bin_start, exon)
            reading_frames[min(exon):max(exon), exon_read_frame] = True
    overlaps = np.where(reading_frames.sum(1) > 1)[0]
    overlap_dict = {g: np.intersect1d(overlaps, v)[::np.sign(v[-1] - v[0])] for g, v in gene_slices.items()}
    return {k: v for k, v in overlap_dict.items() if len(v) > 0}


def coordinates_to_slices(var_sites, genes):
    """
    Returns different methods of indexing the variant sites in a gene.

    Args:
        var_sites:
            List of the reference sequence indices of variable sites.
        genes:
            Dictionary of transcript_ids to list of (start, end) tuples
            containing the coordinates of the exons in that transcript.
    Returns:
        A tuple containing:
            overall_slices: dictionary of transcript_ids to slices of the
                concatenated ref that are in the coding regions of that
                transcript.
            variable_slices: dictionary of transcript_ids to the indices of
                the chunk variant site array that are present in that
                transcript.
            idx_of_var_sites: dictionary of transcript_ids to the location of
                variant sites in the reference sequence of that transcript.
    """
    ##TODO: This is oddly slow, and I'd like to figure out faster ways of doing this.
    # takes no time
    overall_slices = {gene: np.r_[tuple(np.s_[start:stop:np.sign(stop - start)] for start, stop in coords)] for gene, coords in genes.items()}
    # takes 80% of time
    variable_slices = {gene: np.where([p in s for p in var_sites])[0][::np.sign(s[-1] - s[0])] for gene, s in overall_slices.items()}
    # takes 20% of time
    idx_of_var_sites = {gene: np.where([p in var_sites for p in s])[0] for gene, s in overall_slices.items()}
    return overall_slices, variable_slices, idx_of_var_sites


def calc_consensus_seqs(read_cts, ref_seq_array, var_index):
    """
    Determines the within-sample consensus sequence of each sample.

    Args:
        read_cts:
            A (# of samples x # of variant sites x 4) array of A, C, G, and T
            read counts of each sample at each variable site.
        ref_seq_array:
            A (1 x length of ref sequence x 4) one-hot encoding of the
            reference sequence.
        var_index:
            Location of variable sites in the reference sequence.
    Returns:
        A (# of samples x length of ref sequence x 4) array containing a
        one-hot encoding of the consensus sequence of each sample. Non-variable
        sites are assumed to be the reference, while variable sites are
        assigned the major allele at each site in that sample.
    """
    all_ref_seqs = np.tile(ref_seq_array, (read_cts.shape[0], 1, 1)).astype(bool).copy()
    # Fill reference
    read_cts = np.where(np.all(read_cts == 0, axis=2, keepdims=True), all_ref_seqs[:, var_index], read_cts)
    # using argmax to ensure only one nucleotide is True
    var_ref_seqs = np.zeros(read_cts.shape, dtype=bool)
    np.put_along_axis(var_ref_seqs, np.argmax(read_cts, axis=2)[:, :, np.newaxis], True, axis=2)
    all_ref_seqs[:, var_index, :] = var_ref_seqs

    return all_ref_seqs


def one_hot_encode(ACGT_array):
    """
    Given an alphanumeric list or array of nucleotides, returns a one-hot
    encoded representation of that genetic sequence.
    """
    mapping = dict(zip("ACGT", range(4)))
    num_seq = np.vectorize(mapping.__getitem__)(ACGT_array)
    one_hot = np.eye(4)[num_seq]
    return one_hot


def get_idx(sample_consensus_seqs, codon):
    """
    Given a codon, returns the location of every occurance of that codon in
    each sample.

    Args:
        sample_consensus_seqs:
            A (# of samples x # of codons x 12) array of the sample consensus
            codons for every sample for a given coding region. Each codon is
            coded as a 1 x 1 x 12 array of ones and zeros encoding the
            flattened one-hot representation of the three nucleotides composing
            that codon.
        codon:
            A flattened one-hot representation of a codon.
    Returns:
        The indices of all instances of the provided codon in the provided
        array of sample consensus sequences.
    """
    # required optimization. Longest step in piN/piS processing.
    # Current best time is 1.85ms for ixs in 10,000,000 chunk of nucs.
    return np.where(np.all(sample_consensus_seqs == codon, axis=1))[0]


def flatten_codon(codon):
    """
    Given a one hot tuple representation of a codon of the form
    ((x, x, x, x), (x, x, x, x), (x, x, x, x)), flattens it to
    (x, x, x, x, x, x, x, x, x, x, x, x).

    E.g., ACT, encoded as ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 1)), becomes
    (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1).
    """
    return tuple([*chain.from_iterable(x if isinstance(x, tuple) else [x] for x in codon)])
