import numpy as np
from itertools import chain
from diversity_calcs import determine_read_frame


def calc_overlaps(gene_slices, gene_coords, bin_start, bin_end):
    reading_frames = np.zeros(shape=(bin_end - bin_start, 6), dtype=bool)
    for g, s in gene_coords.items():
        for exon in s:
            exon_read_frame = determine_read_frame(bin_start, exon[0], exon[1])
            reading_frames[min(exon):max(exon), exon_read_frame] = True
    overlaps = np.where(reading_frames.sum(1) > 1)[0]
    overlap_dict = {g: np.intersect1d(overlaps, v)[::np.sign(v[-1] - v[0])] for g, v in gene_slices.items()}
    return {k: v for k, v in overlap_dict.items() if len(v) > 0}

def coordinates_to_slices(var_sites, genes):
    '''overall_slices: slices concated ref into gene-specific sites
       variable_slices: slices genome-wide var site array into gene-specific var sites
       idx_of_var_sites: slices gene-specific ref into gene_specific variable sites'''
    # takes no time
    overall_slices = {gene: np.r_[tuple(np.s_[start:stop:np.sign(stop - start)] for start, stop in coords)] for gene, coords in genes.items()}
    # takes 80% of time
    variable_slices = {gene: np.where([p in s for p in var_sites])[0][::np.sign(s[-1] - s[0])] for gene, s in overall_slices.items()}
    # takes 20% of time
    idx_of_var_sites = {gene: np.where([p in var_sites for p in s])[0] for gene, s in overall_slices.items()}
    return overall_slices, variable_slices, idx_of_var_sites


def one_hot_encode(ACGT_array):
    mapping = dict(zip("ACGT", range(4)))
    num_seq = np.vectorize(mapping.__getitem__)(ACGT_array)
    one_hot = np.eye(4)[num_seq]
    return one_hot


def calc_consensus_seqs(read_cts, ref_seq_array, var_index):
    '''given dataframe of VCF with variable sites and numpy refseq array,
    returns numpy array of refseqs with sample key'''
    all_ref_seqs = np.tile(ref_seq_array, (read_cts.shape[0], 1, 1)).astype(bool).copy()
    # using argmax to ensure only one nucleotide is True
    var_ref_seqs = np.zeros(read_cts.shape, dtype=bool)
    np.put_along_axis(var_ref_seqs, np.argmax(read_cts, axis=2)[:, :, np.newaxis], True, axis=2)
    zeros = np.where(np.all(read_cts == 0, axis=2))
    ref_zeros = all_ref_seqs[zeros]
    all_ref_seqs[:, var_index, :] = var_ref_seqs
    all_ref_seqs[zeros] = ref_zeros
    return all_ref_seqs


def get_idx(sample_consensus_seqs, codon):
    '''takes (samples*seqlen)x12 ref_seqs array of type int, applies dict to that array'''
    '''1.85ms for ixs'''
    return np.where(np.all(sample_consensus_seqs == codon, axis=1))[0]


def flatten_codon(codon):
    return tuple([*chain.from_iterable(x if isinstance(x, tuple) else [x] for x in codon)])
