"""
Contains functions to generate numpy masks for synonymous and nonsynonymous
mutations.

This module is the heart of PiMaker. It contains functions that allow PiMaker
to identify synonymous and nonsynonymous sites in a gene's sequence. It also
contains functions that use that information to create masks to zero out
synonymous or nonsynonymous sites when performing pi calculations.

Note that, as a rule, when determining the number of N or S mutations in a
reference codon, any codons that contain an unknown base (eg N) are
removed from all downstream calculations. They are assigned 0 N and S sites.

"""


import numpy as np
from itertools import product
import utils

# Including N to ensures reference sequences with missing nucleotides
# won't raise KeyErrors when determining N/S sites.
nucs = 'ACGTN'
nuc_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
nuc_array = np.concatenate((np.eye(4, dtype=np.int16),
                            np.zeros((1, 4), dtype=np.int16)))
nuc_tuple = tuple(map(tuple, nuc_array))
all_codons = list(product(nuc_tuple, repeat=3))


# Converts alphanumeric codon into corresponding one-letter amino acid code
translate = {
    'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
    'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
    'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
    'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
    'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
    'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
    'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
    'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
    'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
    'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
    'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
    'TAC':'Y', 'TAT':'Y', 'TAA':'*', 'TAG':'*',
    'TGC':'C', 'TGT':'C', 'TGA':'*', 'TGG':'W'
    }


def codon_to_tuple(codon):
    """
    Converts an alphanumeric three-nucleotide codon into a one-hot encoded
    tuple representation of a codon. Performs 'safe' conversion - if the
    input is already a tuple, will return the one-hot tuple input.
    """
    if type(codon) == tuple:
        return codon
    return tuple(nuc_tuple[nucs.index(nuc)] for nuc in codon)


def tuple_to_codon(t):
    """
    Converts a tuple representation of a one-hot encoded three nucleotide
    codon into the alphanumeric codon. Performs 'safe' conversion - if the
    input is already alphanumeric, will return the original input.
    """
    if type(t) == str:
        return t
    else:
        return ''.join([nucs[np.where((nuc_array == nuc).all(1))[0][0]]
                        for nuc in t])


# A one-hot tuple representation of the translate dictionary.
# Used throughout this module.
one_hot_translate = {codon_to_tuple(codon): AA for codon, AA in translate.items()}


def make_num_sites_dict(mutation_rates=None, include_stop_codons=False):
    """
    Produces a codon:count dictionary to calculate the number of synonymous and
    nonsynonymous sites present at each site of a codon.

    dN/dS, which is the basis for piN/piS, is the observed number of
    non/synonymous mutations over the potential number of non/synonymous
    mutations. piN/piS uses the same denominator in its calculation. This
    function calculates the number of potential synonymous and nonsynonymous
    mutations at each position of each codon, and the total number of such
    sites for each codon.

    Note that `Nei and Gojobori (1986)`_ assume a equal likelihood for each
    possible mutation, and measures deviations from that equilibrium. This is
    the formula used in `Nelson and Hughes (2015)`_. However, `Ina 1995`_ noted
    that not all mutations are not equally likely. For example, transitions are
    far more common than transversions for most polymerases. Even better, some
    polymerases have had the relative mutation likelihood empirically
    determined. Not accounting for these differences tends to artificially
    reduce piN and increase piS, missing signs of positive or neutral selection.

    If the user provides a 4x4 matrix of mutation probabilities, PiMaker will
    use the `Ina 1995`_ method to calculate the number of N/S sites per codon.
    Note that the `Ina 1995`_ method does not take into account base usage biases
    such as G/C content, or codon usage biases. Future updates will incorporate
    these biases, likely using methods similar to `Yang and Nielson (2001)`_.

    Args:
        mutation_rates:
            If specified, will incorporate the likelihood of
            mutation when calculating the number of synon/nonsynon sites
            per codon, a la `Ina 1995`_. Optional, default None.
        include_stop_codons:
            If True, will count mutations to or from stop
            codons as nonsynonymous. Most methods of calculating synon/nonsynon
            sites assume nonsense mutations are always evolutionary dead ends,
            and thus should be ignored. In some situations, that may not be the
            case. Optional, default False.
    Returns:
        num_sites_per_codon:
            A dictionary of flattened one-hot tuple
            representations of codons to dictionaries of site indices (or
            'all') to dictionaries of the number of s or n sites in that codon
            at that site (or the codon as a whole, in the case of 'all').
            e.g., num_sites_per_codon[flat_tuple_one_hot_codon][2]['n'] will
            yield the number of nonsynonymous sites at the 3rd nucleotide of
            that codon.
    """
    # num_sites_per_codon is a nested dictionary of
    # flattened tuple representations of
    # codons: mutation position in codon: 's' and 'n' sites
    num_sites_per_codon = {utils.flatten_codon(codon):
                           {0: {'s':0,'n':0},
                            1: {'s':0,'n':0},
                            2: {'s':0,'n':0},
                            'all': {'s':0,'n':0}}
                            for codon in all_codons}

    for codon, AA in one_hot_translate.items():
        flat_codon = utils.flatten_codon(codon)
        if AA == '*' and not include_stop_codons:
            continue
        else:
            for i in range(3):
                for nuc in nuc_tuple:
                    edited_codon = codon[:i] + (nuc,) + codon[i+1:]
                    if codon[i] == nuc:
                        # no need to spend time calculating the same codon
                        continue
                    if (0, 0, 0, 0) in edited_codon or (not include_stop_codons and AA == '*'):
                        # mutations to stop codons are optionally ignored.
                        # mutations to Ns (or codons that contain N) should be ignored.
                        continue
                    elif mutation_rates:
                        mutation_likelihood = mutation_rates[tuple_to_codon(codon[i])][nuc]
                    else:
                        mutation_likelihood = 1/3
                    new_AA = one_hot_translate[edited_codon]
                    # if mutation creates stop codon, ignore it (if specified)
                    if not include_stop_codons and new_AA == '*':
                        continue
                    if AA == new_AA:  # if synon
                        num_sites_per_codon[flat_codon][i]['s'] += mutation_likelihood
                    else:
                        num_sites_per_codon[utils.flatten_codon(codon)][i]['n'] += mutation_likelihood
                num_sites_assigned = num_sites_per_codon[utils.flatten_codon(codon)][i]['n'] + num_sites_per_codon[utils.flatten_codon(codon)][i]['s']
                if num_sites_assigned > 0:
                    num_sites_per_codon[utils.flatten_codon(codon)][i]['n'] /= num_sites_assigned
                    num_sites_per_codon[utils.flatten_codon(codon)][i]['s'] /= num_sites_assigned
        num_sites_per_codon[utils.flatten_codon(codon)]['all']['s'] = sum([num_sites_per_codon[utils.flatten_codon(codon)][i]['s'] for i in range(3)])
        num_sites_per_codon[utils.flatten_codon(codon)]['all']['n'] = sum([num_sites_per_codon[utils.flatten_codon(codon)][i]['n'] for i in range(3)])
    return num_sites_per_codon


def make_synon_nonsynon_site_dict(num_sites_dict, include_stop_codons=False):
    """
    Creates dictionary to assist with calculating synon/nonsynon sites.

    Args:
        num_sites_dict:
            Previously made dictionary of the number of
            synon/nonsynon sites at each position of each codon, after
            accounting for user-provided mutation rates.
        include_stop_codons:
            `Nei and Gojobori (1986)`_ assume the natural likelihood of a nonsense
            mutation is 0. When include_stop_codons is False, nonsense mutations
            are ignored. However, in environments with very weak natural
            selection, or when other individuals can provide functional
            compensating copies of the gene (seen in viruses), this assumption
            may not be correct. In those instances, users should include stop
            codons in their math by setting this variable to True.

    Returns:
        Two dictionaries of codons:3x4 numpy array of floats.
        These dictionaries convert flattened tuple representations of one-hot
        encoded codons into a 3x4 array of the number of synonymous or
        nonsynonymous sites for every possible mutation from that codon.

        Each 3x4 array is indexed as codon site x nucleotide at that site.
        E.g., The flattened one-hot tuple representation of 'AAT', which is
        (1,0,0,0,1,0,0,0,0,0,0,1) would yield an array of the number of
        number of synon/nonsynon sites in these codons:
            [[AAT, CAT, GAT, TAT],
             [AAT, ACT, AGT, ATT],
             [AAA, AAC, AAG, AAT]]
        This representation of the number of sites in each codon allows us
        to determine the number of synon/nonsynon sites at any given mixed site
        By simply multiplying the 3x4 array of nucleotide frequencies at that
        codon by this 3x4 array of the number of sites in that codon.
    """
    num_synon_sites = {utils.flatten_codon(codon): np.zeros((3, 4), dtype=np.float64) for codon in all_codons}
    num_nonsynon_sites = {utils.flatten_codon(codon): np.zeros((3, 4), dtype=np.float64) for codon in all_codons}

    for codon in one_hot_translate.keys():
        if one_hot_translate[codon] == '*' and not include_stop_codons:
            continue
        else:
            for i, refnuc in enumerate(codon):
                flat_codon = utils.flatten_codon(codon)
                for j, nuc in enumerate(nuc_tuple):
                    # Codons that contain an N remain at zero sites
                    if (0, 0, 0, 0) in (refnuc,) + (nuc,):
                        continue
                    mutated_codon = codon[:i] + (nuc,) + codon[i + 1:]
                    flat_new_codon = utils.flatten_codon(mutated_codon)

                    num_synon_sites[flat_codon][i, j] = num_sites_dict[flat_new_codon][i]['s']
                    num_nonsynon_sites[flat_codon][i, j] = num_sites_dict[flat_new_codon][i]['n']
    return num_synon_sites, num_nonsynon_sites


def generate_codon_synon_mutation_filters():
    """
    Creates dictionary to assist with creating synon/nonsynon masks of arrays
    of 'pi math'.

    When calculating overall pi, the `Nelson and Hughes (2015)`_
    method requires us to calculate the product of all possible combinations
    of mutations at each site, and normalized to coverage.
    E.g., at each site, where A = # of reads w/ A at that site (etc.):

    A*C + A*G + A*T + C*G + C*T + G*T
    ---------------------------------
    (total coverage-1)*total coverage

    PiMaker calculates a numpy array (which we term 'pi math' of size
    (7 x number of variable sites) which contains the total coverage at
    column 0, and then these A*C (etc) products in columns 1 through 6.

    When calculating PiN or PiS, Nelson and Hughes (2015) has us only include
    read count products which represent mutations from the reference that are
    synonymous or nonsynonymous.

    This function creates dictionaries of flattened tuple representations of
    codons to 3x7 arrays of 0 and 1. When 3x7 chunks of pi math are multiplied
    by these arrays, products representing nonsynonymous or synonymous
    mutations will be zeroed out, ensuring that downstream math is specific to
    synonymous or nonsynonymous mutations respectively.

    Returns:
        Two dictionaries of codons:3x7 numpy arrays of zeros and ones used
        to generate pi math arrays of synonymous or nonsynonymous mutations.
    """
    mutationID = {'AC': 1, 'CA': 1, 'AG': 2, 'GA': 2,
                  'AT': 3, 'TA': 3, 'CG': 4, 'GC': 4,
                  'CT': 5, 'TC': 5, 'GT': 6, 'TG': 6}

    # Convert to tuple representations of nucleotides
    mutationID = {(nuc_tuple[nucs.index(key[0])] + nuc_tuple[nucs.index(key[1])]): code
                  for key, code in mutationID.items()}

    # generate dictionary of codon to 3x7 nonsynon filter arrays:
    synon_pimath_filter = {utils.flatten_codon(codon): np.zeros((3, 7), dtype=bool) for codon in all_codons}
    nonsynon_pimath_filter = {utils.flatten_codon(codon): np.zeros((3, 7), dtype=bool) for codon in all_codons}
    for codon in all_codons:
        if (0, 0, 0, 0) in codon:
            continue  # Ignore Ns - codons w/ N will not be factored into synon/nonsynon math
        nonsynon_pimath_filter[utils.flatten_codon(codon)][:, 0] = 1
        synon_pimath_filter[utils.flatten_codon(codon)][:, 0] = 1
        for n, refNuc in enumerate(codon):
            for nuc in nuc_tuple:
                if nuc == refNuc or nuc == (0, 0, 0, 0):
                    continue
                altCodon = codon[:n] + (nuc,) + codon[n + 1:]
                if one_hot_translate[codon] != one_hot_translate[altCodon]:
                    nonsynon_pimath_filter[utils.flatten_codon(codon)][n, mutationID[refNuc + nuc]] = 1
                elif one_hot_translate[codon] == one_hot_translate[altCodon]:
                    synon_pimath_filter[utils.flatten_codon(codon)][n, mutationID[refNuc + nuc]] = 1
    return synon_pimath_filter, nonsynon_pimath_filter


# These filters are constant, so they can just be generated upon loading
synon_pimath_filter, nonsynon_pimath_filter = generate_codon_synon_mutation_filters()


def generate_coding_filters(sample_consensus_seqs, num_synon_sites, num_nonsynon_sites, idx_of_var_sites_in_gene=None, gene_name=None):
    """
    Generates filters to calculate synonymous/nonsynoymous site counts and
    isolate synonymous/nonsynonymous pi math from one-hot representations of
    coding sequences. Also returns number of synonymous/nonsynonymous sites
    in non-variable regions of reference genome.

    Args:
        sample_consensus_seqs:
            A one-hot numpy array of size
            (number of samples) x (length of ref sequence) x (4)
            which contains one-hot representations of the within-sample
            consensus sequence of each sample.
        num_synon_sites:
            Dictionary of one-hot flattened tuple representation of three
            nucleotide codon to 3x4 array of the number of synonymous sites
            at that codon (see  :func:`make_synon_nonsynon_site_dict`)

    Returns:
        A tuple containing:
            - One/zero numpy array of shape (# samples x # variable nucs x 7) that
            indicates the location of synonymous mutations in 'pi math'
            arrays of each variable site of each sample.
            - The same array to indicate the location of nonsynonymous
            mutations.
            - Numpy array of shape (# samples x # variable nucs x 4) that
            indicate the number of synonymous sites per mutation for each
            sample.
            - The same array for nonsynonymous sites.
            - The sum of synonymous sites present at sites that do not vary in
            any sample.
            - The sum of nonsynonymous sites present at sites that do not vary in
            any sample.

    """
    # Recieves sample_consensus_seqs for just one gene
    # for all potential codons, find indexes of all instances of codon
    # and put relevant filter in numpy array at that index
    num_samples = sample_consensus_seqs.shape[0]
    diff_from_three = sample_consensus_seqs.shape[1] % 3
    if diff_from_three != 0:
        with open('log.txt', 'a') as log:
            log.write(f'{gene_name} had a length of {sample_consensus_seqs.shape[1]} nt; off by {diff_from_three}. First codon was {tuple(sample_consensus_seqs[0, :3, :].flatten())}\n')
        sample_consensus_seqs = sample_consensus_seqs[:, sample_consensus_seqs.shape[1] % 3, ...]
    sample_consensus_seqs = sample_consensus_seqs.reshape(num_samples, -1, 12).astype(np.uint8)
    num_codons = sample_consensus_seqs.shape[1]

    nonSynonFilter = np.zeros((num_samples, num_codons, 3, 7), dtype=bool)
    synonFilter = np.zeros((num_samples, num_codons, 3, 7), dtype=bool)

    nonSynonSites = np.zeros((num_samples, num_codons, 3, 4), dtype=np.float32)
    synonSites = np.zeros((num_samples, num_codons, 3, 4), dtype=np.float32)

    filters = nonSynonFilter, synonFilter, nonSynonSites, synonSites

    for codon in one_hot_translate.keys():
        filters = translate_codons(sample_consensus_seqs,
                                   utils.flatten_codon(codon),
                                   filters,
                                   num_synon_sites,
                                   num_nonsynon_sites)

    nonSynonFilter, synonFilter, nonSynonSites, synonSites = filters
    nonSynonFilter = nonSynonFilter.reshape(num_samples, -1, 7)
    synonFilter = synonFilter.reshape(num_samples, -1, 7)
    nonSynonSites = nonSynonSites.reshape(num_samples, -1, 4)
    synonSites = synonSites.reshape(num_samples, -1, 4)

    num_const_synon_sites = 0
    num_const_nonsynon_sites = 0

    if idx_of_var_sites_in_gene is not None:
        sample_consensus_seqs = sample_consensus_seqs.reshape(sample_consensus_seqs.shape[0], -1, 4)
        tmp_mask = np.ones(sample_consensus_seqs.shape, dtype=bool)
        tmp_mask[:, idx_of_var_sites_in_gene, :] = False
        num_const_nonsynon_sites = (nonSynonSites * sample_consensus_seqs * tmp_mask).sum(2).sum(1)
        num_const_synon_sites = (synonSites * sample_consensus_seqs * tmp_mask).sum(2).sum(1)
        nonSynonFilter = nonSynonFilter[:, idx_of_var_sites_in_gene, :]
        synonFilter = synonFilter[:, idx_of_var_sites_in_gene, :]
        nonSynonSites = nonSynonSites[:, idx_of_var_sites_in_gene, :]
        synonSites = synonSites[:, idx_of_var_sites_in_gene, :]

    return synonFilter, nonSynonFilter, synonSites, nonSynonSites, num_const_synon_sites, num_const_nonsynon_sites


def translate_codons(sample_consensus_seqs, codon, filters, num_synon_sites, num_nonsynon_sites):
    """
    Identifies the locations of a codon in all sequences, and updates provided
    filters with the appropriate 
    """
    '''time is 2.04ms for all, even w/ conversion (conversion adds 0.1ms)'''
    ixs = utils.get_idx(sample_consensus_seqs.reshape(-1, 12), codon)
    ixs = np.unravel_index(ixs, shape=sample_consensus_seqs.shape[:2])
    ns, s, nss, ss = filters
    ns[ixs[0], ixs[1], :, :] = nonsynon_pimath_filter[codon][np.newaxis, np.newaxis, :, :]
    s[ixs[0], ixs[1], :, :] = synon_pimath_filter[codon][np.newaxis, np.newaxis, :, :]
    nss[ixs[0], ixs[1], :, :] = num_nonsynon_sites[codon][np.newaxis, np.newaxis, :, :]
    ss[ixs[0], ixs[1], :, :] = num_synon_sites[codon][np.newaxis, np.newaxis, :, :]
    return ns, s, nss, ss
