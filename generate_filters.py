import numpy as np
from itertools import product
from .array_manipulation import get_idx, flatten_codon
# import line_profiler, builtins
# profile = line_profiler.LineProfiler()
# builtins.__dict__['profile'] = profile

nuc_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
nuc_array = np.concatenate((np.eye(4, dtype=np.int16), np.zeros((1,4), dtype=np.int16)))
nuc_tuple = tuple(map(tuple, nuc_array))
nucs = "ACGTN"
all_codons = list(product(nuc_tuple, repeat=3))  # ensures all the weird codons (ie w/ N) are represented to avoid key errors

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
    'TGC':'C', 'TGT':'C', 'TGA':'*', 'TGG':'W'}


def codon_to_tuple(codon):
    '''safe codon-to-tuple converter (if input is already in codon format, will return codon'''
    if type(codon) == tuple:
        return codon
    return tuple(nuc_tuple[nucs.index(nuc)] for nuc in codon)


def tuple_to_codon(t):
    if type(t) == str:
        return t
    else:
        return ''.join([nucs[np.where((nuc_array == nuc).all(1))[0][0]] for nuc in t])


one_hot_translate = {codon_to_tuple(codon):AA for codon, AA in translate.items()}


def make_num_sites_dict(mutation_rates=None, include_stop_codons=False):
    numsitespercodon = {flatten_codon(codon):{0:{'s':0,'n':0},1:{'s':0,'n':0},2:{'s':0,'n':0}, 'all':{'s':0,'n':0}} for codon in all_codons}
    for codon, AA in one_hot_translate.items():
        if AA == '*' and not include_stop_codons:
            # numsitespercodon[flatten_codon(codon)][i]['s'] = 0
            # numsitespercodon[flatten_codon(codon)][i]['n'] = 0
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
                    if not include_stop_codons and new_AA == '*':  # if mutation creates stop codon, ignore (if specified)
                        continue
                    if AA == new_AA:  # if synon
                        numsitespercodon[flatten_codon(codon)][i]['s'] += mutation_likelihood
                    else:
                        numsitespercodon[flatten_codon(codon)][i]['n'] += mutation_likelihood
                num_sites_assigned = numsitespercodon[flatten_codon(codon)][i]['n'] + numsitespercodon[flatten_codon(codon)][i]['s']
                if num_sites_assigned > 0:
                    numsitespercodon[flatten_codon(codon)][i]['n'] /= num_sites_assigned
                    numsitespercodon[flatten_codon(codon)][i]['s'] /= num_sites_assigned
        numsitespercodon[flatten_codon(codon)]['all']['s'] = sum([numsitespercodon[flatten_codon(codon)][i]['s'] for i in range(3)])
        numsitespercodon[flatten_codon(codon)]['all']['n'] = sum([numsitespercodon[flatten_codon(codon)][i]['n'] for i in range(3)])
    return numsitespercodon


def make_synon_nonsynon_site_dict(num_sites_dict, include_stop_codons=False):
    '''creates dictionary to assist with calculating synon/nonsynon sites.
       output:
       [codon]:(3,4) np array of floats. 
       eg, for the codon "AAT", the array would be the number of synon/nonsynon sites in:
       [[AAT, CAT, GAT, TAT],
        [AAT, ACT, AGT, ATT],
        [AAA, AAC, AAG, AAT]]
       That way, multiplying array by the SNP frequencies will yield the number of sites in that codon
       '''
    synonSiteCount = {flatten_codon(codon): np.zeros((3, 4), dtype=np.float64) for codon in all_codons}
    nonSynonSiteCount = {flatten_codon(codon): np.zeros((3, 4), dtype=np.float64) for codon in all_codons}
    for codon in one_hot_translate.keys():
        if one_hot_translate[codon] == '*' and not include_stop_codons:
            continue
        else:
            for i, refnuc in enumerate(codon):
                for j, nuc in enumerate(nuc_tuple):
                    if (0, 0, 0, 0) in (refnuc,) + (nuc,):  # Zero out codons with N for synon/nonsynon math
                        continue
                    tmpcodon = codon[:i] + (nuc,) + codon[i + 1:]
                    synonSiteCount[flatten_codon(codon)][i, j] = num_sites_dict[flatten_codon(tmpcodon)][i]['s']
                    nonSynonSiteCount[flatten_codon(codon)][i, j] = num_sites_dict[flatten_codon(tmpcodon)][i]['n']
    return synonSiteCount, nonSynonSiteCount


def generate_codon_synon_mutation_filters():
    '''return dictionary of codons and 7x3 nonsynon filters'''
    mutationID = {'AC': 1, 'CA': 1, 'AG': 2, 'GA': 2, 'AT': 3, 'TA': 3, 'CG': 4, 'GC': 4, 'CT': 5, 'TC': 5, 'GT': 6, 'TG': 6}
    mutationID = {(nuc_tuple[nucs.index(key[0])] + nuc_tuple[nucs.index(key[1])]): code for key, code in mutationID.items()}
    # generate dictionary of codon to 3x7 nonsynon filter arrays:
    synonPiTranslate = {flatten_codon(codon): np.zeros((3, 7), dtype=bool) for codon in all_codons}
    nonSynonPiTranslate = {flatten_codon(codon): np.zeros((3, 7), dtype=bool) for codon in all_codons}
    for codon in all_codons:
        if (0, 0, 0, 0) in codon:
            continue  # Ignore Ns - codons w/ N will not be factored into synon/nonsynon math
        nonSynonPiTranslate[flatten_codon(codon)][:, 0] = 1
        synonPiTranslate[flatten_codon(codon)][:, 0] = 1
        for n, refNuc in enumerate(codon):
            for nuc in nuc_tuple:
                if nuc == refNuc or nuc == (0, 0, 0, 0):
                    continue
                altCodon = codon[:n] + (nuc,) + codon[n + 1:]
                if one_hot_translate[codon] != one_hot_translate[altCodon]:
                    nonSynonPiTranslate[flatten_codon(codon)][n, mutationID[refNuc + nuc]] = 1
                elif one_hot_translate[codon] == one_hot_translate[altCodon]:
                    synonPiTranslate[flatten_codon(codon)][n, mutationID[refNuc + nuc]] = 1
    return synonPiTranslate, nonSynonPiTranslate


def generate_coding_filters(sample_consensus_seqs, synonSiteCount, nonSynonSiteCount, idx_of_var_sites_in_gene=None, gene_name=None):
    '''given numpy array of char sequences, return:
       - #samples by #nucs by 7 synon filter array
       - same thing for nonsynon filter
       - array of synon/nonsynon counts (2 by nucs by samples)'''
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
        filters = translate_codons(sample_consensus_seqs, flatten_codon(codon), filters, synonSiteCount, nonSynonSiteCount)

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


def translate_codons(sample_consensus_seqs, codon, filters, synonSiteCount, nonSynonSiteCount):
    '''time is 2.04ms for all, even w/ conversion (conversion adds 0.1ms)'''
    ixs = get_idx(sample_consensus_seqs.reshape(-1, 12), codon)
    ixs = np.unravel_index(ixs, shape=sample_consensus_seqs.shape[:2])
    ns, s, nss, ss = filters
    ns[ixs[0], ixs[1], :, :] = nonSynonPiTranslate[codon][np.newaxis, np.newaxis, :, :]
    s[ixs[0], ixs[1], :, :] = synonPiTranslate[codon][np.newaxis, np.newaxis, :, :]
    nss[ixs[0], ixs[1], :, :] = nonSynonSiteCount[codon][np.newaxis, np.newaxis, :, :]
    ss[ixs[0], ixs[1], :, :] = synonSiteCount[codon][np.newaxis, np.newaxis, :, :]
    return ns, s, nss, ss


synonPiTranslate, nonSynonPiTranslate = generate_codon_synon_mutation_filters()
