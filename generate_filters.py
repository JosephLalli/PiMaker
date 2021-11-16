import numpy as np
from itertools import product

nuc_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
nuc_array = np.concatenate((np.eye(4, dtype=np.int16), np.zeros((1,4), dtype=np.int16)))
nuc_tuple = tuple(map(tuple, nuc_array))
nucs = "ACGTN"
all_codons = list(product(nuc_tuple, repeat=3)) # ensures all the weird codons (ie w/ N) are represented to avoid key errors

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

def make_num_sites_dict(mutation_rates=None, ignore_stop_codons=False):
    numsitespercodon = {codon:{0:{'s':0,'n':0},1:{'s':0,'n':0},2:{'s':0,'n':0}, 'all':{'s':0,'n':0}} for codon in all_codons}
    for codon, AA in one_hot_translate.items():
        if AA == '*' and ignore_stop_codons:
            numsitespercodon[codon][i]['s'] = 0
            numsitespercodon[codon][i]['n'] = 0
            continue
        else:
            for i in range(3):
                for nuc in nuc_tuple:
                    edited_codon = codon[:i] + (nuc,) + codon[i+1:]
                    if codon[i] == nuc:
                        # no need to spend time calculating the same codon
                        continue
                    if (0, 0, 0, 0) in edited_codon or (ignore_stop_codons and AA == '*'): 
                        # mutations to stop codons are optionally ignored.
                        # mutations to Ns (or codons that contain N) should be ignored.
                        continue
                    elif mutation_rates:
                        mutation_likelihood = mutation_rates[tuple_to_codon(codon[i])][nuc]
                    else:
                        mutation_likelihood = 1/3
                    new_AA = one_hot_translate[edited_codon]
                    # if ignore_stop_codons and new_AA == '*':
                    #     continue
                    if AA == new_AA: # if synon
                        numsitespercodon[codon][i]['s'] += mutation_likelihood
                    else:
                        numsitespercodon[codon][i]['n'] += mutation_likelihood
                num_sites_assigned = numsitespercodon[codon][i]['n'] + numsitespercodon[codon][i]['s']
                if num_sites_assigned > 0:
                    numsitespercodon[codon][i]['n'] /= num_sites_assigned
                    numsitespercodon[codon][i]['s'] /= num_sites_assigned
        numsitespercodon[codon]['all']['s'] = sum([numsitespercodon[codon][i]['s'] for i in range(3)])
        numsitespercodon[codon]['all']['n'] = sum([numsitespercodon[codon][i]['n'] for i in range(3)])
    return numsitespercodon

def make_synon_nonsynon_site_dict(num_sites_dict, ignore_stop_codons=False):
    '''creates dictionary to assist with calculating synon/nonsynon sites.
       output:
       [codon]:(3,4) np array of floats. 
       eg, for the codon "AAT", the array would be the number of synon/nonsynon sites in:
       [[AAT, CAT, GAT, TAT],
        [AAT, ACT, AGT, ATT],
        [AAA, AAC, AAG, AAT]]
       That way, multiplying array by the SNP frequencies will yield the number of sites in that codon
       '''
    synonSiteCount = {codon:np.zeros((3,4), dtype=np.float64) for codon in all_codons}
    nonSynonSiteCount = {codon:np.zeros((3,4), dtype=np.float64) for codon in all_codons}
    for codon in one_hot_translate.keys():
        if one_hot_translate[codon] == '*' and ignore_stop_codons:
            continue
        else:
            for i, refnuc in enumerate(codon):
                for j, nuc in enumerate(nuc_tuple):
                    if (0,0,0,0) in (refnuc,)+(nuc,): # Zero out codons with N for synon/nonsynon math
                        continue
                    tmpcodon = codon[:i] + (nuc,) + codon[i+1:]
                    synonSiteCount[codon][i,j] = num_sites_dict[tmpcodon][i]['s']
                    nonSynonSiteCount[codon][i,j] = num_sites_dict[tmpcodon][i]['n']
    return synonSiteCount, nonSynonSiteCount

def generate_codon_synon_mutation_filters():
    '''return dictionary of codons and 7x3 nonsynon filters'''
    mutationID = {'AC':1,'CA':1, 'AG':2,'GA':2, 'AT':3,'TA':3, 'CG':4,'GC':4, 'CT':5,'TC':5, 'GT':6,'TG':6}
    mutationID = {(nuc_tuple[nucs.index(key[0])] + nuc_tuple[nucs.index(key[1])]):code for key, code in mutationID.items()}
    #generate dictionary of codon to 3x7 nonsynon filter arrays:
    synonPiTranslate = {codon:np.zeros((3,7), dtype=bool) for codon in all_codons}
    nonSynonPiTranslate = {codon:np.zeros((3,7), dtype=bool) for codon in all_codons}
    for codon in all_codons:
        if (0,0,0,0) in codon:
            continue #Ignore Ns - codons w/ N will not be factored into synon/nonsynon math
        nonSynonPiTranslate[codon][:,0] = 1
        synonPiTranslate[codon][:,0] = 1
        for n, refNuc in enumerate(codon):
            for nuc in nuc_tuple:
                if nuc == refNuc or nuc == (0,0,0,0):
                    continue
                altCodon = codon[:n] + (nuc,) + codon[n+1:]
                if one_hot_translate[codon] != one_hot_translate[altCodon]:
                    nonSynonPiTranslate[codon][n, mutationID[refNuc+nuc]] = 1
                elif one_hot_translate[codon] == one_hot_translate[altCodon]:
                    synonPiTranslate[codon][n, mutationID[refNuc+nuc]] = 1
    return synonPiTranslate, nonSynonPiTranslate

