#!/usr/bin/env python
# coding: utf-8

# What am I trying to do? Trying to create something that, given a VCF/DF of SNPs, a reference sequence, and a GTF,
# it will determine the PiN/PiS by GTF category
# from typing_extensions import Concatenate
import numpy as np
from Bio import SeqIO
import pandas as pd
import os
from itertools import combinations, chain, product
from pm_io import create_combined_reference_Array, get_num_var, parseGTF, read_mutation_rates
from generate_filters import make_num_sites_dict, make_synon_nonsynon_site_dict, generate_codon_synon_mutation_filters, one_hot_translate

nuc_array = np.concatenate((np.eye(4, dtype=np.int16), np.zeros((1,4), dtype=np.int16)))
nuc_tuple = tuple(map(tuple, nuc_array))
nucs = "ACGTN"
tuple_dict = {nucs[i]:t for i,t in enumerate(nuc_tuple)}

synonPiTranslate, nonSynonPiTranslate = generate_codon_synon_mutation_filters()

mutation_rates={  #source: Pauley, Procario, Lauring 2017: A novel twelve class fluctuation test reveals higher than expected mutation rates for influenza A viruses
    'A':{'A': 1, 'C':0.41, 'G':4.19, 'T':0.16},
    'C':{'A':0.21,'C': 1, 'G':0.12,'T':0.57},
    'G':{'A':0.89,'C':0.34,'G': 1, 'T':0.75},
    'T':{'A':0.06,'C':3.83,'G':0.45, 'T': 1}
}

def calc_overlaps(gene_slices, gene_coords, bin_start, bin_end):
    reading_frames = np.zeros(shape=(bin_end-bin_start, 6), dtype=bool)
    for g, s in gene_coords.items():
        for exon in s:
            exon_read_frame = determine_read_frame(bin_start, exon[0], exon[1])
            reading_frames[min(exon):max(exon), exon_read_frame] = True
    overlaps = np.where(reading_frames.sum(1) > 1)[0]
    overlap_dict = {g:np.intersect1d(overlaps,v)[::np.sign(v[-1]-v[0])] for g,v in gene_slices.items()}
    return {k:v for k,v in overlap_dict.items() if len(v) > 0}

    # all_sites = np.hstack(list(gene_slices.values()))
    # num_sites = np.unique(all_sites, return_counts=True)[1]
    # overlap_sites = all_sites[np.where(num_sites>1)[0]]
    # overlaps = {gene:set(gene_slice).intersection(set(overlap_sites)) for gene, gene_slice in gene_slices.items()}
    # 

# def calc_overlaps(codingCoords, coords_reference='contig', ignore=[None]):
#     overlaps = dict()
#     for contig, genedict in codingCoords.items():
#         length = len(genedict)
#         if length == 1:
#             continue
#         elif any([gene in ignore for gene in genedict.keys()]):
#             for gene in ignore:
#                 try:
#                     del codingCoords[contig][gene]
#                 except KeyError:
#                     continue
#         else:
#             gene_spans = list(genedict.values())
#             one_gene_sites = set(range(0,5000))
            
#             for spans in gene_spans:
#                 gene_range = set()
#                 for start, end in spans:
#                     gene_range = gene_range.union(set(range(start, end+1)))
#                 one_gene_sites = one_gene_sites.intersection(gene_range)
#             one_gene_sites = sorted(list(one_gene_sites))
            
#             #identify overlaps in 
            
            
#             for gene, spans in genedict.items():
#                 ranges = list()
#                 start_of_gene = spans[0][0]
#                 start = start_of_gene
#                 end = 0
#                 for i in range(len(one_gene_sites)-1):
#                     if one_gene_sites[i]+1 == one_gene_sites[i+1]:
#                         continue
#                     else: #if there's a gap in one_gene_sites
#                         end = one_gene_sites[i]
#                         if coords_reference=='contig':
#                             ranges.append((start, end+1))
#                         elif coords_reference=='in_concat_gene':
#                             ranges.append((start-start_of_gene, end-start_of_gene+1))
#                         else:
#                             ranges.append((start-start_of_gene, end-start_of_gene+1))
#                         start = one_gene_sites[i+1]
#                         end = 0
#                 end = one_gene_sites[-1]
#                 if coords_reference=='contig':
#                     ranges.append((start, end+1))
#                 elif coords_reference=='in_concat_gene':
#                     ranges.append((start-start_of_gene, end-start_of_gene+1))
#                 else:
#                     ranges.append((start-start_of_gene, end-start_of_gene+1))
#                 overlaps[gene] = ranges
#     return overlaps



def range_overlap(r1, r2):
    overlap = (max(r1[0],r2[0]), min(r1[1],r2[1]))
    if overlap[0] > overlap[1]:
        return None
    else:
        return overlap

def coord_overlap(gene_coords):
    ranges = chain(*gene_coords)
    overlaps = list()
    for span_a, span_b in combinations(ranges, 2):
        overlap = range_overlap(span_a, span_b)
        if overlap:
            overlaps.append(overlap)
    return overlaps

def get_nonoverlapping_in_gene_locations(contig, gene, codingCoords):
    overlaps = coord_overlap(codingCoords[contig].values())
    coding_length=0
    non_overlaps = list()
    for region in codingCoords[contig][gene]:
        exon_overlap = coord_overlap([[region], overlaps])
        exon_overlap = range(exon_overlap[0][0], exon_overlap[0][1])
        non_overlaps.append(np.array(list(set(range(region[0],region[1])).symmetric_difference(set(exon_overlap))))-(region[0]-coding_length))
        coding_length = coding_length - region[0] + region[1]
    return np.concatenate(non_overlaps).astype(np.int32)

def one_hot_encode(ACGT_array):
    mapping = dict(zip("ACGT", range(4)))
    num_seq = np.vectorize(mapping.__getitem__)(ACGT_array)
    one_hot = np.eye(4)[num_seq]
    return one_hot

def remove_overlap_regions(overlaps, mask):
    '''given codingcoords, contig_starts, a filter and gene, return masks that will remove overlapping regions'''    
    for gene, regions in overlaps.items():
        for region in regions:
            mask[:, region[0]:region[1],:] = False
    return mask

def calcPerSamplePi(sitePi, length=None, printit=None):
    '''given numpy array of pi values, returns average per sample pi'''
    if length is None:
        length = sitePi.shape[1]
    return np.nansum(sitePi,axis=1)/length

def zero_out_region(regions, array):
    '''zero out regions of array'''
    for region in regions:
        array[:, region[0]:region[1]] = 0
    return array

def reshape_by_codon(numpyarray):
    '''ensures that arrays that are about to undergo
    codon-level manipulation are in the proper shape first'''
    if type(numpyarray) != np.ndarray:
        numpyarray = np.array(numpyarray)
    if len(numpyarray.shape) != 4:
        numpyarray = numpyarray.reshape(numpyarray.shape[0],-1,3,4)
    assert 3 in numpyarray.shape
    return numpyarray

import line_profiler, builtins
profile = line_profiler.LineProfiler()
builtins.__dict__['profile'] = profile

def flatten_codon(codon):
    return tuple([*chain.from_iterable(x if isinstance(x, tuple) else [x] for x in codon)])

def get_idx_unique(x, c, filters, ssc, nonSynonSiteCount):
    u,inv = np.unique(x,return_inverse = True, axis=0)
    return np.array([ssc[tuple(x)] for x in u])[inv].reshape(x.shape)

def get_idx_unique(codingRefSeqs, codon, filters, synonSiteCount, nonSynonSiteCount):
    '''takes (samples*seqlen)x12 ref_seqs array of type int, applies dict to that array'''
    '''143 ms!'''
    #tmp line until I fix codons
    # synonSiteCount = {flatten_codon(k):v for k, v in synonSiteCount.items()}
    # codingRefSeqs = codingRefSeqs.reshape(codingRefSeqs.shape[0:2]+(12,)).reshape(-1, 12)
    u,inv = np.unique(codingRefSeqs,return_inverse = True,axis=0)
    # u = unique_codons, inv=which codon id (aka u[ind] gets you the original array)
    # u,inv = np.unique(codingRefSeqs, return_inverse = True)
    return np.array([synonSiteCount[x] for x in u])[inv].reshape(codingRefSeqs.shape)

# def vec_translate(a, d):
#     '''takes array of codons, but codons must be bit-packed (as must dict)''' 
#     return np.vectorize(d.__getitem__)(a)

# def loop_translate(a,d):
#     n = np.ndarray(a.shape)
#     for k in d:
#         n[a == k] = d[k]
#     return n
from numba import njit
@njit
def njit_fill_in_filters(codingRefSeqs, codon):
    '''takes (samples*seqlen)x12 ref_seqs array of type int, applies dict to that array'''
    '''takes 5.7ms even w/ njit'''
    idx = np.zeros(shape=codingRefSeqs.shape[0])
    for x in range(codingRefSeqs.shape[0]):
        idx[x] = np.all(codingRefSeqs[x] == codon)
    return np.where(idx)
    ixs = np.where(np.all(codingRefSeqs==codon, axis=1))

def get_idx(codingRefSeqs, codon):
    '''takes (samples*seqlen)x12 ref_seqs array of type int, applies dict to that array'''
    '''1.85ms for ixs'''
    return np.where(np.all(codingRefSeqs==codon, axis=1))[0]

def get_idx_np_all_og(codingRefSeqs, codon, filters, synonSiteCount, nonSynonSiteCount):
    '''time is 4.5ms for ixs, 5 for all'''
    ixs = np.asarray(np.all(codingRefSeqs==codon, axis=3).all(axis=2)).nonzero()
    ns, s, nss, ss = filters
    # ns[ixs[0],ixs[1],:,:] = nonSynonPiTranslate[codon][np.newaxis,np.newaxis,:,:]
    # s[ixs[0],ixs[1],:,:] = synonPiTranslate[codon][np.newaxis,np.newaxis,:,:]
    nss[ixs[0],ixs[1],:,:] = nonSynonSiteCount[codon][np.newaxis,np.newaxis,:,:]
    # ss[ixs[0],ixs[1],:,:] = synonSiteCount[codon][np.newaxis,np.newaxis,:,:]
    return ns#ns, s, nss, ss

@profile
def translate_codons(codingRefSeqs, codon, filters, synonSiteCount, nonSynonSiteCount):
    '''time is 2.04ms for all, even w/ conversion (conversion adds 0.1ms)'''
    ixs = get_idx(codingRefSeqs.reshape(-1,12).astype(np.uint8), flatten_codon(codon))
    ixs = np.unravel_index(ixs, shape=codingRefSeqs.shape[:2])
    ns, s, nss, ss = filters
    ns[ixs[0],ixs[1],:,:] = nonSynonPiTranslate[codon][np.newaxis,np.newaxis,:,:]
    s[ixs[0],ixs[1],:,:] = synonPiTranslate[codon][np.newaxis,np.newaxis,:,:]
    nss[ixs[0],ixs[1],:,:] = nonSynonSiteCount[codon][np.newaxis,np.newaxis,:,:]
    ss[ixs[0],ixs[1],:,:] = synonSiteCount[codon][np.newaxis,np.newaxis,:,:]
    return ns,s,nss,ss


@profile
def generateSynonFilters(codingRefSeqs, synonSiteCount, nonSynonSiteCount, idx_of_var_sites_in_gene=None):
    '''given numpy array of char sequences, return:
       - #samples by #nucs by 7 synon filter array
       - same thing for nonsynon filter
       - array of synon/nonsynon counts (2 by nucs by samples)'''
    # Recieves codingRefSeqs for just one gene
    #for all potential codons, find indexes of all instances of codon
    #and put relevant filter in numpy array at that index
    codingRefSeqs = reshape_by_codon(codingRefSeqs)
    num_samples = codingRefSeqs.shape[0]
    num_codons = codingRefSeqs.shape[1]

    nonSynonFilter = np.zeros((num_samples, num_codons, 3, 7), dtype=bool)
    synonFilter = np.zeros((num_samples, num_codons, 3, 7), dtype=bool)

    nonSynonSites = np.zeros((num_samples, num_codons, 3, 4), dtype=bool)
    synonSites = np.zeros((num_samples, num_codons, 3, 4), dtype=bool)

    filters = nonSynonFilter, synonFilter, nonSynonSites, synonSites

    for codon in one_hot_translate.keys():
        filters = translate_codons(codingRefSeqs, codon, filters, synonSiteCount, nonSynonSiteCount)
        # ixs = np.asarray(np.all(codingRefSeqs==codon, axis=3).all(axis=2)).nonzero()
        # nonSynonFilter[ixs[0],ixs[1],:,:] = nonSynonPiTranslate[codon][np.newaxis,np.newaxis,:,:]
        # synonFilter[ixs[0],ixs[1],:,:] = synonPiTranslate[codon][np.newaxis,np.newaxis,:,:]
        # nonSynonSites[ixs[0],ixs[1],:,:] = nonSynonSiteCount[codon][np.newaxis,np.newaxis,:,:]
        # synonSites[ixs[0],ixs[1],:,:] = synonSiteCount[codon][np.newaxis,np.newaxis,:,:]

    nonSynonFilter = nonSynonFilter.reshape(num_samples, -1, 7)
    synonFilter = synonFilter.reshape(num_samples, -1, 7)
    nonSynonSites = nonSynonSites.reshape(num_samples, -1, 4)
    synonSites = synonSites.reshape(num_samples, -1, 4)
    
    num_const_synon_sites = 0
    num_const_nonsynon_sites = 0

    if idx_of_var_sites_in_gene is not None:
        codingRefSeqs = codingRefSeqs.reshape(codingRefSeqs.shape[0],-1,4)
        tmp_mask = np.ones(codingRefSeqs.shape, dtype=bool)
        tmp_mask[:, idx_of_var_sites_in_gene, :] = False
        num_const_nonsynon_sites = (nonSynonSites*codingRefSeqs*tmp_mask).sum(2).sum(1)
        num_const_synon_sites = (synonSites*codingRefSeqs*tmp_mask).sum(2).sum(1)
        nonSynonFilter = nonSynonFilter[:, idx_of_var_sites_in_gene, :]
        synonFilter = synonFilter[:, idx_of_var_sites_in_gene, :]
        nonSynonSites = nonSynonSites[:, idx_of_var_sites_in_gene, :]
        synonSites = synonSites[:, idx_of_var_sites_in_gene, :]
    
    return synonFilter, nonSynonFilter, synonSites, nonSynonSites, num_const_synon_sites, num_const_nonsynon_sites

def get_num_sites(allRefs, read_cts):
    '''given synon or nonsynon filter, returns
    number of synon/nonsynon sites at each nucleotide for each sample
    (return format: np.ndarray, 1D, numofsamples, values=number of sites in sample)'''
    if type(allRefs) != np.ndarray:
        allRefs = np.array(allRefs)
    if len(allRefs.shape) != 3:
        allRefs = codingRefSeqs.reshape(len(codingRefSeqs),-1,3)
    assert codingRefSeqs.shape[-1] == 3
    sites = np.sum(npfilter[:,:,1:], axis=2)/3

    if overlap_to_remove:
        overlaps = calc_overlaps(codingCoords)
        genes = overlaps.keys()
        if gene not in genes:
            return np.sum(sites,axis=1)
        ranges = overlaps[gene]
        for start, end in ranges:
            for start, end in ranges:
                sites[:,start:end] = 0
    return np.sum(sites,axis=1)

def read_cts_into_SNP_freqs(read_cts, seqArray):
    one_hot_ref = one_hot_encode(seqArray)
    SNP_freqs = np.where(read_cts.sum(axis=2, keepdims=True)==0, one_hot_ref, read_cts)
    SNP_freqs = SNP_freqs/SNP_freqs.sum(axis=2, keepdims=True)
    return SNP_freqs

def performPiCalc(read_cts):
    piCalcs = np.zeros(shape=(read_cts.shape[0:2]+(7,)), dtype=np.int32)
    piCalcs[:,:,0] = np.sum(read_cts, axis=2)
    piCalcs[:,:,1] = read_cts[:,:,0]*read_cts[:,:,1]
    piCalcs[:,:,2] = read_cts[:,:,0]*read_cts[:,:,2]
    piCalcs[:,:,3] = read_cts[:,:,0]*read_cts[:,:,3]
    piCalcs[:,:,4] = read_cts[:,:,1]*read_cts[:,:,2]
    piCalcs[:,:,5] = read_cts[:,:,1]*read_cts[:,:,3]
    piCalcs[:,:,6] = read_cts[:,:,2]*read_cts[:,:,3]
    return piCalcs

def calcPerSitePi(piCalcs):
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.sum(piCalcs[:,:,1:],axis=2)/((piCalcs[:,:,0]**2-piCalcs[:,:,0])/2)
    return np.nan_to_num(result)

def slicesFromCodingCoords(codingCoords, var_sites, genes):
    '''overall_slices: slices concated ref into gene-specific sites
       variable_slices: slices genome-wide var site array into gene-specific var sites
       idx_of_var_sites: slices gene-specific ref into gene_specific variable sites'''
    overall_slices = {gene:np.r_[tuple(np.s_[start:stop:np.sign(stop-start)] for start,stop in coords)] for gene, coords in genes.items()}
    variable_slices = {gene: np.where([p in s for p in var_sites])[0][::np.sign(s[-1]-s[0])] for gene, s in overall_slices.items()}
    idx_of_var_sites = {gene: np.where([p in var_sites for p in s])[0] for gene, s in overall_slices.items()}
    return overall_slices, variable_slices, idx_of_var_sites


def getRefSeqs(read_cts, ref_seq_array, var_index):
    '''given dataframe of VCF with variable sites and numpy refseq array,
    returns numpy array of refseqs with sample key'''
    all_ref_seqs = np.tile(ref_seq_array, (read_cts.shape[0], 1, 1)).astype(bool).copy()
    # var_ref_seqs = read_cts==(read_cts.max(2, keepdims=True))
    # using argmax ensures only one nucleotide is True
    var_ref_seqs = np.zeros(read_cts.shape, dtype=bool)
    np.put_along_axis(var_ref_seqs, np.argmax(read_cts, axis=2)[:,:, np.newaxis], True, axis=2)
    zeros = np.where(np.all(read_cts==0, axis=2))
    ref_zeros = all_ref_seqs[zeros]
    all_ref_seqs[:, var_index, :] = var_ref_seqs
    all_ref_seqs[zeros] = ref_zeros
    return all_ref_seqs
    
# %timeit np.put_along_axis(x, np.argmax(read_cts, axis=2)[:,:, np.newaxis], 2, axis=2)
@profile
def process_gene(read_cts, ref_array, muts_in_gene, synon_cts, SNP_freqs, genePiMathArray, sample_list,
                overlapping_out_of_frame_idx, gene, gene_slice, basic_gene_data):
    # Indexing notes:
    # gene_slice: index of gene sites in chunk
    # gene_var_slice: index of gene's mutable sites in list of mutable sites
    # muts_in_gene: index of gene's mutable sites in gene slice
    # so gene_slice[muts_in_gene] should equal var_pos[gene_var_slice] - it does
    # var_pos[gene_var_slice] - min(gene_slice) = in-gene-slice indexes of mutable sites
    # read_cts: whole chunk
    # refseqArray: whole chunk
    # var_pos: index of all mutable sites in chunk

    refSeqArrays = getRefSeqs(read_cts, ref_array, muts_in_gene)
    synon_filters = generateSynonFilters(refSeqArrays, *synon_cts, idx_of_var_sites_in_gene=muts_in_gene)
    synonFilter, nonSynonFilter, synonSiteFilter, nonSynonSiteFilter, num_const_synon_sites, num_const_nonsynon_sites = synon_filters

    # gene_read_cts = read_cts[:,:,gene_var_slice]
    # gene_read_cts = np.ma.array(read_cts, mask=np.broadcast_to(abs(mask-1)[np.newaxis,:,np.newaxis], read_cts.shape))
    # gene_read_cts = gene_read_cts.compressed().reshape(synonSiteFilter.shape)

    # SNP_freqs = freq_cts[:, gene_var_slice, :]
    # remove start codon from math if requested
    # if ignore_stop_codons:
    #     nonSynonSiteFilter[:,0:3,:] = 1
    #     synonSiteFilter[:,0:3,:] = 0

    nonsynon_sites = (SNP_freqs*nonSynonSiteFilter).sum(axis=2)
    synon_sites = (SNP_freqs*synonSiteFilter).sum(axis=2)

    # The number of columns to keep track of here is large, so I will fill in a numpy array to keep everything straight.
    # cols to fill in are: 'pi', 'piN', 'piS', 'N_sites', 'S_sites', 'piN_no_overlap', 'piS_no_overlap','N_sites_no_overlap', 'S_sites_no_overlap'
    chunk_gene_pi_vals = np.empty(shape=(9, len(sample_list)))
    chunk_gene_pi_vals[:] = np.nan
    # piMathMaskedArray = np.ma.array(piMath, mask=np.broadcast_to(abs(mask-1)[np.newaxis,:,np.newaxis], piMath.shape))
    # genePiMathArray = piMathMaskedArray.compressed().reshape(synonFilter.shape)
    # genePiMathArray =   piMath[]
    genePerSitePi =     calcPerSitePi(genePiMathArray)
    nonSynonPerSitePi = calcPerSitePi(genePiMathArray*nonSynonFilter)
    synonPerSitePi =    calcPerSitePi(genePiMathArray*synonFilter)
    chunk_gene_pi_vals[0] = calcPerSamplePi(genePerSitePi, length=len(gene_slice))
    chunk_gene_pi_vals[1] = calcPerSamplePi(nonSynonPerSitePi, length=nonsynon_sites.sum(axis=1)+num_const_nonsynon_sites)
    chunk_gene_pi_vals[2] = calcPerSamplePi(synonPerSitePi, length=synon_sites.sum(axis=1)+num_const_synon_sites)
    chunk_gene_pi_vals[3] = nonsynon_sites.sum(axis=1) + num_const_nonsynon_sites
    chunk_gene_pi_vals[4] = synon_sites.sum(axis=1) + num_const_synon_sites
    # print(gene, genePiN[gene])
    #And now do the same thing w/o overlapping regions to accurately determine whole-sample piN/piS
    if gene in overlapping_out_of_frame_idx.keys():
        # overlap_ix = set().union(*[set(range(start,end)) for start,end in overlaps[gene]])
        overlap_ix = overlapping_out_of_frame_idx[gene]
        # np.in1d(gene_slice, overlap_ix)[muts_in_gene] is the parts of the gene that are mutable and in the overlap_ix.
        # ~that indicates we want the mutable sites that *aren't* in that list.
        remove_overlap_sites = ~np.in1d(gene_slice, overlap_ix)[muts_in_gene]

        synonFilter_no_overlap = synonFilter[:,remove_overlap_sites,:]
        nonSynonFilter_no_overlap = nonSynonFilter[:,remove_overlap_sites,:]

        sample_gene_N_sites_no_overlap = synon_sites[:, remove_overlap_sites].sum(axis=1)
        sample_gene_S_sites_no_overlap = nonsynon_sites[:, remove_overlap_sites].sum(axis=1)
        nonSynonPerSitePi_no_overlap = calcPerSitePi(genePiMathArray[:,remove_overlap_sites,:]*nonSynonFilter_no_overlap)
        synonPerSitePi_no_overlap = calcPerSitePi(genePiMathArray[:,remove_overlap_sites,:]*synonFilter_no_overlap)

        chunk_gene_pi_vals[5] = calcPerSamplePi(nonSynonPerSitePi_no_overlap, length=sample_gene_S_sites_no_overlap)
        chunk_gene_pi_vals[6] = calcPerSamplePi(synonPerSitePi_no_overlap, length=sample_gene_N_sites_no_overlap)
        chunk_gene_pi_vals[7] = sample_gene_N_sites_no_overlap
        chunk_gene_pi_vals[8] = sample_gene_S_sites_no_overlap
    else:
        # data with no overlaps is identical to data w/ overlaps
        chunk_gene_pi_vals[5:9] = chunk_gene_pi_vals[1:5]
    return [(sample_id,)+basic_gene_data+tuple(data) for sample_id, data, in zip(sample_list, chunk_gene_pi_vals.T)]

maf = 0.01; mutation_rates = None; rollingWindow=None; synon_nonsynon=False; ignore_stop_codons=False; binsize=1e6

@profile
def calcPi(vcf_file, ref_fasta, gtf_file=None, maf = 0.01, mutation_rates = None, 
           rollingWindow=None, synon_nonsynon=False, ignore_stop_codons=False,
           binsize=1e6):
    # Main body of program
    print ("Welcome to PiMaker!")
    if mutation_rates:
        mutation_rates = read_mutation_rates(mutation_rates)

    num_sites_dict = make_num_sites_dict(mutation_rates, ignore_stop_codons)
    synon_cts = make_synon_nonsynon_site_dict(num_sites_dict, ignore_stop_codons)
    print(f'Loading reference sequence from {ref_fasta}...')
    ref_array, contig_starts, contig_coords = create_combined_reference_Array(ref_fasta)
    print(f'Loading annotation information from {gtf_file}...')
    gene_coords, transcript_to_gene_id, id_to_symbol = parseGTF(gtf_file, contig_starts)

    sample_list = get_sample_list(vcf_file)
    num_var = get_num_var(vcf_file)

    overall_sample_pi_columns = list(['sample_id','contig','chunk_id','chunk_len', 'stat_name', 'stat'])
    site_sample_pi_columns =    list()
    gene_sample_pi_columns =    list(['sample_id','contig','chunk_id', 'gene_id','gene_symbol', 'transcript_id',
                                        'transcript', 'transcript_len', 'pi', 'piN', 'piS', 'N_sites', 'S_sites',
                                        'piN_no_overlap', 'piS_no_overlap','N_sites_no_overlap', 'S_sites_no_overlap']) # sample_ids x context + all per gene stats (slightly different; not quite longform)
    overall_sample_pi = list()
    gene_sample_pi = list()
    site_sample_pi = {contig:{sample_id: list() for sample_id in sample_list} for contig in contig_starts.keys()}

    ##TODO: add MAF functionality
    ##TODO: How do I speed this up?? Grr.
    record_iterator = iterate_records(vcf_file, ref_array, contig_coords, gene_coords, num_var=num_var, binsize=int(binsize))

    for chunk_id, (read_cts, var_pos, genes, bin_start, bin_end, contig) in enumerate(record_iterator):
        if contig != '2L':
            break
        # Create lists to track results in this chunk:
        chunk_sample_pi, chunk_gene_pi, chunk_site_pi = list(), list(), list()
        # The number of columns in the gene data is unwieldly to do anything but a numpy array that I fill in
        site_sample_pi_columns.extend([var_site-contig_coords[contig][0] for var_site in var_pos])
        chunk_len = bin_end - bin_start
        basic_data = (contig, chunk_id, chunk_len)

        print(f'Calculating Pi...')
        piMath = performPiCalc(read_cts)
        with np.errstate(divide='ignore', invalid='ignore'):
            freq_cts = read_cts.astype(np.float32)/read_cts.sum(2, keepdims=True).astype(np.float32)
            freq_cts = np.nan_to_num(freq_cts)

        chunk_per_site_pi = calcPerSitePi(piMath)
        chunk_per_sample_pi = calcPerSamplePi(chunk_per_site_pi, length=(bin_end-bin_start))

        # chunk_per_site_pi_df = pd.DataFrame(chunk_per_site_pi, index=sample_list).dropna(how='all',axis=1)
        # chunk_per_sample_pi_df = pd.DataFrame(chunk_per_sample_pi, index=sample_list, columns=['pi_sample'])
        # chunk_per_site_pi_df['contig'] = contig
        # chunk_per_sample_pi_df['contig'] = contig
        # chunk_per_sample_pi_df['chunk_len'] = (bin_end-bin_start)

        # before processing gene results, record pi data
        for sample_id, chunk_pi_vals in zip(sample_list, chunk_per_site_pi):
            site_sample_pi[contig][sample_id].extend(chunk_pi_vals)
        chunk_sample_pi.extend([(sample_id,) + basic_data+('pi',) + tuple((pi,)) for sample_id, pi in zip(sample_list, chunk_per_sample_pi)])
        #contig Pi
        # perContigPiDF = pd.DataFrame(index=sample_list)

        # for contig, coords in contig_coords.items():
        #     perContigPiDF[contig] = pd.Series(calcPerSamplePi(chunk_per_site_pi[:,coords[0]:coords[1]], length = coords[1]-coords[0]), index=sample_list)
        # perContigPiDF = perContigPiDF.stack().reset_index().rename(columns={'level_0':'sampleID','level_1':'contig',0:'pi_contig'})

        # Now calculate coding regions' Pi/PiN/PiS
        ##TODO: This step takes a surprising amount of time
        gene_slices, gene_var_slices, idx_of_var_sites = slicesFromCodingCoords(contig_starts, var_pos, genes)
        #question: I have the idx of nucleotides that are in genes, and I have the idx of concated nucleotide 
        overlapping_out_of_frame_idx = calc_overlaps(gene_slices, genes, bin_start, bin_end)

        # print('Calculating PiN and PiS for each gene...')
        #create slices per gene, and pin/pis masks per gene
        for i, (gene, gene_slice) in enumerate(tqdm(gene_slices.items(), position=1)): #if already looping through genes/masks, 
            basic_gene_data = (contig, chunk_id, gene, id_to_symbol[gene], transcript_to_gene_id[gene],
                                id_to_symbol[transcript_to_gene_id[gene]], len(gene_slice))
            gene_var_slice = gene_var_slices[gene]
            muts_in_gene = idx_of_var_sites[gene]
            gene_args = read_cts[:,gene_var_slice,:], ref_array[:,gene_slice,:], muts_in_gene, synon_cts, freq_cts[:, gene_var_slice, :], piMath[:,gene_var_slice,:], sample_list, overlapping_out_of_frame_idx, gene, gene_slice, basic_gene_data
            chunk_gene_pi.extend(process_gene(*gene_args))
            if i > 20:
                break

        #Finally finally, add chunk results to list of overall results
        overall_sample_pi.extend(chunk_sample_pi)
        gene_sample_pi.extend(chunk_gene_pi)
        site_sample_pi[contig] = {sample:site_sample_pi[contig][sample]+list(data) for sample, data in zip(sample_list, chunk_per_site_pi)}
        break
    profile.print_stats()
    return 'Done'
    # Once all chunks have been processed, assemble chunks together
    pi_per_sample_df = pd.DataFrame(overall_sample_pi, columns = overall_sample_pi_columns)
    pi_per_gene_df = pd.DataFrame(gene_sample_pi, columns = gene_sample_pi_columns)

    # This is just an ugly way to do this, but the obvious, simple method of organizing this is escaping me right now
    # pi_per_site_dfs = dict of dfs, one per contig, with all sites
    pi_per_site_dfs = dict()
    for contig in site_sample_pi.keys():
        pi_per_site_dfs[contig] = pd.DataFrame().from_dict(site_sample_pi[contig], orient='index', columns = site_sample_pi_columns)

    # Stitch together chunks using groupby - note that both per_site stats and per_gene stats are, by their nature, not subject to stat 'splitting' from 
    pi_per_sample_df = pi_per_sample_df.groupby(['sample_id', 'stat_name']).apply(lambda x: np.average(x.stat, weights=x.chunk_len))
    pi_per_sample_df['piN'] = pi_per_gene_df.groupby(['sample_id']).apply(lambda x: np.average(x.piN, weights=x.N_sites)).values
    pi_per_sample_df['piS'] = pi_per_gene_df.groupby(['sample_id']).apply(lambda x: np.average(x.piS, weights=x.S_sites)).values
    pi_per_sample_df['N_sites'] = pi_per_gene_df.groupby(['sample_id']).apply(lambda x: np.sum(x.N_sites)).values
    pi_per_sample_df['S_sites'] = pi_per_gene_df.groupby(['sample_id']).apply(lambda x: np.sum(x.S_sites)).values

    pi_per_contig_df = pi_per_gene_df.groupby(['contig','sample_id']).apply(lambda x: np.average(x.piN, weights=x.N_sites)).reset_index().rename(columns={0:'piN'})
    # pi_per_contig_df['piN'] = pi_per_gene_df.groupby(['contig','sample_id']).apply(lambda x: np.average(x.piN, weights=x.N_sites)).reset_index()
    pi_per_contig_df['piS'] = pi_per_gene_df.groupby(['contig','sample_id']).apply(lambda x: np.average(x.piS, weights=x.S_sites)).values
    pi_per_contig_df['N_sites'] = pi_per_gene_df.groupby(['contig','sample_id']).apply(lambda x: np.sum(x.N_sites)).values
    pi_per_contig_df['S_sites'] = pi_per_gene_df.groupby(['contig','sample_id']).apply(lambda x: np.sum(x.S_sites)).values
    pi_per_contig_df['pi'] = np.nan
    for contig in pi_per_site_dfs.keys():
        pi_per_contig_df.loc[pi_per_contig_df.contig==contig, 'pi'] = (pi_per_site_dfs[contig].sum(1)/(contig_coords[contig][1]-contig_coords[contig][0])).values
    return pi_per_sample_df, pi_per_contig_df, pi_per_gene_df, pi_per_site_dfs


from tqdm import trange, tqdm
from collections import OrderedDict
import pysam

vcf_file = '/mnt/d/projects/pimaker/test_data/drosophilia/dest.PoolSeq.PoolSNP.001.50.10Nov2020.ann.vcf.gz'
ref_fasta = '/mnt/d/projects/pimaker/test_data/drosophilia/dmel-all-chromosome-r6.12.fasta.gz'
gtf_file = '/mnt/d/projects/pimaker/test_data/drosophilia/dmel-all-r6.12.gtf.gz'



def get_sample_list(vcf_file):
    vcf = pysam.VariantFile(vcf_file, threads = os.cpu_count()/2)
    test_record = next(vcf)
    sample_list = list(test_record.samples.keys())
    return sample_list

def iterate_records(vcf_file, ref_array, contig_coords, gene_coords, num_var=None, binsize=int(1e6)):
    '''yield chunk coordinates, mutations, genes, and gene_coordinates, ref_array chunk'''
    #create cache folder
    if not os.path.isdir('chunk_cache'):
        os.mkdir('chunk_cache')
    vcf = pysam.VariantFile(vcf_file, threads = os.cpu_count()/2)
    if num_var is None:
        num_var = get_num_var(vcf_file)
    sample_list = get_sample_list(vcf_file)
    progress = 0
    bin_start = 0
    bin_end = 0
    for contig, (contig_start, contig_end) in contig_coords.items():
        # x = np.array([x[-1][-1] for x in gene_coords['2L'].values()])
        # bins = np.digitize(x, np.arange(0, x.max(), binsize))
        bin_end = 0
        for i in trange(int((contig_end-contig_start)//binsize), total=int(num_var//binsize)):
            bin_start = i*binsize
            bin_end = np.min((binsize*(i+1)+contig_start, contig_end))
            genes, bin_end = retrieve_genes_in_region(bin_start, bin_end, gene_coords[contig])
            muts = vcf.fetch(contig=contig, start=bin_start, end=bin_end)

            #loac read_cts and var_pos from cache if available, since this takes so long
            cache_base_name = f'chunk_cache/{contig}_{bin_start}_{bin_end}'
            if os.path.exists(f'{cache_base_name}_read_cts.npy'):
                read_cts = np.load(f'{cache_base_name}_read_cts.npy')
                var_pos = np.load(f'{cache_base_name}_var_pos.npy')
            else:
                read_cts, var_pos = format_mut_array(muts, len(sample_list), bin_end-bin_start, contig_coords)
                np.save(f'{cache_base_name}_read_cts.npy', read_cts)
                np.save(f'{cache_base_name}_var_pos.npy', var_pos)
            
            # How to I keep track of start, bin_end?
            # How to I update w/ each new contig?
            # Should I yield mut_array and start,end to slice ref_array later? - No, yield relevent genes and gene_coords, use those to slice ref array
            yield read_cts, var_pos, genes, bin_start, bin_end, contig

def retrieve_genes_in_region(start, stop, gene_coords):
    '''record genes present in region.
       while recording, if pos is in middle of gene, adjust it'''
    genes = OrderedDict()
    for gene, coords in gene_coords.items():
        gene_start = coords[0][0]
        gene_end = coords[-1][-1]
        if gene_start > stop: # if gene begins after region, stop iteration.
            break
        elif gene_start >= start: # if gene begins after start of region and before end of region, record gene
            genes[gene] = coords
            if gene_end >= stop: # if the gene begins within region but ends after region, adjust end of region.
                stop = gene_end+1
    return genes, stop

def format_mut_array(muts, num_samples, chunklen, contig_coords):
    '''given iterator of pysam muts in a region,
       return array of read counts'''
    num_records = 0
    # I don't have a great way of determining the number of mutations in a chunk
    read_cts = np.zeros((num_samples, chunklen, 4),dtype=np.uint16)
    var_pos = np.zeros(chunklen, dtype=np.uint64)
    i = 0
    for r in tqdm(muts, position=1):
        if is_snp(r):
            var_pos[i] = r.pos + contig_coords[r.contig][0]
            nucs = ((r.ref,) + tuple(a for a in r.alts if len(a)==1 and a is not None))
            one_hots = np.array(list(tuple_dict[n] for n in nucs), dtype=np.uint16)
            try:
                read_cts[:,i,:] = np.stack([format_sample_cts(s, one_hots) for s in r.samples.values()])
            except Exception as e:
                print (e)
                return r, i, read_cts
            i += 1
    return read_cts[:, :i, :], var_pos[:i]

def format_sample_cts(s, one_hots):
    if s['RD'] is None:
        return np.zeros(4, dtype=np.uint16)
    else:
        alleles = s.allele_indices
        cts = np.array((s['RD'],) + s['AD'], dtype=np.uint16)[:, np.newaxis]
        if alleles[0] == alleles[1]: # if monoallelic
            alleles = alleles[0:1]
            cts = cts[alleles[0]:alleles[0]+1,:]
        elif alleles[0] != 0:
            alleles = (0,)+alleles
        s_hots = one_hots[alleles,:]
        return (cts*s_hots).sum(0)

def is_monoallelic(alleles):
    return alleles[0] == alleles[1]

def is_snp(r):
    '''determines if a pysam variant record is a snp or not'''
    return (len(r.ref) == 1) and any([len(a)==1 for a in r.alts]) and (r.ref is not None)

def create_allele_idx_to_nuc_idx_dict():
    possible_keys = chain.from_iterable([permutations('ACGT', r) for r in range(2, len(s)+1)])
    return {k:tuple('ACGT'.index(i) for i in k) for k in possible_keys}


def determine_read_frame(chrm_ref_idx, start, end):
    '''read frame notation: 0,1,2 = fwd1, fwd2, fwd3
                            3,4,5 = rev1, rev2, rev3'''
    fwd_rev_adjust = 0
    if np.sign(end-start) < 0:
        fwd_rev_adjust = 3
    return (start-chrm_ref_idx)%3+fwd_rev_adjust