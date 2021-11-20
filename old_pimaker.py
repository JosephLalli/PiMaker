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

# import line_profiler, builtins
# profile = line_profiler.LineProfiler()
# builtins.__dict__['profile'] = profile





#@profile
def generate_coding_filters(sample_consensus_seqs, synonSiteCount, nonSynonSiteCount, idx_of_var_sites_in_gene=None):
    '''given numpy array of char sequences, return:
       - #samples by #nucs by 7 synon filter array
       - same thing for nonsynon filter
       - array of synon/nonsynon counts (2 by nucs by samples)'''
    # Recieves sample_consensus_seqs for just one gene
    #for all potential codons, find indexes of all instances of codon
    #and put relevant filter in numpy array at that index
    sample_consensus_seqs = reshape_by_codon(sample_consensus_seqs)
    num_samples = sample_consensus_seqs.shape[0]
    num_codons = sample_consensus_seqs.shape[1]

    nonSynonFilter = np.zeros((num_samples, num_codons, 3, 7), dtype=bool)
    synonFilter = np.zeros((num_samples, num_codons, 3, 7), dtype=bool)

    nonSynonSites = np.zeros((num_samples, num_codons, 3, 4), dtype=bool)
    synonSites = np.zeros((num_samples, num_codons, 3, 4), dtype=bool)

    filters = nonSynonFilter, synonFilter, nonSynonSites, synonSites

    for codon in one_hot_translate.keys():
        filters = translate_codons(sample_consensus_seqs, flatten_codon(codon), filters, synonSiteCount, nonSynonSiteCount)
        # ixs = np.asarray(np.all(sample_consensus_seqs==codon, axis=3).all(axis=2)).nonzero()
        # nonSynonFilter[ixs[0],ixs[1],:,:] = nonSynonPiTranslate[codon][np.newaxis,np.newaxis,:,:]
        # synonFilter[ixs[0],ixs[1],:,:] = synonPiTranslate[codon][np.newaxis,np.newaxis,:,:]
        # nonSynonSites[ixs[0],ixs[1],:,:] = nonSynonSiteCount[codon][np.newaxis,np.newaxis,:,:]
        # synonSites[ixs[0],ixs[1],:,:] = synonSiteCount[codon][np.newaxis,np.newaxis,:,:]

    nonSynonFilter, synonFilter, nonSynonSites, synonSites = filters
    nonSynonFilter = nonSynonFilter.reshape(num_samples, -1, 7)
    synonFilter = synonFilter.reshape(num_samples, -1, 7)
    nonSynonSites = nonSynonSites.reshape(num_samples, -1, 4)
    synonSites = synonSites.reshape(num_samples, -1, 4)
    
    num_const_synon_sites = 0
    num_const_nonsynon_sites = 0

    if idx_of_var_sites_in_gene is not None:
        sample_consensus_seqs = sample_consensus_seqs.reshape(sample_consensus_seqs.shape[0],-1,4)
        tmp_mask = np.ones(sample_consensus_seqs.shape, dtype=bool)
        tmp_mask[:, idx_of_var_sites_in_gene, :] = False
        num_const_nonsynon_sites = (nonSynonSites*sample_consensus_seqs*tmp_mask).sum(2).sum(1)
        num_const_synon_sites = (synonSites*sample_consensus_seqs*tmp_mask).sum(2).sum(1)
        nonSynonFilter = nonSynonFilter[:, idx_of_var_sites_in_gene, :]
        synonFilter = synonFilter[:, idx_of_var_sites_in_gene, :]
        nonSynonSites = nonSynonSites[:, idx_of_var_sites_in_gene, :]
        synonSites = synonSites[:, idx_of_var_sites_in_gene, :]
    
    return synonFilter, nonSynonFilter, synonSites, nonSynonSites, num_const_synon_sites, num_const_nonsynon_sites



# %timeit np.put_along_axis(x, np.argmax(read_cts, axis=2)[:,:, np.newaxis], 2, axis=2)
#@profile
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

    refSeqArrays = calc_consensus_seqs(read_cts, ref_array, muts_in_gene)
    # mut_codons = np.unravel_index(muts_in_gene, (len(gene_slice)//3,3))[0]
    # refSeqCodons = refSeqArrays.reshape(len(sample_list), -1, 12)
    # var_codons = refSeqCodons[:,mut_codons,:]
    # synon_filters = generate_coding_filters(refSeqArrays.reshape(len(sample_list), -1, 12)[:,mut_codons,:], *synon_cts, idx_of_var_sites_in_gene=muts_in_gene)
    synon_filters = generate_coding_filters(refSeqArrays, *synon_cts, idx_of_var_sites_in_gene=muts_in_gene)
    synonFilter, nonSynonFilter, synonSiteFilter, nonSynonSiteFilter, num_const_synon_sites, num_const_nonsynon_sites = synon_filters

    # gene_read_cts = read_cts[:,:,gene_var_slice]
    # gene_read_cts = np.ma.array(read_cts, mask=np.broadcast_to(abs(mask-1)[np.newaxis,:,np.newaxis], read_cts.shape))
    # gene_read_cts = gene_read_cts.compressed().reshape(synonSiteFilter.shape)

    # SNP_freqs = freq_cts[:, gene_var_slice, :]
    # remove start codon from math if requested
    # if include_stop_codons:
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

maf = 0.01; mutation_rates = None; rollingWindow=None; synon_nonsynon=False; include_stop_codons=False; binsize=1e6

import multiprocessing as mp
from mp_utils import mp_iterate_records, process_chunk
#@profile
def calcPi(vcf_file, ref_fasta, gtf_file=None, maf = 0.01, mutation_rates = None, 
           rollingWindow=None, synon_nonsynon=False, include_stop_codons=False,
           binsize=1e6, num_processes=1):
    # Main body of program
    print ("Welcome to PiMaker!")
    if mutation_rates:
        mutation_rates = read_mutation_rates(mutation_rates)

    num_sites_dict = make_num_sites_dict(mutation_rates, include_stop_codons)
    synon_cts = make_synon_nonsynon_site_dict(num_sites_dict, include_stop_codons)

    filters = nonSynonPiTranslate, synonPiTranslate, *synon_cts

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
    
    chunk_result_holder = list()

    ##TODO: add MAF functionality
    ##TODO: How do I speed this up?? Grr.
    # record_iterator = iterate_records(vcf_file, ref_array, contig_coords, gene_coords, num_var=num_var, binsize=int(binsize))

    chunk_queue = mp.Queue()
    results_queue = np.Queue()
    iterator_args = vcf_file, ref_array, contig_coords, gene_coords, chunk_queue, results_queue
    iterator_kwargs = {'num_var':num_var, 'num_processes':num_processes, 'binsize':binsize}
    iterator_process = mp.Process(target=mp_iterate_records, args=iterator_args, kwargs=iterator_kwargs)
    iterator_process.start()

    chunk_args = (chunk_queue, results_queue, sample_list, contig_coords, contig_starts, id_to_symbol, transcript_to_gene_id, filters)
    executors = [mp.Process(target=process_chunk, args=chunk_args) for _ in range(num_processes)]

    none_tracker = 0
    while True:
        results = results_queue.get()
        if results == None:
            none_tracker += 0
            if none_tracker == num_processes:
                break
        else:
            chunk_result_holder.append(results)
            
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


    # for chunk_id, (read_cts, var_pos, genes, bin_start, bin_end, contig) in enumerate(record_iterator):
    #     if contig != '2L':
    #         break
    #     # Create lists to track results in this chunk:
    #     chunk_sample_pi, chunk_gene_pi, chunk_site_pi = list(), list(), list()
    #     # The number of columns in the gene data is unwieldly to do anything but a numpy array that I fill in
    #     site_sample_pi_columns.extend([var_site-contig_coords[contig][0] for var_site in var_pos])
    #     chunk_len = bin_end - bin_start
    #     basic_data = (contig, chunk_id, chunk_len)

    #     print(f'Calculating Pi...')
    #     piMath = performPiCalc(read_cts)
    #     with np.errstate(divide='ignore', invalid='ignore'):
    #         freq_cts = read_cts.astype(np.float32)/read_cts.sum(2, keepdims=True).astype(np.float32)
    #         freq_cts = np.nan_to_num(freq_cts)

    #     chunk_per_site_pi = calcPerSitePi(piMath)
    #     chunk_per_sample_pi = calcPerSamplePi(chunk_per_site_pi, length=(bin_end-bin_start))

    #     # chunk_per_site_pi_df = pd.DataFrame(chunk_per_site_pi, index=sample_list).dropna(how='all',axis=1)
    #     # chunk_per_sample_pi_df = pd.DataFrame(chunk_per_sample_pi, index=sample_list, columns=['pi_sample'])
    #     # chunk_per_site_pi_df['contig'] = contig
    #     # chunk_per_sample_pi_df['contig'] = contig
    #     # chunk_per_sample_pi_df['chunk_len'] = (bin_end-bin_start)

    #     # before processing gene results, record pi data
    #     for sample_id, chunk_pi_vals in zip(sample_list, chunk_per_site_pi):
    #         site_sample_pi[contig][sample_id].extend(chunk_pi_vals)
    #     chunk_sample_pi.extend([(sample_id,) + basic_data+('pi',) + tuple((pi,)) for sample_id, pi in zip(sample_list, chunk_per_sample_pi)])
    #     #contig Pi
    #     # perContigPiDF = pd.DataFrame(index=sample_list)

    #     # for contig, coords in contig_coords.items():
    #     #     perContigPiDF[contig] = pd.Series(calcPerSamplePi(chunk_per_site_pi[:,coords[0]:coords[1]], length = coords[1]-coords[0]), index=sample_list)
    #     # perContigPiDF = perContigPiDF.stack().reset_index().rename(columns={'level_0':'sampleID','level_1':'contig',0:'pi_contig'})

    #     # Now calculate coding regions' Pi/PiN/PiS
    #     ##TODO: This step takes a surprising amount of time
    #     gene_slices, gene_var_slices, idx_of_var_sites = coordinates_to_slices(contig_starts, var_pos, genes)
    #     #question: I have the idx of nucleotides that are in genes, and I have the idx of concated nucleotide 
    #     overlapping_out_of_frame_idx = calc_overlaps(gene_slices, genes, bin_start, bin_end)

    #     # print('Calculating PiN and PiS for each gene...')
    #     #create slices per gene, and pin/pis masks per gene
    #     for i, (gene, gene_slice) in enumerate(tqdm(gene_slices.items(), position=1)): #if already looping through genes/masks, 
    #         basic_gene_data = (contig, chunk_id, gene, id_to_symbol[gene], transcript_to_gene_id[gene],
    #                             id_to_symbol[transcript_to_gene_id[gene]], len(gene_slice))
    #         gene_var_slice = gene_var_slices[gene]
    #         muts_in_gene = idx_of_var_sites[gene]
    #         gene_args = read_cts[:,gene_var_slice,:], ref_array[:,gene_slice,:], muts_in_gene, synon_cts, freq_cts[:, gene_var_slice, :], piMath[:,gene_var_slice,:], sample_list, overlapping_out_of_frame_idx, gene, gene_slice, basic_gene_data
    #         chunk_gene_pi.extend(process_gene(*gene_args))

    #     #Finally finally, add chunk results to list of overall results
    #     overall_sample_pi.extend(chunk_sample_pi)
    #     gene_sample_pi.extend(chunk_gene_pi)
    #     site_sample_pi[contig] = {sample:site_sample_pi[contig][sample]+list(data) for sample, data in zip(sample_list, chunk_per_site_pi)}
    #     break
    # profile.print_stats()
    # return 'Done'
    return pi_per_sample_df, pi_per_contig_df, pi_per_gene_df, pi_per_site_dfs


from tqdm import trange, tqdm
from collections import OrderedDict
import pysam

vcf_file = '/mnt/d/projects/pimaker/test_data/drosophilia/dest.PoolSeq.PoolSNP.001.50.10Nov2020.ann.vcf.gz'
ref_fasta = '/mnt/d/projects/pimaker/test_data/drosophilia/dmel-all-chromosome-r6.12.fasta.gz'
gtf_file = '/mnt/d/projects/pimaker/test_data/drosophilia/dmel-all-r6.12.gtf.gz'



# def iterate_records(vcf_file, ref_array, contig_coords, gene_coords, num_var=None, binsize=int(1e6)):
#     '''yield chunk coordinates, mutations, genes, and gene_coordinates, ref_array chunk'''
#     #create cache folder
#     if not os.path.isdir('chunk_cache'):
#         os.mkdir('chunk_cache')
#     vcf = pysam.VariantFile(vcf_file, threads = os.cpu_count()/2)
#     if num_var is None:
#         num_var = get_num_var(vcf_file)
#     sample_list = get_sample_list(vcf_file)
#     progress = 0
#     bin_start = 0
#     bin_end = 0
#     for contig, (contig_start, contig_end) in contig_coords.items():
#         # x = np.array([x[-1][-1] for x in gene_coords['2L'].values()])
#         # bins = np.digitize(x, np.arange(0, x.max(), binsize))
#         bin_end = 0
#         for i in trange(int((contig_end-contig_start)//binsize), total=int(num_var//binsize)):
#             bin_start = i*binsize
#             bin_end = np.min((binsize*(i+1)+contig_start, contig_end))
#             genes, bin_end = retrieve_genes_in_region(bin_start, bin_end, gene_coords[contig])
#             muts = vcf.fetch(contig=contig, start=bin_start, end=bin_end)

#             #loac read_cts and var_pos from cache if available, since this takes so long
#             cache_base_name = f'chunk_cache/{contig}_{bin_start}_{bin_end}'
#             if os.path.exists(f'{cache_base_name}_read_cts.npy'):
#                 read_cts = np.load(f'{cache_base_name}_read_cts.npy')
#                 var_pos = np.load(f'{cache_base_name}_var_pos.npy')
#             else:
#                 read_cts, var_pos = format_mut_array(muts, len(sample_list), bin_end-bin_start, contig_coords)
#                 np.save(f'{cache_base_name}_read_cts.npy', read_cts)
#                 np.save(f'{cache_base_name}_var_pos.npy', var_pos)
            
#             # How to I keep track of start, bin_end?
#             # How to I update w/ each new contig?
#             # Should I yield mut_array and start,end to slice ref_array later? - No, yield relevent genes and gene_coords, use those to slice ref array
#             yield read_cts, var_pos, genes, bin_start, bin_end, contig


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

