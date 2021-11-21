#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import os
from pm_io import create_combined_reference_Array, get_num_var, get_sample_list, parseGTF, read_mutation_rates, save_tmp_chunk_results
from generate_filters import make_num_sites_dict, make_synon_nonsynon_site_dict, generate_codon_synon_mutation_filters
import multiprocessing as mp
from mp_utils import mp_iterate_records, process_chunk
import pysam
from tqdm import tqdm
# from __main__ import main
# import line_profiler, builtins
# profile = line_profiler.LineProfiler()
# builtins.__dict__['profile'] = profile

vcf_file = '/mnt/d/projects/pimaker/test_data/drosophilia/dest.PoolSeq.PoolSNP.001.50.10Nov2020.ann.vcf.gz'
ref_fasta = '/mnt/d/projects/pimaker/test_data/drosophilia/dmel-all-chromosome-r6.12.fasta.gz'
gtf_file = '/mnt/d/projects/pimaker/test_data/drosophilia/dmel-all-r6.12.gtf.gz'


nuc_array = np.concatenate((np.eye(4, dtype=np.int16), np.zeros((1, 4), dtype=np.int16)))
nuc_tuple = tuple(map(tuple, nuc_array))
nucs = "ACGTN"
tuple_dict = {nucs[i]: t for i, t in enumerate(nuc_tuple)}

synonPiTranslate, nonSynonPiTranslate = generate_codon_synon_mutation_filters()

# num_processes=1; maf = 0.01; mutation_rates = None; rollingWindow=None; synon_nonsynon=False; include_stop_codons=False; binsize=int(1e6)


mutation_rates={  #source: Pauley, Procario, Lauring 2017: A novel twelve class fluctuation test reveals higher than expected mutation rates for influenza A viruses
    'A':{'A': 1, 'C':0.41, 'G':4.19, 'T':0.16},
    'C':{'A':0.21,'C': 1, 'G':0.12,'T':0.57},
    'G':{'A':0.89,'C':0.34,'G': 1, 'T':0.75},
    'T':{'A':0.06,'C':3.83,'G':0.45, 'T': 1}
}


#@profile
def calcPi(vcf_file, ref_fasta, gtf_file=None, output_file_prefix=None, maf=0.01, mutation_rates=None,
           rolling_window=None, pi_only=True, include_stop_codons=False,
           binsize=int(1e6), cache_folder='chunk_cache', num_processes=1):
    print ("Welcome to PiMaker!")
    if mutation_rates:
        mutation_rates = read_mutation_rates(mutation_rates)

    if not os.path.isdir(cache_folder):
        os.mkdir(cache_folder)

    output_folder = '/'.join(output_file_prefix.split('/')[:-1])
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    num_sites_dict = make_num_sites_dict(mutation_rates, include_stop_codons)
    synon_cts = make_synon_nonsynon_site_dict(num_sites_dict, include_stop_codons)

    print (f'Loading reference sequence from {ref_fasta}...')
    ref_array, contig_starts, contig_coords = create_combined_reference_Array(ref_fasta)
    print (f'Loading annotation information from {gtf_file}...')
    gene_coords, transcript_to_gene_id, id_to_symbol = parseGTF(gtf_file, contig_starts)

    sample_list = get_sample_list(vcf_file)
    num_var = get_num_var(vcf_file)
                                        # sample_ids x context + all per gene stats (slightly different; not quite longform)
    overall_sample_pi = list()
    gene_sample_pi = list()
    site_sample_pi = {contig: list() for contig in contig_starts.keys()}

    chunk_result_holder = list()

    chunk_queue = mp.Queue(num_processes * 2)
    results_queue = mp.Queue(num_processes * 2)
    tqdm_lock = tqdm.get_lock()

    iterator_args = vcf_file, ref_array, contig_coords, gene_coords, chunk_queue, results_queue, tqdm_lock
    iterator_kwargs = {'num_var': num_var, 'num_processes': num_processes, 'binsize': binsize, 'maf': maf}
    iterator_process = mp.Process(target=mp_iterate_records, args=iterator_args, kwargs=iterator_kwargs)
    iterator_process.start()
    # x = mp_iterate_records(*iterator_args, **iterator_kwargs)
    chunk_args = (chunk_queue, results_queue, sample_list, contig_coords, id_to_symbol, transcript_to_gene_id, synon_cts, tqdm_lock)
    # process_chunk(*chunk_args)
    executors = [mp.Process(target=process_chunk, args=chunk_args + (i,)) for i in range(num_processes)]
    for executor in executors:
        executor.start()


    none_tracker = 0
    while True:
        result = results_queue.get()
        if result is None:
            none_tracker += 1
            if none_tracker == num_processes:
                break
        else:
            sample_filename, gene_filename, site_filename = save_tmp_chunk_results(*result, cache_folder)
            overall_sample_pi.append(sample_filename)
            gene_sample_pi.append(gene_filename)
            site_sample_pi[result[0]].append(site_filename)

    iterator_process.join()
    iterator_process.close()
    for executor in executors:
        executor.join()
        executor.close()

    pi_per_gene_df = pd.DataFrame()
    for gene_filename in overall_sample_pi:
        pi_per_gene_df = pi_per_gene_df.append(pd.read_csv(gene_filename))
    #this is a very large file. calc the things we need for the sample and contig dataframes, save and close right away.
    per_sample_piN = pi_per_gene_df.groupby(['gene_id', 'sample_id']).first().groupby(['sample_id']).apply(lambda x: np.average(x.piN_no_overlap, weights=x.N_sites_no_overlap)).values
    per_sample_piS = pi_per_gene_df.groupby(['gene_id', 'sample_id']).first().groupby(['sample_id']).apply(lambda x: np.average(x.piS_no_overlap, weights=x.S_sites_no_overlap)).values
    per_sample_N_sites = pi_per_gene_df.groupby(['gene_id', 'sample_id']).first().groupby(['sample_id']).apply(lambda x: np.sum(x.N_sites_no_overlap)).values
    per_sample_S_sites = pi_per_gene_df.groupby(['gene_id', 'sample_id']).first().groupby(['sample_id']).apply(lambda x: np.sum(x.S_sites_no_overlap)).values
    per_contig_piN = pi_per_gene_df.groupby(['gene_id', 'sample_id']).first().groupby(['contig', 'sample_id']).apply(lambda x: np.average(x.piN_no_overlap, weights=x.N_sites_no_overlap)).reset_index()
    per_contig_piS = pi_per_gene_df.groupby(['gene_id', 'sample_id']).first().groupby(['contig', 'sample_id']).apply(lambda x: np.average(x.piS_no_overlap, weights=x.S_sites_no_overlap)).reset_index()
    per_contig_N_sites = pi_per_gene_df.groupby(['gene_id', 'sample_id']).first().groupby(['contig', 'sample_id']).apply(lambda x: np.sum(x.N_sites_no_overlap)).reset_index()
    per_contig_S_sites = pi_per_gene_df.groupby(['gene_id', 'sample_id']).first().groupby(['contig', 'sample_id']).apply(lambda x: np.sum(x.S_sites_no_overlap)).reset_index()

    pi_per_gene_df.to_csv(output_file_prefix+"_genes.csv")
    del pi_per_gene_df

    pi_per_sample_df = pd.DataFrame()
    for sample_filename in overall_sample_pi:
        pi_per_sample_df = pi_per_sample_df.append(pd.read_csv(sample_filename))

    # Stitch together chunks using groupby - note that both per_site stats and per_gene stats are, by their nature, not subject to stat 'splitting' from 
    pi_per_sample_df = pi_per_sample_df.groupby(['sample_id', 'stat_name']).apply(lambda x: np.average(x.stat, weights=x.chunk_len))
    pi_per_sample_df['piN'] = per_sample_piN
    pi_per_sample_df['piS'] = per_sample_piS
    pi_per_sample_df['N_sites'] = per_sample_N_sites
    pi_per_sample_df['S_sites'] = per_sample_S_sites

    pi_per_sample_df.to_csv(output_file_prefix + "_samples.csv")
    del pi_per_sample_df

    contig_pis = {contig: None for contig in site_sample_pi.keys()}
    contig_lengths = {contig: (contig_coords[contig][1] - contig_coords[contig][0]) for contig in contig_coords.keys()}  # pi_per_sample_df.groupby(['contig', 'chunk_id']).first().groupby('contig').sum()['chunk_len'].to_dict()

    for contig, dfs in site_sample_pi.items():
        site_pi_df = pd.read_csv(dfs[0])
        for df in dfs[1:]:
            site_pi_df = site_pi_df.merge(df, on=['sample_id', 'contig', 'stat_name'])
        site_pi_df.to_csv(output_file_prefix + f'_{contig}_sites.csv')
        contig_pis[contig] = site_sample_pi[contig].iloc[:, 3:].sum(1) / contig_lengths[contig]
        del site_pi_df
        sample_contig_stats = [{'sample_id': sample_id, 'contig': contig, 'contig_length': contig_lengths[contig], 'pi': pi} for sample_id, pi in zip(sample_list, contig_pis[contig])]
        pi_per_contig_df = pd.DataFrame(sample_contig_stats)
        pi_per_contig_df['piN'] = per_contig_piN.loc[per_contig_piN.contig==contig].values
        pi_per_contig_df['piS'] = per_contig_piS.loc[per_contig_piS.contig==contig].values
        pi_per_contig_df['N_sites'] = per_contig_N_sites.loc[per_contig_N_sites.contig==contig].values
        pi_per_contig_df['S_sites'] = per_contig_S_sites.loc[per_contig_S_sites.contig==contig].values
        pi_per_contig_df.to_csv(output_file_prefix + f'_{contig}_summary_statistics.csv')
        del pi_per_contig_df

    return None

    # Once all chunks have been processed, assemble chunks together
    # pi_per_sample_df = pd.DataFrame(overall_sample_pi, columns=overall_sample_pi_columns)
    # pi_per_gene_df = pd.DataFrame(gene_sample_pi, columns=gene_sample_pi_columns)


    # contig_lengths = {contig: (contig_coords[contig][1] - contig_coords[contig][0]) for contig in contig_coords.keys()}  # pi_per_sample_df.groupby(['contig', 'chunk_id']).first().groupby('contig').sum()['chunk_len'].to_dict()
    # contig_pi = site_sample_pi[contig].iloc[:, 5:].sum(1)/contig_lengths[contig]
    # sample_contig_stats = [{'sample_id': sample_id, 'contig': contig, 'contig_length': contig_lengths[contig], 'pi': pi} for sample_id, pi in zip(sample_list, contig_pi) for contig in site_sample_pi.keys() if site_sample_pi[contig] is not None]
    # pi_per_contig_df = pd.DataFrame(sample_contig_stats)


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
    



# def iterate_records(vcf_file, ref_array, contig_coords, gene_coords, num_var=None, binsize=int(1e6)):
#     '''yield chunk coordinates, mutations, genes, and gene_coordinates, ref_array chunk'''
#     #create cache folder
    if not os.path.isdir('chunk_cache'):
        os.mkdir('chunk_cache')
    vcf = pysam.VariantFile(vcf_file, threads = os.cpu_count()/2)
    if num_var is None:
        num_var = get_num_var(vcf_file)
    sample_list = get_sample_list(vcf_file)
    progress = 0
    bin_start = 0
    bin_end = 0
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


# if __name__ == '__main__':
#     main()