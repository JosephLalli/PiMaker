import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from .pm_io import create_combined_reference_Array, get_num_var, get_sample_list, parseGTF, read_mutation_rates, save_tmp_chunk_results
from .generate_filters import make_num_sites_dict, make_synon_nonsynon_site_dict, generate_codon_synon_mutation_filters
import multiprocessing as mp
from .mp_utils import mp_iterate_records, process_chunk

nuc_array = np.concatenate((np.eye(4, dtype=np.int16), np.zeros((1, 4), dtype=np.int16)))
nuc_tuple = tuple(map(tuple, nuc_array))
nucs = "ACGTN"
tuple_dict = {nucs[i]: t for i, t in enumerate(nuc_tuple)}

synonPiTranslate, nonSynonPiTranslate = generate_codon_synon_mutation_filters()


def calcPi(vcf_file, ref_fasta, gtf_file=None, output_file_prefix='pimaker/results', maf=0.01, mutation_rates=None,
           rolling_window=None, pi_only=True, include_stop_codons=False,
           binsize=int(1e6), return_results=False, cache_folder='chunk_cache', num_processes=1, return_csv=False):

    print ("Welcome to PiMaker!")
    if mutation_rates:
        mutation_rates = read_mutation_rates(mutation_rates)
    if not os.path.isdir(cache_folder):
        os.mkdir(cache_folder)

    output_folder = os.path.abspath('/'.join(output_file_prefix.split('/')[:-1]))

    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    num_sites_dict = make_num_sites_dict(mutation_rates, include_stop_codons)
    synon_cts = make_synon_nonsynon_site_dict(num_sites_dict, include_stop_codons)

    print (f'Loading reference sequence from {ref_fasta}...')
    ref_array, contig_starts, contig_coords = create_combined_reference_Array(ref_fasta)

    sample_list = get_sample_list(vcf_file)
    num_var = get_num_var(vcf_file)

    overall_sample_pi = list()
    gene_sample_pi = list()
    site_sample_pi = {contig: list() for contig in contig_starts.keys()}

    print (f'Loading annotation information from {gtf_file}...')
    gene_coords, transcript_to_gene_id, id_to_symbol = parseGTF(gtf_file, contig_starts)

    chunk_queue = mp.Queue(num_processes * 2)
    results_queue = mp.Queue(num_processes * 2)
    tqdm_lock = tqdm.get_lock()

    iterator_args = vcf_file, ref_array, contig_coords, gene_coords, chunk_queue, tqdm_lock
    iterator_kwargs = {'num_var': num_var, 'num_processes': num_processes, 'binsize': binsize, 'maf': maf}
    iterator_process = mp.Process(target=mp_iterate_records, args=iterator_args, kwargs=iterator_kwargs)
    iterator_process.start()

    chunk_args = (chunk_queue, results_queue, sample_list, contig_coords, id_to_symbol, transcript_to_gene_id, synon_cts, tqdm_lock)

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

    if return_csv:
        return overall_sample_pi, gene_sample_pi, site_sample_pi

    # try:
    print ('Making gene report...')
    pi_per_gene_df = pd.DataFrame()
    for gene_filename in gene_sample_pi:
        pi_per_gene_df = pi_per_gene_df.append(pd.read_csv(gene_filename))

    # this is a very large file. calc the things we need for the sample and contig dataframes, save and close right away.
    # At the moment these statistics are calculated from the first transcript in a gene
    per_sample_piN = pi_per_gene_df.groupby(['gene_id', 'sample_id']).first().groupby(['sample_id']).apply(lambda x: np.average(x.piN_no_overlap, weights=x.N_sites_no_overlap)).values
    per_sample_piS = pi_per_gene_df.groupby(['gene_id', 'sample_id']).first().groupby(['sample_id']).apply(lambda x: np.average(x.piS_no_overlap, weights=x.S_sites_no_overlap)).values
    per_sample_N_sites = pi_per_gene_df.groupby(['gene_id', 'sample_id']).first().groupby(['sample_id']).apply(lambda x: np.sum(x.N_sites_no_overlap)).values
    per_sample_S_sites = pi_per_gene_df.groupby(['gene_id', 'sample_id']).first().groupby(['sample_id']).apply(lambda x: np.sum(x.S_sites_no_overlap)).values
    per_contig_piN = pi_per_gene_df.groupby(['gene_id', 'sample_id']).first().groupby(['contig', 'sample_id']).apply(lambda x: np.average(x.piN_no_overlap, weights=x.N_sites_no_overlap)).reset_index()
    per_contig_piS = pi_per_gene_df.groupby(['gene_id', 'sample_id']).first().groupby(['contig', 'sample_id']).apply(lambda x: np.average(x.piS_no_overlap, weights=x.S_sites_no_overlap)).reset_index()
    per_contig_N_sites = pi_per_gene_df.groupby(['gene_id', 'sample_id']).first().groupby(['contig', 'sample_id']).apply(lambda x: np.sum(x.N_sites_no_overlap)).reset_index()
    per_contig_S_sites = pi_per_gene_df.groupby(['gene_id', 'sample_id']).first().groupby(['contig', 'sample_id']).apply(lambda x: np.sum(x.S_sites_no_overlap)).reset_index()

    # remove temp files and save main file
    for gene_filename in gene_sample_pi:
        os.remove(gene_filename)

    pi_per_gene_df.to_csv(output_file_prefix + "_genes.csv")
    if not return_results:
        del pi_per_gene_df

    print ('Making sample report...')
    pi_per_sample_df = pd.DataFrame()
    for sample_filename in overall_sample_pi:
        pi_per_sample_df = pi_per_sample_df.append(pd.read_csv(sample_filename))

    # Stitch together chunks using groupby - note that both per_site stats and per_gene stats are, by their nature, not subject to stat 'splitting'
    pi_per_sample_df = pi_per_sample_df.groupby(['sample_id', 'stat_name']).apply(lambda x: np.average(x.stat, weights=x.chunk_len))
    pi_per_sample_df['piN'] = per_sample_piN
    pi_per_sample_df['piS'] = per_sample_piS
    pi_per_sample_df['N_sites'] = per_sample_N_sites
    pi_per_sample_df['S_sites'] = per_sample_S_sites

    for sample_filename in overall_sample_pi:
        os.remove(sample_filename)

    pi_per_sample_df.to_csv(output_file_prefix + "_samples.csv")
    if not return_results:
        del pi_per_sample_df

    print ('Making contig report...')
    contig_pis = {contig: None for contig in site_sample_pi.keys()}
    contig_lengths = {contig: (contig_coords[contig][1] - contig_coords[contig][0]) for contig in contig_coords.keys()}
    contig_dfs = pd.DataFrame()
    for contig, dfs in site_sample_pi.items():
        site_pi_df = pd.read_csv(dfs[0])
        for df in dfs[1:]:
            site_pi_df = site_pi_df.merge(df, on=['sample_id', 'contig', 'stat_name'])
        site_pi_df.to_csv(output_file_prefix + f'_{contig}_sites.csv')
        contig_pis[contig] = site_pi_df.iloc[:, 3:].sum(1) / contig_lengths[contig]
        del site_pi_df
        sample_contig_stats = [{'sample_id': sample_id, 'contig': contig, 'contig_length': contig_lengths[contig], 'pi': pi}
                               for sample_id, pi in zip(sample_list, contig_pis[contig])]
        pi_per_contig_df = pd.DataFrame(sample_contig_stats)
        pi_per_contig_df = pi_per_contig_df.merge(per_contig_piN, on=['contig', 'sample_id'], how='left').rename(columns={0: 'piN'})
        pi_per_contig_df = pi_per_contig_df.merge(per_contig_piS, on=['contig', 'sample_id'], how='left').rename(columns={0: 'piS'})
        pi_per_contig_df = pi_per_contig_df.merge(per_contig_N_sites, on=['contig', 'sample_id'], how='left').rename(columns={0: 'N_sites'})
        pi_per_contig_df = pi_per_contig_df.merge(per_contig_S_sites, on=['contig', 'sample_id'], how='left').rename(columns={0: 'S_sites'})

        for tmp_df in dfs:
            os.remove(tmp_df)
        pi_per_contig_df.to_csv(output_file_prefix + f'_{contig}_summary_statistics.csv.gz', compression='gzip')
        if not return_results:
            del pi_per_contig_df
        else:
            contig_dfs = contig_dfs.append(pi_per_contig_df)

    if return_results:
        return pi_per_sample_df, contig_dfs, pi_per_gene_df
    else:
        return None
