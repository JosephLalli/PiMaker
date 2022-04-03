#!/usr/bin/env python
# coding: utf-8

"""
Run PiMaker and organize output.

This module contains the central calcPi function, which calls functions to load
input into memory, run data processing functions in multiple processes, and then
compiling results into summary dataframes.

"""

import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
import fileio
import filters
import chunk
from timeit import default_timer as timer
import dask


def make_pi(vcf_file, ref_fasta, gtf_file=None,
            output_file_prefix='pimaker/results', maf=0.01,
            mutation_rates=None, include_stop_codons=False,
            rolling_window=None, pi_only=False, chunk_size=int(1e6),
            cache_folder=None, num_processes=1, return_results=False,
            return_csv=False):
    """
    Calculates pi, piN, and piS. Implements PiMaker algorithm.

    Args:
        vcf_file:
            Path to the vcf file of variants. VCF file should contain all
            samples/pools sequenced.
        ref_fasta:
            Fasta file containing the reference sequence of all contigs. The
            names of each contig should be identical to the contig names in the
            vcf file.
        gtf_file:
            Optional, default None. Path to the gtf file or gff3 file that
            contains the coordinates of all coding regions. Necessary to
            calculate piN and piS values.
        output_file_prefix:
            Optional, default 'pimaker/results'. Specifies the folder to which
            results will be saved, and any prefix to be applied to result files
            from this run. For example, the default value will save sample
            summary statistics to the current working directory +
            '/pimaker/results_sample.csv'.
        maf:
            Optional, default 0.01. Minimum allele frequency of variants to
            be considered. For each sample/pool, if a variant's within-sample
            frequency is less than this value, that variant will be zeroed out
            in that sample/pool.
        mutation_rates:
            Optional, default None. Either a 4x4 array-like object containing
            relative mutation rates from ACGT (rows) to ACGT (columns), or a
            path to a file containing that object. Allows users to incorporate
            empirically determined mutation rates (or relative
            transversion/transition rates) when determining the number of
            synonymous or nonsynonymous sites per sample. (Implements
            `Ina (1995)`_ method of site count adjustment.)
        include_stop_codons:
            If True, will count mutations to or from stop
            codons as nonsynonymous. Most methods of calculating synon/nonsynon
            sites assume nonsense mutations are always evolutionary dead ends,
            and thus should be ignored. In some situations, that may not be the
            case. Optional, default False.
        rolling_window:
            Optional, default None. Currently unimplemented.
        pi_only:
            Optional, defualt False. Currently unimplemented.
        chunk_size:
            Optional, default 1e6. Number of nucleotides of each contig's
            reference sequence to be processed per chunk. If 0, each contig
            will be placed into the chunk queue as one chunk.
        cache_folder: 
            Optional, default None. Folder for PiMaker to save cached,
            processed variant data.
        num_processes:
            Optional, default 1. Number of processes running
            :func:`process_chunk`.
        return_results:
            If True, returns three pandas dataframe containing summary results
            for each sample, for each contig, and for each gene.
        return_csv:
            If True, returns the paths of the .csv files where the summary
            results for each sample, for each contig, and for each gene were
            saved.
    Returns:
        By default, returns None.

        If return_results is True, returns three pandas dataframe containing
        summary results for each sample, for each contig, and for each gene.

        If return_csv, returns the paths of the .csv files where the summary
        results for each sample, for each contig, and for each gene were saved.
    """
    print ("Welcome to PiMaker!")
    output_file_prefix, cache_folder = fileio.setup_environment(output_file_prefix,
                                                                cache_folder=cache_folder)

    if mutation_rates:
        mutation_rates = fileio.read_mutation_rates(mutation_rates)

    num_sites_dict = filters.make_num_sites_dict(mutation_rates, include_stop_codons)
    synon_cts = filters.make_synon_nonsynon_site_dict(num_sites_dict, include_stop_codons)

    print (f'Loading reference sequence from {ref_fasta}...')
    ref_array, contig_starts, contig_coords = fileio.load_references(ref_fasta)

    sample_list = fileio.get_sample_list(vcf_file)
    num_var = fileio.get_num_var(vcf_file)

    print (f'Loading annotation information from {gtf_file}...')
    gene_coords, transcript_to_gene_id, id_to_symbol = fileio.parse_gtf(gtf_file, contig_starts)

    # Now that data is loaded, set up multiprocessing queues
    chunk_queue = mp.Queue(num_processes * 2)
    results_queue = mp.Queue(num_processes * 2)
    tqdm_lock = tqdm.get_lock()

    # Create process to load queue with chunks of data
    iterator_args = vcf_file, ref_array, contig_coords, gene_coords, chunk_queue, tqdm_lock
    iterator_kwargs = {'num_var': num_var, 'num_processes': num_processes, 'chunk_size': chunk_size, 'maf': maf}
    iterator_process = mp.Process(target=chunk.iterate_records, args=iterator_args, kwargs=iterator_kwargs)
    iterator_process.start()

    # Create processes to process chunks of data
    chunk_args = (chunk_queue, results_queue, sample_list, id_to_symbol, transcript_to_gene_id, synon_cts, tqdm_lock)
    executors = [mp.Process(target=chunk.process_chunk, args=chunk_args + (i,)) for i in range(num_processes)]
    for executor in executors:
        executor.start()

    # Set up lists to collect the locations of result files
    result_files_sample_pi = list()
    result_files_genes_pi = list()
    result_files_sites_pi = {contig: list() for contig in contig_starts.keys()}

    # Wait for results to come in, save to HDD to save memory and store result file locations
    none_tracker = 0
    while True:
        result = results_queue.get()
        if result is None:
            none_tracker += 1
            if none_tracker == num_processes:
                break
        else:
            sample_filename, gene_filename, site_filename = fileio.save_tmp_chunk_results(*result, cache_folder)
            result_files_sample_pi.append(sample_filename)
            result_files_genes_pi.append(gene_filename)
            result_files_sites_pi[result[0]].append(site_filename)

    # Join and close processes before moving on
    iterator_process.join()
    iterator_process.close()
    for executor in executors:
        executor.join()
        executor.close()

    # If the user wants the raw output files, return those locations.
    if return_csv:
        return result_files_sample_pi, result_files_genes_pi, result_files_sites_pi

    # Else, compile formatted, collated reports
    print ('Making gene reports...')
    pi_per_gene_df = pd.DataFrame()
    for gene_filename in result_files_genes_pi:
        pi_per_gene_df = pi_per_gene_df.append(pd.read_csv(gene_filename))

    def _weighted_avg_nonsynon(x):
        return np.average(x.piN_no_overlap, weights=x.N_sites_no_overlap)

    def _weighted_avg_synon(x):
        return np.average(x.piS_no_overlap, weights=x.S_sites_no_overlap)

    # these are very large files. calc the things we need for the sample and contig dataframes
    # save and close right away.
    # At the moment these statistics are calculated from the first transcript in a gene.
    first_transcripts = pi_per_gene_df.groupby(['gene_id', 'sample_id']).first()
    transcripts_per_sample = first_transcripts.groupby(['sample_id'])
    transcripts_per_contig = first_transcripts.groupby(['contig', 'sample_id'])

    per_sample_piN = transcripts_per_sample.apply(_weighted_avg_nonsynon).values
    per_sample_piS = transcripts_per_sample.apply(_weighted_avg_synon).values

    per_sample_N_sites = transcripts_per_sample.apply(lambda x: np.sum(x.N_sites_no_overlap)).values
    per_sample_S_sites = transcripts_per_sample.apply(lambda x: np.sum(x.S_sites_no_overlap)).values

    per_contig_piN = transcripts_per_contig.apply(_weighted_avg_nonsynon).reset_index()
    per_contig_piS = transcripts_per_contig.apply(_weighted_avg_synon).reset_index()

    per_contig_N_sites = transcripts_per_contig.apply(lambda x: np.sum(x.N_sites_no_overlap)).reset_index()
    per_contig_S_sites = transcripts_per_contig.apply(lambda x: np.sum(x.S_sites_no_overlap)).reset_index()

    # remove the tmp per-chunk files and save main results file
    for gene_filename in result_files_genes_pi:
        os.remove(gene_filename)

    pi_per_gene_df.to_csv(output_file_prefix + "_genes.csv")
    if not return_results:
        del pi_per_gene_df

    print ('Making sample report...')
    pi_per_sample_df = pd.DataFrame()
    for sample_filename in result_files_sample_pi:
        pi_per_sample_df = pi_per_sample_df.append(pd.read_csv(sample_filename))

    # Stitch together chunks using groupby.
    # note that both per_site stats and per_gene stats are not going to be
    # affected by chunking, since chunks are always divided in intergenic
    # regions
    per_sample_and_stat = pi_per_sample_df.groupby(['sample_id', 'stat_name'])
    pi_per_sample_df = per_sample_and_stat.apply(lambda x: np.average(x.stat, weights=x.chunk_len))
    pi_per_sample_df['piN'] = per_sample_piN
    pi_per_sample_df['piS'] = per_sample_piS
    pi_per_sample_df['N_sites'] = per_sample_N_sites
    pi_per_sample_df['S_sites'] = per_sample_S_sites

    for sample_filename in result_files_sample_pi:
        os.remove(sample_filename)

    pi_per_sample_df.to_csv(output_file_prefix + "_samples.csv")
    if not return_results:
        del pi_per_sample_df

    print ('Making contig report...')
    contig_dfs = pd.DataFrame()
    contig_pis = {contig: None for contig in result_files_sites_pi.keys()}
    contig_lengths = {contig: (contig_coords[contig][1] - contig_coords[contig][0])
                      for contig in contig_coords.keys()}

    for contig, site_chunk_filenames in result_files_sites_pi.items():
        site_pi_df = pd.read_csv(site_chunk_filenames[0])
        for filename in site_chunk_filenames[1:]:
            df = pd.read_csv(filename)
            site_pi_df = site_pi_df.merge(df, on=['sample_id', 'contig', 'stat_name'])

        site_pi_df.to_csv(output_file_prefix + f'_{contig}_sites.csv.gz', compression='gzip')
        contig_pis[contig] = site_pi_df.iloc[:, 3:].sum(1) / contig_lengths[contig]

        # delete contig's per-site data and clear per-site tmp files right away,
        # since this is a huge file
        del site_pi_df
        for filename in site_chunk_filenames:
            os.remove(filename)

        # Now, process per-contig data
        sample_contig_stats = [{'sample_id': sample_id, 'contig': contig, 'contig_length': contig_lengths[contig], 'pi': pi}
                               for sample_id, pi in zip(sample_list, contig_pis[contig])]
        pi_per_contig_df = pd.DataFrame(sample_contig_stats)
        pi_per_contig_df = pi_per_contig_df.merge(per_contig_piN, on=['contig', 'sample_id'], how='left').rename(columns={0: 'piN'})
        pi_per_contig_df = pi_per_contig_df.merge(per_contig_piS, on=['contig', 'sample_id'], how='left').rename(columns={0: 'piS'})
        pi_per_contig_df = pi_per_contig_df.merge(per_contig_N_sites, on=['contig', 'sample_id'], how='left').rename(columns={0: 'N_sites'})
        pi_per_contig_df = pi_per_contig_df.merge(per_contig_S_sites, on=['contig', 'sample_id'], how='left').rename(columns={0: 'S_sites'})

        pi_per_contig_df.to_csv(output_file_prefix + f'_{contig}_summary_statistics.csv.gz', compression='gzip')
        if not return_results:
            del pi_per_contig_df
        else:
            contig_dfs = contig_dfs.append(pi_per_contig_df)

    # Finally, return results or filenames of results
    if return_results:
        return pi_per_sample_df, contig_dfs, pi_per_gene_df
    else:
        return None


def main(args=None):
    """
    Main function called during command line use of PiMaker. Obtains command
    line arguments, and runs :func:makepi. 
    """
    print('Welcome to PiMaker!')
    parser = args.get_parser()
    arg_name_dict = {'vcf': 'vcf_file', 'gtf': 'gtf_file', 'threads': 'num_processes', 'output': 'output_file_prefix'}
    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()
    else:
        args = parser.parse_args()
        args = vars(args)
        args = {arg_name_dict.get(k, k): v for k, v in args.items()}
    print (args)
    print ('\n')
    start = timer()
    with open('log.txt', 'a') as f:
        f.write(f'{start}\n')
    make_pi(**args)
    stop = timer()

    print(stop - start)
    with open('log.txt', 'a') as f:
        f.write(f'{stop}\n')
        f.write(f'{stop - start}\n')
    return None


if __name__ == '__main__':
    main()
