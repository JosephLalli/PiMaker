# -*- coding: utf-8 -*-
"""
Multiprocessing-compatable data processing functions.

This module contains the functions that run in separate processes to
create chunks of data or to perform calculations on that chunk of data.

"""

import numpy as np
from tqdm import tqdm
# import pickle
import fileio
import calcpi
import filters
import utils
# from intervaltree import Interval, IntervalTree


def process_chunk(chunk_queue, result_queue, sample_list, id_to_symbol,
                  transcript_to_gene_id, synon_cts, contig_coords, tqdm_lock, process_id,
                  pi_only=False):
    """
    A worker function that takes data from queue of chunks of
    sequences/metadata, calculates all relevent statistics, and places them in
    the queue of results.

    This is a function that is designed to run in its own process. It first
    pulls chunks of sequences and associated metadata off a queue and
    calculates overall pi, piN, piS for that chunk. It then finds all
    transcripts in that chunk, and calculates the same statistics for each
    transcript. It finally organizes the results into a list of tuples of
    results with metadata, and places the results into a queue for downstream
    processing.

    Args:
        chunk_queue:
            py:class:`~multiprocessing.Queue` object into which
            iterate_record places sequences, coordinates and metadata.
        result_queue:
            py:class:`~multiprocessing.Queue` object into which
            this function places results
        sample_list:
            Sorted list of sample names.
        id_to_symbol:
            Dictionary of gene_id values and transcript_id values
            to more human-friendly gene symbols and transcript symbols.
        transcript_to_gene_id:
            Dictionary of transcript_ids to the gene_ids they code for.
        synon_cts:
            tuple of two dictionaries, both of the forn codon:number of
            synonymous and nonsynonymous (respectively) sites per mutation.
        tqdm_lock:
            py:class:`~multiprocessing.Lock` object given to tqdm to
            allow for updating each process's progress bar w/o interfering
            with other processes.
        process_id:
            ID number given to each process.
        pi_only:
            Optional, default False. If True, this function will not
            calculate piN/piS values. Greatly speeds up performance.

    Returns:
        None
    """
    tqdm.set_lock(tqdm_lock)
    with tqdm(position=(process_id + 2)) as pbar:
        while True:
            chunk = chunk_queue.get()
            if chunk is None:
                result_queue.put(None)
                pbar.close()
                pbar.reset()
                break
            # note: transcript_coords now links to a intervaltree of cds coords, not a tuple of cds coords
            read_cts, var_pos, transcript_coords, ref_array_chunk, bin_start, bin_end, contig, chunk_id = chunk
            pbar.reset()
            pbar.set_description(f'{contig}:{bin_start}-{bin_end}')
            pbar.refresh()
            # Create lists to track results in this chunk:
            chunk_sample_pi, chunk_transcript_pi, chunk_site_pi = list(), list(), list()

            # The number of columns in the transcript data is unwieldly to do anything but a numpy array that I fill in
            chunk_len = bin_end - bin_start
            basic_data = (contig, chunk_id, chunk_len)

            # print ('Calculating Pi...')
            piMath = calcpi.calculate_pi_math(read_cts)
            with np.errstate(divide='ignore', invalid='ignore'):
                freq_cts = np.where(read_cts.sum(axis=2, keepdims=True) == 0, ref_array_chunk[:, var_pos, :], read_cts)
                freq_cts = freq_cts / freq_cts.sum(axis=2, keepdims=True)

            chunk_per_site_pi = calcpi.per_site_pi(piMath)
            chunk_per_sample_pi = calcpi.avg_pi_per_sample(chunk_per_site_pi, length=(bin_end - bin_start))

            # before processing transcript results, record pi data
            chunk_sample_pi.extend([(sample_id,) + basic_data + ('pi',) + (pi,) for sample_id, pi in zip(sample_list, chunk_per_sample_pi)])
            chunk_site_pi.extend([(sample_id, contig, 'pi') + tuple(pi) for sample_id, pi in zip(sample_list, chunk_per_site_pi)])

            # Now calculate coding regions' Pi/PiN/PiS
            ##TODO: This step takes a surprising amount of time
            transcript_slices, transcript_var_slices, idx_of_var_sites = utils.coordinates_to_slices(var_pos, transcript_coords)
            oof_overlaps_idx = utils.calc_overlaps(transcript_slices, transcript_coords, bin_start, bin_end)
            try:
                sample_consensus_seqs = utils.calc_consensus_seqs(read_cts, ref_array_chunk, var_pos)
            except IndexError as exception:
                print (exception)
                print (f'\n\n\n\n\n\n\n\n\n\ncontig:{contig}, bin_start: {bin_start}, bin_end {bin_end}, var_pos start {var_pos[0]}, var_pos end {var_pos[-1]}, read_cts size {read_cts.shape}, ref_array_chunk size {ref_array_chunk.shape} errored out with an index error for incorrectly sized var pos\n\n\n\n\n\n\n\n\n\n')
                continue
            pbar.total = len(transcript_slices)
            pbar.refresh()
            for i, (transcript, transcript_slice) in enumerate(transcript_slices.items()):
                basic_gene_data = (contig, chunk_id, transcript, id_to_symbol[transcript], transcript_to_gene_id[transcript],
                                id_to_symbol[transcript_to_gene_id[transcript]], len(transcript_slice))
                transcript_var_slice = transcript_var_slices[transcript]
                muts_in_transcript = idx_of_var_sites[transcript]
                if transcript in oof_overlaps_idx.keys():
                    transcript_oof_overlaps_idx = oof_overlaps_idx[transcript]
                else:
                    transcript_oof_overlaps_idx = None
                transcript_args = (sample_consensus_seqs[:, transcript_slice, :], muts_in_transcript,
                            synon_cts, freq_cts[:, transcript_var_slice, :], piMath[:, transcript_var_slice, :],
                            sample_list, transcript_oof_overlaps_idx, transcript, transcript_slice, basic_gene_data,
                            pi_only)
                chunk_transcript_pi.extend(process_transcript(*transcript_args))
                pbar.update(1)
            pbar.refresh()
            result_queue.put((contig, chunk_id, chunk_sample_pi, chunk_transcript_pi, chunk_site_pi, var_pos + bin_start - contig_coords[contig][0]))
            # to_save = (read_cts, var_pos, transcript_coords, ref_array_chunk, bin_start, bin_end, contig,
            #            chunk_id, piMath, freq_cts, chunk_per_site_pi, chunk_per_sample_pi, chunk_sample_pi,
            #            chunk_site_pi, read_cts[:, transcript_var_slice, :],
            #            sample_consensus_seqs[:, transcript_slice, :], muts_in_transcript, synon_cts,
            #            freq_cts[:, transcript_var_slice, :], piMath[:, transcript_var_slice, :], sample_list,
            #            oof_overlaps_idx, transcript, transcript_slice, basic_gene_data, transcript_var_slice)
            # if 'NP' in contig:
            #     with open('NP.pkl', 'wb') as f:
            #         pickle.dump(to_save, f)
        pbar.clear()
    return None


def iterate_records(vcf_file, ref_array, contig_coords, transcript_coords, chunk_queue, tqdm_lock,
                    num_var=None, num_processes=1, chunk_size=int(1e6), maf=0, 
                    cache_folder='chunk_cache', FST=False):
    """
    A worker function that takes a concatenated one-hot reference sequence,
    input vcf file path, data from queue of chunks of
    sequences/metadata, calculates all relevent statistics, and places them in
    the queue of results.

    This is a function that is designed to run in its own process. It first
    pulls chunks of sequences and associated metadata off a queue and
    calculates overall pi, piN, piS for that chunk. It then finds all
    transcripts in that chunk, and calculates the same statistics for each
    transcript. It finally organizes the results into a list of tuples of
    results with metadata, and places the results into a queue for downstream
    processing.

    Args:
        vcf_file: Path to the .vcf formatted file of populations and SNV calls.
        ref_array: Concatenated one-hot n x 4 numpy array of the entire
            reference sequence.
        contig_coords: Dictionary of contig names to (start, end) tuples of
            the starting and ending coordinates of each contig in the
            concatenated reference array.
        transcript_coords: Nested dictionary of contig names to transcript ids
            to a list of tuples. Each tuple is (start, end) index in the
            concatenated reference array of each exon in the transcript.
        chunk_queue: py:class:`~multiprocessing.Queue` object into which
            iterate_record places sequences, coordinates and metadata.
        tqdm_lock: py:class:`~multiprocessing.Lock` object given to tqdm to
            allow for updating each process's progress bar w/o interfering
            with other processes.
        num_var: Optional, default None. Number of variants in vcf file.
        num_processes: Optional, default 1. Number of processes running
            :func:`process_chunk`.
        chunk_size: Optional, default 1e6. Number of nucleotides of each contig's
            reference sequence to be processed per chunk. If 0, each contig
            will be placed into the chunk queue as one chunk.
        maf: Optional, default 0. Minimum allele frequency, defined as the
            minimum per-sample minor variant frequency. Minor variants below
            this percentage of read counts will be zeroed out. Float value
            between 0 and 1.
        cache_folder: Optional, default None. Folder for PiMaker to save
            cached, processed variant data.
    Returns:
        None
    """
    tqdm.set_lock(tqdm_lock)

    ##TODO: Need to figure out how to iterate over sync files w/o loading everything into memory
    # mut_suffix = vcf_file.split('.')[-1:]
    # if 'sync' in mut_suffix:
    #     read_cts, var_pos, _ = pool_to_numpy(vcf_file)

    if num_var is None:
        num_var = fileio.get_num_var(vcf_file)

    num_var = fileio.get_num_var(vcf_file, contig=True)
    contigs_with_variants = list(num_var.keys())
    num_var = sum(list(num_var.values()))

    chunk_id = 0
    with tqdm(position=1) as pbar:
        for contig, (contig_start, contig_end) in contig_coords.items():
            if contig not in contigs_with_variants:
                pbar.update(1)
                continue
            pbar.set_description(f'Loading contig {contig}')
            pbar.refresh()
            bin_start = contig_start
            bin_end = 0
            pbar.reset(total=int((contig_end - contig_start) // chunk_size) + 1)
            for i in range(int((contig_end - contig_start) // chunk_size) + 1):
                bin_start = np.max((((i * chunk_size) + contig_start), bin_end))
                if bin_start > contig_end:
                    continue
                bin_end = np.min(((chunk_size * (i + 1) + contig_start), contig_end))
                if bin_end > contig_end:
                    bin_end = contig_end
                transcripts, bin_end = fileio.retrieve_transcripts_in_region(bin_start, bin_end, transcript_coords[contig])
                # transcripts, bin_end = fileio.retrieve_transcripts_in_region_interval_tree(bin_start, bin_end, transcript_interval_tree, gene_intervals)
                transcripts = {t: tuple((exon_start - bin_start, exon_end - bin_start)
                               for exon_start, exon_end in coords) for t, coords in transcripts.items()}

                # Code to cache loaded data. Keeping code even though allel is pretty fast now.
                # load read_cts and var_pos from cache if available, since this takes so long
                # cache_base_name = f'{cache_folder}/{contig}_{bin_start}_{bin_end}'
                # if os.path.exists(f'{cache_base_name}_read_cts.npy'):
                #     read_cts = np.load(f'{cache_base_name}_read_cts.npy')
                #     var_pos = np.load(f'{cache_base_name}_var_pos.npy')
                # else:
                #    np.save(f'{cache_base_name}_read_cts.npy', read_cts)
                #    np.save(f'{cache_base_name}_var_pos.npy', var_pos)

                # allel is 1-indexed
                region_string = f'{contig}:{bin_start - contig_coords[contig][0]+1}-{bin_end - contig_coords[contig][0]+1}'
                # print (f'\n\n{region_string}\n\n')
                read_cts, var_pos = fileio.vcf_to_numpy_array_read_cts(vcf_file, contig_coords[contig], region=region_string, maf=maf)
                if read_cts is None:  # if no mutations in region string
                    chunk_id += 1
                    pbar.update(1)
                    continue
                chunk_queue.put((read_cts, var_pos - bin_start, transcripts, ref_array[:, bin_start: bin_end+1, :], bin_start, bin_end, contig, chunk_id))
                chunk_id += 1
                pbar.update(1)
            pbar.refresh()
        pbar.clear()

    # Once out of data, put num_processes number of Nones in chunk_queue
    for _ in range(num_processes):
        chunk_queue.put(None)
    pbar.clear()
    pbar.close()

    return None


def process_transcript(sample_consensus_seqs, muts_in_transcript, synon_cts,
                       SNP_freqs, transcript_pi_math, sample_list,
                       oof_overlaps_idx, transcript, transcript_slice,
                       basic_gene_data, pi_only=False):
    """
    Caculates diversity statistics for transcripts.

    Calculates pi, piN, and piS for transcripts and returns list of tuples
    containing the results.

    Args:
        sample_consensus_seqs: A num_samples x num_var_sites x 4 one-hot array
            defining the major allele for each sample at each variable site of
            the transcript.
        muts_in_transcript: numpy slice of the variable locations in the
            transcript.
        synon_cts: tuple of two dictionaries, both of the forn codon:number
            of synonymous and nonsynonymous (respectively) sites per mutation.
        SNP_freqs: A num_samples x num_var_sites x 4 array of allele
            frequencies for each sample at each variable site in the transcript
        transcript_pi_math: A num_samples x num_var_sites x 7 array of pi math
            for the transcript
        sample_list:
            Sample ID values for each sample.
        oof_overlaps_idx:
            Numpy indices of variable sites in the transcript that lie within
            regions that overlap with another coding region that is in a
            different reading frame.
        transcript:
            Transcript ID.
        transcript_slice:
            Numpy indices of locations in chunk that are within the transcript.
        basic_gene_data:
            Tuple containing metadata for this transcript. Specifically,
            a tuple containing the contig name, chunk_id, transcript id,
            transcript symbol, gene_id, gene symbol, length of transcript
        pi_only:
            Optional, default False. If True, this function will not
            calculate piN/piS values. Greatly speeds up performance.

    Returns:
        A list of tuples containing the transcript metadata in basic_gene_data
        and the diversity statistics for each sample.
    """
    # Indexing notes:
    # transcript_slice: index of transcript sites in chunk
    # transcript_var_slice: index of transcript's mutable sites in list of mutable sites
    # muts_in_transcript: index of transcript's mutable sites in transcript slice
    # so transcript_slice[muts_in_transcript] should equal var_pos[transcript_var_slice] - it does
    # var_pos[transcript_var_slice] - min(transcript_slice) = in-transcript-slice indexes of mutable sites
    # read_cts: whole chunk
    # ref_seq_array: whole chunk
    # var_pos: index of all mutable sites in chunk

    chunk_transcript_pi_vals = np.empty(shape=(10, len(sample_list)))
    chunk_transcript_pi_vals[:] = np.nan
    if not pi_only:
        try:
            synon_filters = filters.generate_coding_filters(sample_consensus_seqs,
                                                            *synon_cts,
                                                            idx_of_var_sites_in_transcript=muts_in_transcript,
                                                            transcript_name=transcript)
        except ValueError:
            with open('log.txt', 'a') as log:
                log.write(f'Chrom {basic_gene_data[0]}, chunk id {basic_gene_data[1]}, transcript {basic_gene_data[2]}, gene {basic_gene_data[4]} errored out. Transcript length was {basic_gene_data[6]}, which is {basic_gene_data[6]%3} off. First few nucs were {tuple(sample_consensus_seqs[0, :6, :].flatten())}\n')
            return [(sample_id,) + basic_gene_data + (basic_gene_data[-1],) + tuple(data)
                    for sample_id, data, in zip(sample_list, chunk_transcript_pi_vals.T)]
        synon_filter, nonsynon_filter, synon_site_filter, nonsynon_site_filter, num_const_synon_sites, num_const_nonsynon_sites = synon_filters

        nonsynon_sites = (SNP_freqs * nonsynon_site_filter).sum(axis=2)
        synon_sites = (SNP_freqs * synon_site_filter).sum(axis=2)

    # The number of columns to keep track of here is large,
    # so I will fill in a numpy array to keep everything straight.
    transcript_per_site_pi = calcpi.per_site_pi(transcript_pi_math)
    chunk_transcript_pi_vals[0] = calcpi.avg_pi_per_sample(transcript_per_site_pi, length=len(transcript_slice))

    if not pi_only:
        nonSynonPerSitePi = calcpi.per_site_pi(transcript_pi_math * nonsynon_filter)
        synonPerSitePi = calcpi.per_site_pi(transcript_pi_math * synon_filter)
        chunk_transcript_pi_vals[1] = calcpi.avg_pi_per_sample(nonSynonPerSitePi, length=nonsynon_sites.sum(axis=1) + num_const_nonsynon_sites)
        chunk_transcript_pi_vals[2] = calcpi.avg_pi_per_sample(synonPerSitePi, length=synon_sites.sum(axis=1) + num_const_synon_sites)
        chunk_transcript_pi_vals[3] = nonsynon_sites.sum(axis=1) + num_const_nonsynon_sites
        chunk_transcript_pi_vals[4] = synon_sites.sum(axis=1) + num_const_synon_sites

    # And now do the same thing w/o overlapping regions to accurately determine whole-sample piN/piS
    if oof_overlaps_idx is not None:
        # np.in1d(transcript_slice, overlap_ix)[muts_in_transcript] is the parts of the transcript that are mutable and in the overlap_ix.
        # ~that indicates we want the mutable sites that *aren't* in that list.
        overlap_sites = np.in1d(transcript_slice, oof_overlaps_idx)[muts_in_transcript]
        ##TODO: instead of ignoring overlapping sites, implement OLGenie algorithm
        remove_overlap_sites = ~overlap_sites
        chunk_transcript_pi_vals[5] = calcpi.avg_pi_per_sample(transcript_per_site_pi[:, remove_overlap_sites], length=(len(transcript_slice) - overlap_sites.sum()))

        if not pi_only:
            nonsynon_filter_no_overlap = nonsynon_filter[:, remove_overlap_sites, :]
            synon_filter_no_overlap = synon_filter[:, remove_overlap_sites, :]

            sample_transcript_N_sites_no_overlap = chunk_transcript_pi_vals[3] - nonsynon_sites[:, overlap_sites].sum(axis=1)
            sample_transcript_S_sites_no_overlap = chunk_transcript_pi_vals[4] - synon_sites[:, overlap_sites].sum(axis=1)
            nonSynonPerSitePi_no_overlap = calcpi.per_site_pi(transcript_pi_math[:, remove_overlap_sites, :] * nonsynon_filter_no_overlap)
            synonPerSitePi_no_overlap = calcpi.per_site_pi(transcript_pi_math[:, remove_overlap_sites, :] * synon_filter_no_overlap)

            chunk_transcript_pi_vals[6] = calcpi.avg_pi_per_sample(nonSynonPerSitePi_no_overlap, length=sample_transcript_N_sites_no_overlap)
            chunk_transcript_pi_vals[7] = calcpi.avg_pi_per_sample(synonPerSitePi_no_overlap, length=sample_transcript_S_sites_no_overlap)
            chunk_transcript_pi_vals[8] = sample_transcript_N_sites_no_overlap
            chunk_transcript_pi_vals[9] = sample_transcript_S_sites_no_overlap
        basic_gene_data += ((basic_gene_data[-1] - overlap_sites.sum()),)
    else:
        # data with no overlaps is identical to data w/ overlaps
        chunk_transcript_pi_vals[5:10] = chunk_transcript_pi_vals[0:5]
        basic_gene_data += (basic_gene_data[-1],)
    return [(sample_id,) + basic_gene_data + tuple(data) for sample_id, data, in zip(sample_list, chunk_transcript_pi_vals.T)]
