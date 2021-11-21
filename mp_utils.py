import os
import numpy as np
import pysam
from tqdm import tqdm, trange
from pm_io import get_num_var, get_sample_list, pool_to_numpy, retrieve_genes_in_region, format_mut_array, vcf_to_numpy_array_read_cts
from diversity_calcs import performPiCalc, calcPerSitePi, calcPerSamplePi
from generate_filters import generate_coding_filters
from array_manipulation import calc_consensus_seqs, coordinates_to_slices, calc_overlaps

use_allel = True


def process_chunk(chunk_queue, result_queue, sample_list, contig_coords, id_to_symbol, 
                  transcript_to_gene_id, synon_cts, tqdm_lock, process_id, pi_only=False):
    tqdm.set_lock(tqdm_lock)
    with tqdm(position=process_id+2) as pbar:
        while True:
            chunk = chunk_queue.get()
            if chunk is None:
                result_queue.put(None)
                pbar.close()
                pbar.reset()
                return None
            read_cts, var_pos, gene_coords, ref_array_chunk, bin_start, bin_end, contig, chunk_id = chunk
            pbar.reset()
            pbar.set_description(f'{contig}:{bin_start}-{bin_end}')
            pbar.refresh()
            # Create lists to track results in this chunk:
            chunk_sample_pi, chunk_gene_pi, chunk_site_pi = list(), list(), list()

            # The number of columns in the gene data is unwieldly to do anything but a numpy array that I fill in
            chunk_len = bin_end - bin_start
            basic_data = (contig, chunk_id, chunk_len)

            # print ('Calculating Pi...')
            piMath = performPiCalc(read_cts)
            with np.errstate(divide='ignore', invalid='ignore'):
                freq_cts = read_cts.astype(np.float32) / read_cts.sum(2, keepdims=True).astype(np.float32)
                freq_cts = np.nan_to_num(freq_cts)

            chunk_per_site_pi = calcPerSitePi(piMath)
            chunk_per_sample_pi = calcPerSamplePi(chunk_per_site_pi, length=(bin_end - bin_start))

            # before processing gene results, record pi data
            chunk_sample_pi.extend([(sample_id,) + basic_data + ('pi',) + tuple((pi,)) for sample_id, pi in zip(sample_list, chunk_per_sample_pi)])
            chunk_site_pi.extend([(sample_id, contig, 'pi') + tuple(tuple(pi,)) for sample_id, pi in zip(sample_list, chunk_per_site_pi)])

            # Now calculate coding regions' Pi/PiN/PiS
            ##TODO: This step takes a surprising amount of time
            gene_slices, gene_var_slices, idx_of_var_sites = coordinates_to_slices(var_pos, gene_coords)
            overlapping_out_of_frame_idx = calc_overlaps(gene_slices, gene_coords, bin_start, bin_end)
            try:
                sample_consensus_seqs = calc_consensus_seqs(read_cts, ref_array_chunk, var_pos)
            except IndexError as exception:
                print (exception)
                print (f'\n\n\n\n\n\n\n\n\n\ncontig:{contig}, bin_start: {bin_start}, bin_end {bin_end}, var_pos start {var_pos[0]}, var_pos end {var_pos[-1]}, read_cts size {read_cts.shape}, ref_array_chunk size {ref_array_chunk.shape} errored out with an index error for incorrectly sized var pos\n\n\n\n\n\n\n\n\n\n')
                continue
            pbar.total = len(gene_slices)
            pbar.refresh()
            for i, (gene, gene_slice) in enumerate(gene_slices.items()):
                basic_gene_data = (contig, chunk_id, gene, id_to_symbol[gene], transcript_to_gene_id[gene],
                                id_to_symbol[transcript_to_gene_id[gene]], len(gene_slice))
                gene_var_slice = gene_var_slices[gene]
                muts_in_gene = idx_of_var_sites[gene]
                gene_args = (read_cts[:, gene_var_slice, :], sample_consensus_seqs[:, gene_slice, :], muts_in_gene,
                            synon_cts, freq_cts[:, gene_var_slice, :], piMath[:, gene_var_slice, :],
                            sample_list, overlapping_out_of_frame_idx, gene, gene_slice, basic_gene_data, pi_only)
                chunk_gene_pi.extend(process_gene(*gene_args))
                pbar.update(1)
            pbar.refresh()
            result_queue.put((contig, chunk_id, chunk_sample_pi, chunk_gene_pi, chunk_site_pi, var_pos))


def mp_iterate_records(vcf_file, ref_array, contig_coords, gene_coords, chunk_queue, tqdm_lock,
                       num_var=None, num_processes=1, binsize=int(1e6), maf=0, cache_folder='chunk_cache'):
    '''pre-process chunk and place in mp.Queue'''
    # create cache folder
    tqdm.set_lock(tqdm_lock)
    mut_suffix = vcf_file.split('.')[-1:]

    if 'sync' in mut_suffix:
        read_cts, var_pos, sample_list = pool_to_numpy(vcf_file)
    else:  # assume vcf
        if use_allel:  # if using allel
            sample_list = get_sample_list(vcf_file)
        else:  # if using pysam
            vcf = pysam.VariantFile(vcf_file, threads=os.cpu_count() / 2)
            sample_list = get_sample_list(vcf_file)

    if num_var is None:
        num_var = get_num_var(vcf_file)
    print ('Number of chunks to get through (in theory):')
    print (ref_array.shape[1] // binsize)
    chunk_id = 0
    with tqdm(position=1) as pbar:
        for contig, (contig_start, contig_end) in contig_coords.items():
            pbar.set_description(f'Loading contig {contig}')
            pbar.refresh()
            bin_start = contig_start
            bin_end = 0
            pbar.reset(total=int((contig_end - contig_start) // binsize) + 1)
            for i in range(int((contig_end - contig_start) // binsize) + 1):
                if chunk_id < 66:
                    chunk_id += 1
                    continue
                bin_start = np.max((((i * binsize) + contig_start), bin_end))
                if bin_start > contig_end:
                    continue
                bin_end = np.min(((binsize * (i + 1) + contig_start), contig_end))
                if bin_end > contig_end:
                    bin_end = contig_end
                genes, bin_end = retrieve_genes_in_region(bin_start, bin_end, gene_coords[contig], contig_start)
                genes = {g: tuple((exon_start - bin_start, exon_end - bin_start) for exon_start, exon_end in coords) for g, coords in genes.items()}

                cache_base_name = f'{cache_folder}/{contig}_{bin_start}_{bin_end}'

                # if os.path.exists(f'{cache_base_name}_read_cts.npy'):
                #     read_cts = np.load(f'{cache_base_name}_read_cts.npy')
                #     var_pos = np.load(f'{cache_base_name}_var_pos.npy')
                # if 'sync' in mut_suffix:
                #     np.save(f'{cache_base_name}_read_cts.npy', read_cts)
                #     np.save(f'{cache_base_name}_var_pos.npy', var_pos)
                # else:  # assume vcf
                if use_allel:
                    try:
                        # allel is 1-indexed
                        region_string = f'{contig}:{bin_start - contig_coords[contig][0]}-{bin_end - contig_coords[contig][0]}'
                        read_cts, var_pos = vcf_to_numpy_array_read_cts(vcf_file, contig_coords[contig], region=region_string, maf=maf)
                    except (AttributeError, TypeError):
                        print (f'\n\n\n\n\n\n\n\n\n\ncontig:{contig}, bin_start: {bin_start}, bin_end {bin_end}, region string {contig}:{bin_start-contig_coords[contig][0]}-{bin_end-contig_coords[contig][0]} errored out with a None returned VCF file\n\n\n\n\n\n\n\n\n\n')
                        continue
                    # print (contig, bin_start, bin_end, var_pos[0], var_pos[-1])
                else:
                    muts = vcf.fetch(contig=contig, start=bin_start, end=bin_end)
                    read_cts, var_pos = format_mut_array(muts, len(sample_list), bin_end - bin_start, contig_coords, maf, tqdm_lock)

                    # np.save(f'{cache_base_name}_read_cts.npy', read_cts)
                    # np.save(f'{cache_base_name}_var_pos.npy', var_pos)
                # load read_cts and var_pos from cache if available, since this takes so long

                chunk_queue.put((read_cts, var_pos - bin_start, genes, ref_array[:, bin_start:bin_end, :], bin_start, bin_end, contig, chunk_id))
                chunk_id += 1
                pbar.update(1)
            pbar.refresh()
            
    # Once out of data, put num_processes number of Nones in chunk_queue
    for _ in range(num_processes):
        chunk_queue.put(None)
    pbar.close()
    pbar.refresh()
    return None


def process_gene(read_cts, sample_consensus_seqs, muts_in_gene, synon_cts, SNP_freqs, genePiMathArray, sample_list,
                 overlapping_out_of_frame_idx, gene, gene_slice, basic_gene_data, pi_only=False):
    # Indexing notes:
    # gene_slice: index of gene sites in chunk
    # gene_var_slice: index of gene's mutable sites in list of mutable sites
    # muts_in_gene: index of gene's mutable sites in gene slice
    # so gene_slice[muts_in_gene] should equal var_pos[gene_var_slice] - it does
    # var_pos[gene_var_slice] - min(gene_slice) = in-gene-slice indexes of mutable sites
    # read_cts: whole chunk
    # refseqArray: whole chunk
    # var_pos: index of all mutable sites in chunk

    chunk_gene_pi_vals = np.empty(shape=(10, len(sample_list)))
    chunk_gene_pi_vals[:] = np.nan
    if not pi_only:
        try:
            synon_filters = generate_coding_filters(sample_consensus_seqs, *synon_cts, idx_of_var_sites_in_gene=muts_in_gene, gene_name=gene)
        except ValueError:
            with open('log.txt', 'a') as log:
                log.write(f'Chrom {basic_gene_data[0]}, chunk id {basic_gene_data[1]}, transcript {basic_gene_data[2]}, gene {basic_gene_data[4]} errored out. Gene length was {basic_gene_data[6]}, which is {basic_gene_data[6]%3} off. First few nucs were {tuple(sample_consensus_seqs[0, :6, :].flatten())}\n')
            return [(sample_id,) + basic_gene_data+(basic_gene_data[-1],) + tuple(data) for sample_id, data, in zip(sample_list, chunk_gene_pi_vals.T)]
        synonFilter, nonSynonFilter, synonSiteFilter, nonSynonSiteFilter, num_const_synon_sites, num_const_nonsynon_sites = synon_filters

        nonsynon_sites = (SNP_freqs * nonSynonSiteFilter).sum(axis=2)
        synon_sites = (SNP_freqs * synonSiteFilter).sum(axis=2)

    # The number of columns to keep track of here is large, so I will fill in a numpy array to keep everything straight.
    # cols to fill in are: 'pi', 'piN', 'piS', 'N_sites', 'S_sites', 'piN_no_overlap', 'piS_no_overlap','N_sites_no_overlap', 'S_sites_no_overlap'

    genePerSitePi = calcPerSitePi(genePiMathArray)
    chunk_gene_pi_vals[0] = calcPerSamplePi(genePerSitePi, length=len(gene_slice))

    if not pi_only:
        nonSynonPerSitePi = calcPerSitePi(genePiMathArray * nonSynonFilter)
        synonPerSitePi = calcPerSitePi(genePiMathArray * synonFilter)
        chunk_gene_pi_vals[1] = calcPerSamplePi(nonSynonPerSitePi, length=nonsynon_sites.sum(axis=1) + num_const_nonsynon_sites)
        chunk_gene_pi_vals[2] = calcPerSamplePi(synonPerSitePi, length=synon_sites.sum(axis=1) + num_const_synon_sites)
        chunk_gene_pi_vals[3] = nonsynon_sites.sum(axis=1) + num_const_nonsynon_sites
        chunk_gene_pi_vals[4] = synon_sites.sum(axis=1) + num_const_synon_sites

    # And now do the same thing w/o overlapping regions to accurately determine whole-sample piN/piS
    if gene in overlapping_out_of_frame_idx.keys():
        # overlap_ix = set().union(*[set(range(start,end)) for start,end in overlaps[gene]])
        overlap_ix = overlapping_out_of_frame_idx[gene]
        # np.in1d(gene_slice, overlap_ix)[muts_in_gene] is the parts of the gene that are mutable and in the overlap_ix.
        # ~that indicates we want the mutable sites that *aren't* in that list.
        overlap_sites = np.in1d(gene_slice, overlap_ix)[muts_in_gene]
        remove_overlap_sites = ~overlap_sites
        chunk_gene_pi_vals[5] = calcPerSamplePi(genePerSitePi[:, remove_overlap_sites], length=(len(gene_slice) - overlap_sites.sum()))

        if not pi_only:
            nonSynonFilter_no_overlap = nonSynonFilter[:, remove_overlap_sites, :]
            synonFilter_no_overlap = synonFilter[:, remove_overlap_sites, :]

            sample_gene_N_sites_no_overlap = chunk_gene_pi_vals[3] - nonsynon_sites[:, overlap_sites].sum(axis=1)
            sample_gene_S_sites_no_overlap = chunk_gene_pi_vals[4] - synon_sites[:, overlap_sites].sum(axis=1)
            nonSynonPerSitePi_no_overlap = calcPerSitePi(genePiMathArray[:, remove_overlap_sites, :] * nonSynonFilter_no_overlap)
            synonPerSitePi_no_overlap = calcPerSitePi(genePiMathArray[:, remove_overlap_sites, :] * synonFilter_no_overlap)

            chunk_gene_pi_vals[6] = calcPerSamplePi(nonSynonPerSitePi_no_overlap, length=sample_gene_N_sites_no_overlap)
            chunk_gene_pi_vals[7] = calcPerSamplePi(synonPerSitePi_no_overlap, length=sample_gene_S_sites_no_overlap)
            chunk_gene_pi_vals[8] = sample_gene_N_sites_no_overlap
            chunk_gene_pi_vals[9] = sample_gene_S_sites_no_overlap
        basic_gene_data += ((basic_gene_data[-1] - overlap_sites.sum()),)
    else:
        # data with no overlaps is identical to data w/ overlaps
        chunk_gene_pi_vals[5:10] = chunk_gene_pi_vals[0:5]
        basic_gene_data += (basic_gene_data[-1],)
    return [(sample_id,) + basic_gene_data + tuple(data) for sample_id, data, in zip(sample_list, chunk_gene_pi_vals.T)]
