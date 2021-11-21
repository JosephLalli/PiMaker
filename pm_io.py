import numpy as np
import pandas as pd
import allel
# from Bio import SeqIO
# from Bio.SeqIO.FastaIO import SimpleFastaParser
import pysam
import pathlib
import gzip
import contextlib
from collections import OrderedDict
from tqdm import tqdm
import os
from timeit import default_timer as timer


def timeit(func, n=10, return_result=False, args=list(), kwargs=dict()):
    def wrapper(func, args=list(), kwargs=dict()):  # don't return results; prevents calcing time dealing w/ memory allocation of results
        r = func(*args, **kwargs)
        if hasattr(r, '__next__'):
            r = list(r)
        if return_result:
            return r
    start = timer()
    r = [wrapper(func, args, kwargs) for _ in range(n)]
    end = timer()
    print(str(round((end - start) * 1000 / n, 5)) + 'ms')
    if return_result:
        return r


@contextlib.contextmanager
def safe_open(filepath, rw):
    if '.gz' in pathlib.Path(filepath).suffixes:
        file = gzip.open(filepath, rw+'t')
    else:
        file = open(filepath, rw)
    try:
        yield file
    finally:
        file.close()


bcf_file = '/mnt/d/projects/pimaker/test_data/influenza/Orchards_H3N2_a.bcf.gz'
vcf_file = '/mnt/d/projects/pimaker/test_data/influenza/Orchards_H3N2_a.vcf'
ref_fasta = '/mnt/d/projects/pimaker/test_data/influenza/A_Singapore_INFIMH-16-0019_2016.fasta'
gtf_file = '/mnt/d/projects/pimaker/test_data/influenza/A_Singapore_INFIMH-16-0019_2016_antigenic_w_glycosylation.gtf'
mutation_rates_file = '/mnt/d/projects/pimaker/test_data/influenza/test_data/influenza/pauley_mutation_rates.tsv'
nucs = 'ACGTN'
nuc_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
nuc_array = np.concatenate((np.eye(4, dtype=np.int16), np.zeros((1, 4), dtype=np.int16)))
array_dict = {n: a for n, a in zip(nucs, nuc_array)}
nuc_tuple = tuple(map(tuple, nuc_array))
tuple_dict = {nucs[i]: t for i, t in enumerate(nuc_tuple)}


def get_num_var(vcf_file, contig=False):
    if contig:
        cmd = f'bcftools index --stats {vcf_file}'
        retval = os.popen(cmd).read()
        result = [c.split('\t') for c in retval.strip().split('\n')]
        retval = {contig: int(num_var) for contig, _, num_var in result}
    else:
        cmd = f'bcftools index --nrecords {vcf_file}'
        retval = os.popen(cmd).read()
        retval = int(retval.strip())
    return retval


def get_sample_list(vcf_file):
    retval = os.popen(
        f'bcftools query -l {vcf_file}').read()
    return retval.strip().split('\n')

# def get_sample_list(vcf_file):
#     vcf = pysam.VariantFile(vcf_file, threads=os.cpu_count() / 2)
#     test_record = next(vcf)
#     sample_list = list(test_record.samples.keys())
#     return sample_list

# x = timeit(create_combined_reference_Array, n=1, return_result=True, args=[ref_fasta])

def timeit(func, n=1, return_result=False, args=list(), kwargs=dict()):
    def wrapper(func, args=list()):  # don't return results; prevents calcing time dealing w/ memory allocation of results
        r = func(*args, **kwargs)
        if hasattr(r, '__iter__'):
            r = list(r)
        if return_result:
            return r
    
    start=timer()
    r = [wrapper(func, args) for _ in range(n)]
    end=timer()
    print(str(round((end-start)*1000/n, 5))+'ms')
    if return_result:
        return r

def SimpleFastaParser(handle):
    """Shamelessly stolen from Bio.SeqIO
    Iterate over Fasta records as string tuples.
    Arguments:
     - handle - input stream opened in text mode
    For each record a tuple of two strings is returned, the FASTA title
    line (without the leading '>' character), and the sequence (with any
    whitespace removed). The title line is not divided up into an
    identifier (the first word) and comment or description.
    >>> with open("Fasta/dups.fasta") as handle:
    ...     for values in SimpleFastaParser(handle):
    ...         print(values)
    ...
    ('alpha', 'ACGTA')
    ('beta', 'CGTC')
    ('gamma', 'CCGCC')
    ('alpha (again - this is a duplicate entry to test the indexing code)', 'ACGTA')
    ('delta', 'CGCGC')
    """
    # Skip any text before the first record (e.g. blank lines, comments)
    for line in handle:
        if line[0] == ">":
            title = line[1:].rstrip()
            break
    else:
        # no break encountered - probably an empty file
        return
    # Main logic
    # Note, remove trailing whitespace, and any internal spaces
    # (and any embedded \r which are possible in mangled files
    # when not opened in universal read lines mode)
    lines = []
    for line in handle:
        if line[0] == ">":
            yield title, "".join(lines).replace(" ", "").replace("\r", "")
            lines = []
            title = line[1:].rstrip()
            continue
        lines.append(line.rstrip())
    yield title, "".join(lines).replace(" ", "").replace("\r", "")


# @numba.njit(parallel=True)
# def do_the_thing(lines):
#     seq_id = ''
#     sequence, line = '', ''
#     for line in lines:
#         if line.startswith('>'):
#             if len(sequence) != 0: # if not the first line
#                 yield seq_id, sequence
#             sequence = ''
#             seq_id = line[1:].split(' ')[0]
#         elif line != '\n':
#             sequence += line.replace(" ", "").replace("\r", "").strip()
#     yield seq_id, sequence

def format_sample_cts(s, one_hots):
    if s['AD'] is None:
        s['AD'] = 0
    try:
        s['RD'] is None
    except KeyError:  # SARS-CoV-2 SNPeff VCFs
        if s['DP'] is None:
            return np.zeros(4, dtype=np.uint16)
        else:
            cts = np.array((s['DP']-s['AD'], s['AD']))[:, np.newaxis]
            alleles = s.allele_indices
            if (len(alleles) == 1 )or (alleles[0] == alleles[1]):  # if monoallelic
                alleles = alleles[0:1]
                cts = cts[alleles[0]:alleles[0] + 1, :]
            elif alleles[0] != 0:
                alleles = (0,) + alleles
            s_hots = one_hots[alleles, :]
            return (cts * s_hots).sum(0)
    
    if s['RD'] is None:
        return np.zeros(4, dtype=np.uint16)
    else:
        alleles = s.allele_indices
        cts = np.array((s['RD'],) + s['AD'], dtype=np.uint16)[:, np.newaxis]
        if alleles[0] == alleles[1]:  # if monoallelic
            alleles = alleles[0:1]
            cts = cts[alleles[0]:alleles[0] + 1, :]
        elif alleles[0] != 0:
            alleles = (0,) + alleles
        s_hots = one_hots[alleles, :]
        return (cts * s_hots).sum(0)


def is_snp(r):
    '''determines if a pysam variant record is a snp or not'''
    return (len(r.ref) == 1) and any([len(a)==1 for a in r.alts]) and (r.ref is not None)


def read_fasta_as_idx(path):
    with safe_open(ref_fasta, 'r') as fasta:
        refseq = [(id.split(' ')[0], seq) for id, seq in SimpleFastaParser(fasta)]
    return refseq


def read_mutation_rates(mutation_rates):
    if mutation_rates is None:
        return None
    elif type(mutation_rates) == np.array:
        # if user passed an array of mutation rates directly to the calc_Pi function, then they're all set
        pass
    elif type(mutation_rates) == str:  # if they gave a file, import file
        suffix = mutation_rates.split('.')[0]
        if suffix == 'npy':
            mutation_rates = np.load(mutation_rates).values
        elif suffix == 'json':
            mutation_rates = pd.read_json(mutation_rates).values
        elif suffix == 'csv':
            mutation_rates = pd.read_csv(mutation_rates).values
        elif suffix == 'tsv':
            mutation_rates = pd.read_csv(mutation_rates, suffix='\t').values
        elif 'xls' in suffix:
            mutation_rates = pd.read_excel(mutation_rates).values
        else:
            raise NotImplementedError
    else:
        mutation_rates = np.array(mutation_rates)
    assert mutation_rates.shape == (4, 4), 'Mutation rate table must be a 4x4 table of mutation rates from ACGT(rows) to ACGT(columns)'
    return mutation_rates


def convert_to_onehot(seq, default_idx=(0, 0, 0, 0)):
    u, inv = np.unique(seq, return_inverse=True)
    return np.array([tuple_dict.get(x, default_idx) for x in u], dtype=np.int16)[inv]


def convert_to_idx(seq, default_idx=-1):
    '''uses a fun trick from stackoverflow to quickly apply dict to np array'''
    u, inv = np.unique(seq, return_inverse=True)
    return np.array([nuc_dict.get(x, default_idx) for x in u], dtype=np.int16)[inv]


def create_combined_reference_Array(ref_fasta):
    with safe_open(ref_fasta, 'r') as fasta:
        refseq = [(id.split(' ')[0], seq.upper()) for id, seq in SimpleFastaParser(fasta)]
    # refseq = list(read_fasta_as_idx(ref_fasta))
    refseqArray, contig_starts, contig_coords = combine_contigs(refseq)
    refseqArray = convert_to_onehot(refseqArray)[np.newaxis, :, :]
    return refseqArray, contig_starts, contig_coords


def vcf_to_numpy_array_read_cts(vcf_file, contig_coords, region=None, maf=0):
    # '''as currently coded, errors if region is not specified to at least one contig'''
    # if region is None:
    #     raise NotImplementedError('Currently we cannot read more than one contig\'s mutations at once. Apologies. Please specify the contig in the region parameter.')
    better_names_for_vcf_fields = {'calldata/AD': 'AD', 'calldata/DP': 'DP', 'calldata/RD': 'RD',
                                    'variants/ALT': 'alt', 'variants/CHROM': 'contig',
                                    'variants/POS': 'pos', 'variants/REF': 'ref'}
    fields_to_extract = list(better_names_for_vcf_fields.keys()) + ['samples']
    vcf = allel.read_vcf(vcf_file, fields=fields_to_extract, rename_fields=better_names_for_vcf_fields, region=region)
    # if vcf is None:
    #     print ('allel returned a vcf of none!')
    #     print (vcf_file, region)
    #     print (vcf.keys())
    #     print (vcf['pos'])
    #     raise TypeError
    # pos is 1-indexed in allel
    if region is None:
        vcf['contig']
    else:
        og_var_pos = vcf['pos'] + contig_coords[0] - 1  # var_pos = [pos + contig_starts[contig] for pos, contig in zip(vcf['pos'], vcf['contig'])]
        var_pos, var_pos_ind = np.unique(og_var_pos, return_index=True)
    if len(var_pos) < len(og_var_pos):  # if multiallelic sites are already split up, I need to concat them back together
        vcf['ref'] = vcf['ref'][var_pos_ind]
        vcf['RD'] = merge_split_multiallelics(vcf['RD'], og_var_pos, action='sum')
        vcf['AD'] = merge_split_multiallelics(vcf['AD'][:, :, 0], og_var_pos, action='reshape counts')
        vcf['alt'] = merge_split_multiallelics(vcf['alt'], og_var_pos, action='reshape nucs')

    read_cts = convert_to_onehot(vcf['ref'])[np.newaxis, :, :]
    vcf['RD'][vcf['RD'] == -1] = 0
    vcf['AD'][vcf['AD'] == -1] = 0
    vcf['RD'] = vcf['RD'].T[:, :, np.newaxis]
    vcf['AD'] = vcf['AD'].swapaxes(0, 1)
    read_cts = read_cts * vcf['RD']
    for i, alts in enumerate(vcf['alt'].T):
        read_cts += convert_to_onehot(alts, default_idx=tuple_dict['N'])[np.newaxis, :, :] * vcf['AD'][:, :, i:(i + 1)]
    read_cts = np.where(read_cts < maf * read_cts.sum(2, keepdims=True), 0, read_cts)
    read_cts = read_cts.astype(np.uint16)
    var_pos = var_pos.astype(np.uint64)
    return read_cts, var_pos


def merge_split_multiallelics(values, index, action=None):
    import pandas as pd
    values_df = pd.DataFrame(values, index=index).reset_index().groupby('index')
    if action == 'sum':
        return values_df.apply(lambda x: x[x.columns[1:]].sum()).values
    elif action == 'reshape counts':
        df_of_arrays = values_df.apply(lambda x: np.array([tuple(x[c])+(-1,)*(4-len(x[c])) for c in x.columns[1:]]))
        return np.stack(df_of_arrays.values)
    elif action == 'reshape nucs':
        values_df = values_df.apply(lambda x: pd.Series([tuple(x[0].values)+('',)*(3-len(x))][0]))
        return values_df.values
    else:
        raise ValueError('merge_split_multiallelics requires a specified action to handle the merge')


def save_tmp_chunk_results(contig, chunk_id, chunk_sample_pi, chunk_gene_pi, chunk_site_pi, var_site_positions, cache_folder):
    overall_sample_pi_columns = list(['sample_id', 'contig', 'chunk_id', 'chunk_len', 'stat_name', 'stat'])
    site_sample_pi_columns =    tuple(['sample_id', 'contig', 'stat_name'])
    gene_sample_pi_columns =    list(['sample_id', 'contig', 'chunk_id', 'transcript_id', 'transcript', 'gene_id', 'gene_symbol',
                                      'transcript_len', 'transcript_len_no_overlap', 'pi', 'piN', 'piS', 'N_sites', 'S_sites',
                                      'pi_no_overlap', 'piN_no_overlap', 'piS_no_overlap', 'N_sites_no_overlap', 'S_sites_no_overlap'])
    chunk_site_pi_df = pd.DataFrame(chunk_site_pi, columns=(site_sample_pi_columns + tuple(var_site_positions)))
    chunk_sample_pi_df = pd.DataFrame(chunk_sample_pi, columns=overall_sample_pi_columns)
    chunk_gene_pi_df = pd.DataFrame(chunk_gene_pi, columns=gene_sample_pi_columns)
    site_filename = f'{cache_folder}/{contig}_{chunk_id}_site.csv'
    sample_filename = f'{cache_folder}/{contig}_{chunk_id}_sample.csv'
    gene_filename = f'{cache_folder}/{contig}_{chunk_id}_gene.csv'
    chunk_site_pi_df.to_csv(site_filename)
    chunk_sample_pi_df.to_csv(sample_filename)
    chunk_gene_pi_df.to_csv(gene_filename)
    return sample_filename, gene_filename, site_filename
# from numba import vectorize, int16

# timeit(vcf_to_numpy_array_read_cts, args=(vcf_file, refseqArray, contig_starts),n=2)

# @vectorize([int16(int16, int16)])
# def multiply_arrays(x, y):
#     return x * y

# @numba.njit(parallel=True)#([int16(int16, int16)])
# def multiply_arrays(x, y):
#     return x * y

# def multiply_arrays(x,y):
#     y = np.tile(y, np.ceil(np.array(x.shape)/np.array(y.shape)).astype(np.int8))
#     return numba_mult(x,y)

# @numba.jit(nopython=True, parallel=True)
# def numba_mult(x,y):
#     result = np.zeros((y.shape[0], y.shape[1], x.shape[2]), dtype=np.int16)
#     r = result[:]
#     for i in numba.prange(y.shape[0]):
#         for j in numba.prange(y.shape[1]):
#             for k in numba.prange(x.shape[2]):
#                 if x[0,j,k] == 0:
#                     pass#result[i,j,k] == np.int16(0)
#                 else:
#                     r[i,j,k] == y[i,j,0]
#     return result
# RD[RD == -1] = np.int16(0)
# AD[AD == -1] = np.int16(0)
# RD = RD.T[:,:,np.newaxis]
# AD = AD.swapaxes(0,1)
# ref = ref[np.newaxis,:,:]
# q = nconeumba.typed.List()
# for x in alts:
#     q.append(x)#[x[np.newaxis, :, :].T for x in alts])
# @numba.njit()
# def set_up_read_cts(refs, alts_onehot, RD, AD):
#     read_cts = refs * RD
#     for i, alts in enumerate(alts_onehot):
#         # alt_idx = convert_to_onehot(alts, default_idx = nucs.index('N'))
#         read_cts += alts * AD[:,:,i:i+1]
#     return read_cts
# # @numba.njit
# def get_varpos(pos, contig, contig_starts):
#     var_pos = [pos+contig_starts[contig] for pos, contig in zip(pos, contig)]
#     var_pos, var_pos_idx = np.unique(var_pos, return_index=True)
#     return var_pos, var_pos_idx


def vcf_to_numpy_array_pysam(vcf_file, refseqArray, contig_starts, maf=0, contig_coords=None):
    vcf = pysam.VariantFile(vcf_file, threads=os.cpu_count() / 2)
    test_record = next(vcf)
    sample_list = list(test_record.samples.keys())
    vcf.reset()
    num_records = get_num_var(vcf_file)
    read_cts = np.zeros((len(sample_list), num_records, 4), dtype=np.int16)
    var_pos = np.zeros(num_records)
    for i, r in enumerate(vcf):
        var_pos[i] = r.pos + contig_starts[r.contig]
        r_nuc_array = np.array([tuple_dict.get(x, (0, 0, 0, 0)) for x in ((r.ref,) + r.alts)])
        read_cts[:, i, :] = np.stack([(np.array((s['RD'], s['AD']))[:, np.newaxis] * r_nuc_array[np.unique((0,) + tuple(a for a in s.allele_indices if a is not None)), :]).sum(0) for j, s in enumerate(r.samples.values())])
    return read_cts, var_pos

# def vcf_to_numpy_array_cyvcf2(vcf_file, refseqArray, contig_starts, maf=0, contig_coords=None):
# vcf = cyvcf2.VCF(vcf_file, threads = os.cpu_count()/2)
# test_record = next(vcf)
# sample_list = list(test_record.samples.keys())
# vcf.reset()
# num_records = get_num_var(vcf_file)
# read_cts = np.zeros((len(sample_list), num_records, 4),dtype=np.int16)
# var_pos = np.zeros(num_records)
# for i, r in enumerate(vcf):
#     var_pos = [r.pos + contig_starts[r.contig] for r in vcf]
#         r_nuc_array = np.array([tuple_dict.get(x, (0,0,0,0)) for x in ((r.ref,)+r.alts)])
#         read_cts[:,i,:] = np.stack([(np.array((s['RD'], s['AD']))[:,np.newaxis]*r_nuc_array[np.unique((0,)+tuple(a for a in s.allele_indices if a is not None)), :]).sum(0) for j,s in enumerate(r.samples.values())])
#     return read_cts, var_pos

    # var_pos = [r.pos + contig_starts[r.contig] for r in vcf]
    # var_pos, var_pos_idx, inv_var = np.unique(var_pos, return_index=True, return_inverse=True)
    # num_records = len(var_pos)
    # vcf.reset()
    # read_cts = np.zeros(shape = (len(sample_list), num_records, 4))
    # for i, r in enumerate(vcf):
    #     assert len(r.alts) == 1
    #     if len(r.ref) != len(r.alts[0]): #if indel
    #         continue
    #     r_pos = inv_var[i]
    #     r_cts = np.stack([array_dict[r.ref]*samp_record['RD'] + array_dict[r.alts[0]]*samp_record['AD'] for samp_record in r.samples.values()])
    #     read_cts[:, r_pos, :] += r_cts

    # # fill in samples where a variant wasn't called in that sample
    # read_cts = np.where(read_cts.sum(2, keepdims=True)==0, refseqArray[:,var_pos,:], read_cts)
    # read_cts[read_cts/read_cts.sum(2, keepdims=True) < maf] = 0
    # return read_cts, sample_list, var_pos, var_pos_idx


def combine_contigs(refseq):
    '''Combine contig sequences into one string, annotate contig locations within concatenated genome'''
    concatrefseq = "".join([seq[1] for seq in refseq])
    contigStarts = dict()
    contigCoords = dict()
    runningtally = 0
    for id, seq in refseq:
        contigStarts[id] = runningtally
        contigCoords[id] = (runningtally, runningtally + len(seq))
        runningtally += len(seq)
    refseqArray = np.array(list(concatrefseq))
    return refseqArray, contigStarts, contigCoords

def parseGTF(gtffile, contig_coords=None):
    '''given file location of gtf, and dictionary of starting locations
       of each chrom in a concatenated sequence, return dictionary of
       {gene product : numpy filter for concatenated sequence'''
    # Note to self: Stupid GTFs are stupid 1-indexed with stupid inclusive ends
    with safe_open(gtffile, 'r') as g:
        gtf = g.readlines()

    gene_coords = OrderedDict()#(lambda: defaultdict(list))
    id_to_symbol = dict()
    transcript_to_gene_id = dict()

    for chrom in contig_coords.keys():
        gene_coords[chrom] = OrderedDict()

    for line in gtf:
        line = line.replace("/", "_")
        mandatory_items = line.split("\t")
        custom_items = [item.replace("\"", '').strip() for item in mandatory_items[-1].split(';') if not item.isspace()]
        custom_items = [tuple(item.split(' ')) for item in custom_items if item[0] != '#']
        custom_items = {k: v for k, v in custom_items}
        mandatory_items = mandatory_items[:-1]
        contig_name = mandatory_items[0]
        if contig_name not in gene_coords.keys():
            gene_coords[contig_name] = OrderedDict()
        annotation_type = mandatory_items[2]
        start = int(mandatory_items[3]) - 1  # adding the -1 here to convert to 0 indexing
        stop = int(mandatory_items[4])  # not adding -1 because, while GTF is 1-indexed, its inclusive-ended. Converting to standard 0-indexing would mean 1-10 in GTF is equivelent to [0:10]
        if contig_coords is not None:
            start += contig_coords[contig_name]
            stop += contig_coords[contig_name]
        strand = mandatory_items[6]
        if strand == '-':
            start, stop = stop, start
        if annotation_type.lower() == "cds":
            if custom_items['transcript_id'] not in gene_coords[contig_name].keys():
                gene_coords[contig_name][custom_items['transcript_id']] = tuple()
            transcript_to_gene_id[custom_items['transcript_id']] = custom_items['gene_id']
            id_to_symbol[custom_items['gene_id']] = custom_items['gene_symbol']
            id_to_symbol[custom_items['transcript_id']] = custom_items['transcript_symbol']
            gene_coords[contig_name][custom_items['transcript_id']]+=((start, stop),)
    for chrom in gene_coords.keys():
        gene_coords[chrom] = OrderedDict(sorted(gene_coords[chrom].items(), key=lambda item: min(item[1][0][0], item[1][-1][-1])))
    return gene_coords, transcript_to_gene_id, id_to_symbol


def retrieve_genes_in_region(start, stop, gene_coords, contig_start):
    '''record genes present in region.
       while recording, if pos is in middle of gene, adjust it'''
    genes = OrderedDict()
    # start -= contig_start
    # stop -= contig_start
    for gene, coords in gene_coords.items():
        gene_start = min(coords[0][0], coords[-1][-1])
        gene_end = max(coords[0][0], coords[-1][-1])
        if gene_start > stop:  # if gene begins after region, stop iteration.
            break
        elif gene_start >= start:  # if gene begins after start of region and before end of region, record gene
            genes[gene] = coords
            if gene_end >= stop:  # if the gene begins within region but ends after region, adjust end of region.
                stop = gene_end + 1
    return genes, stop


def format_mut_array(muts, num_samples, chunklen, contig_coords, maf, tqdm_lock=None):
    '''given iterator of pysam muts in a region,
       return array of read counts'''
    # I don't have a great way of determining the number of mutations in a chunk
    read_cts = np.zeros((num_samples, chunklen, 4), dtype=np.uint16)
    var_pos = np.zeros(chunklen, dtype=np.uint64)
    i = 0
    if tqdm_lock:
        tqdm.set_lock(tqdm_lock)
    for r in tqdm(muts, position=0):
        if is_snp(r):
            var_pos[i] = r.pos + contig_coords[r.contig][0]
            nucs = ((r.ref,) + tuple(a for a in r.alts if len(a) == 1 and a is not None))
            one_hots = np.array(list(tuple_dict[n] for n in nucs), dtype=np.uint16)
            try:
                read_cts[:, i, :] = np.stack([format_sample_cts(s, one_hots) for s in r.samples.values()])
            except Exception as e:
                print (e)
                return r, i, read_cts
            i += 1

    read_cts = np.where(read_cts < maf * read_cts.sum(2, keepdims=True), 0, read_cts)
    return read_cts[:, :i, :], var_pos[:i]


def pool_to_numpy(poolfile):
    pool = pd.read_csv(poolfile, sep='\t')
    sample_list = pool.columns[2:]
    var_pos = pool.Position.astype(int)
    
    def col_to_nump(column):
        return np.stack(column.str.split(':').values).astype(int)

    read_cts = np.stack([col_to_nump(pool[sample])[:, :-2] for sample in sample_list])
    return read_cts, var_pos, sample_list

import copy
def vcf_to_sync(vcf_file, sync_file=None):
    if sync_file is None:
        sync_file = vcf_file.replace('.vcf', '.sync').replace('.bcf','.sync').replace('.gz','')
    vcf = pysam.VariantFile(vcf_file, threads=os.cpu_count() / 2)
    test_record = next(vcf)
    sample_list = list(test_record.samples.keys())
    vcf.reset()
    num_records = get_num_var(vcf_file, contig=True)
    read_cts = dict()
    var_pos = dict()
    ref = dict()
    contigs = list()
    contig_adjust = 0
    pool_fmt_dfs = list()
    for i, r in enumerate(vcf):
        if r.contig not in var_pos.keys():
            var_pos[r.contig] = np.zeros(num_records[r.contig])
            ref[r.contig] = ''
            read_cts[r.contig] = np.zeros((len(sample_list), num_records[r.contig], 6), dtype=np.int16)
            contig_adjust = i # assumes vcf sorted
            
        i = i-contig_adjust
        contigs.append(r.chrom)
        var_pos[r.contig][i] = r.pos
        r_nuc_array = np.array([tuple_dict.get(x, (0, 0, 0, 0)) + (0, 0) for x in ((r.ref,) + r.alts)])
        read_cts[r.contig][:, i, :] = np.stack([(np.array((s['RD'], s['AD']))[:, np.newaxis] * r_nuc_array[np.unique((0,) + tuple(a for a in s.allele_indices if a is not None)), :]).sum(0) for j, s in enumerate(r.samples.values())])
        ref[r.contig] += r.ref
        

    # Convert ACGT to ATCG because popoolation is silly
    for contig in tqdm(read_cts.keys()):
        tmp_reads = copy.deepcopy(read_cts[contig][..., 1])
        read_cts[contig][..., 1] = read_cts[contig][..., 3]
        read_cts[contig][..., 3] = tmp_reads
        del tmp_reads

        # if multiallelic already split, recombine
        og_var_pos = copy.deepcopy(var_pos[contig])
        var_pos[contig], var_pos_ind = np.unique(og_var_pos, return_index=True)
        if len(var_pos[contig]) < len(og_var_pos):  # if multiallelic sites are already split up, I need to concat them back together
            ref[contig] = ''.join(list(np.array(list(ref[contig]))[var_pos_ind]))
            read_cts[contig] = np.stack([merge_split_multiallelics(read_cts[contig][:, :, i].T, og_var_pos, action='sum').T for i in range(6)], axis=2)

        pool_fmt_list = [np.array2string(read_cts[contig][i,:,:].astype(int), separator=':', threshold=num_records[contig]*2, edgeitems=num_records[contig]*2).replace(' ','').replace('[','').replace(']','').split(':\n') for i in range(len(sample_list))]
        pool_fmt_cts = pd.DataFrame(pool_fmt_list).T
        pool_fmt_cts.columns = sample_list
        pool_fmt_cts['Contig'] = contig
        pool_fmt_cts['Position'] = var_pos[contig].astype(int)
        pool_fmt_cts['Ref'] = list(ref[contig])
        pool_fmt_cts = pool_fmt_cts[['Contig', 'Position', 'Ref'] + sample_list]
        pool_fmt_dfs.append(pool_fmt_cts)

    sync = pd.DataFrame()
    for df in pool_fmt_dfs:
        sync = sync.append(df)
    sync.to_csv(sync_file, sep='\t', index=False, header=False)
    return sync_file
