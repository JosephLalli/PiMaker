import numpy as np
import pandas as pd
import allel
from Bio import SeqIO
import pathlib
import gzip
import numba
import contextlib
from collections import defaultdict, OrderedDict
import os
from timeit import default_timer as timer

# x = timeit(create_combined_reference_Array, n=1, return_result=True, args=[ref_fasta])

def timeit(func, n=10, return_result=False, args=list(), kwargs=dict()):
    def wrapper(func, args=list(), kwargs=dict()):  # don't return results; prevents calcing time dealing w/ memory allocation of results
        r = func(*args, **kwargs)
        if hasattr(r, '__next__'):
            r = list(r)
        if return_result:
            return r
    start=timer()
    r = [wrapper(func, args, kwargs) for _ in range(n)]
    end=timer()
    print(str(round((end-start)*1000/n, 5))+'ms')
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
nuc_array = np.concatenate((np.eye(4, dtype=np.int16), np.zeros((1,4), dtype=np.int16)))
array_dict = {n:a for n,a in zip(nucs, nuc_array)}
nuc_tuple = tuple(map(tuple, nuc_array))
tuple_dict = {nucs[i]:t for i,t in enumerate(nuc_tuple)}

def get_num_var(vcf_file):
    retval = os.popen(
        f'bcftools index --nrecords {vcf_file}').read()
    return int(retval.strip())

# x = timeit(create_combined_reference_Array, n=1, return_result=True, args=[ref_fasta])

def timeit(func, n=1, return_result=False, args=list()):
    def wrapper(func, args=list()):  # don't return results; prevents calcing time dealing w/ memory allocation of results
        r = func(*args)
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

# def SimpleFastaParser(handle):
#     """Iterate over Fasta records as string tuples.
#     Arguments:
#      - handle - input stream opened in text mode
#     For each record a tuple of two strings is returned, the FASTA title
#     line (without the leading '>' character), and the sequence (with any
#     whitespace removed). The title line is not divided up into an
#     identifier (the first word) and comment or description.
#     >>> with open("Fasta/dups.fasta") as handle:
#     ...     for values in SimpleFastaParser(handle):
#     ...         print(values)
#     ...
#     ('alpha', 'ACGTA')
#     ('beta', 'CGTC')
#     ('gamma', 'CCGCC')
#     ('alpha (again - this is a duplicate entry to test the indexing code)', 'ACGTA')
#     ('delta', 'CGCGC')
#     """
#     # Skip any text before the first record (e.g. blank lines, comments)
#     for line in handle:
#         if line[0] == ">":
#             title = line[1:].rstrip()
#             break
#     else:
#         # no break encountered - probably an empty file
#         return
#     # Main logic
#     # Note, remove trailing whitespace, and any internal spaces
#     # (and any embedded \r which are possible in mangled files
#     # when not opened in universal read lines mode)
#     lines = []
#     for line in handle:
#         if line[0] == ">":
#             yield title, "".join(lines).replace(" ", "").replace("\r", "")
#             lines = []
#             title = line[1:].rstrip()
#             continue
#         lines.append(line.rstrip())
#     yield title, "".join(lines).replace(" ", "").replace("\r", "")


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

def read_fasta_as_idx(path):
    with safe_open(ref_fasta, 'r') as fasta:
        refseq = [(id.split(' ')[0], seq) for id, seq in SimpleFastaParser(fasta)]
    return refseq
    with safe_open(path, "r") as f:
        lines = numba.typed.List(f.readlines())
        yield do_the_thing(lines)
        yield do_the_thing(f)
        yield seq_id, sequence

def read_mutation_rates(mutation_rates):
    if type(mutation_rates) == np.array:
        # if user passed an array of mutation rates directly to the calc_Pi function, then they're all set
        pass
    elif type(mutation_rates) == str: # if they gave a file, import file
        suffix = mutation_rates.split('.')[0]
        if suffix == 'npy':
            mutation_rates = np.load(mutation_rates).values
        elif suffix == 'json':
            mutation_rates = pd.read_json(mutation_rates).values
        elif suffix == 'csv':
            mutation_rates = pd.read_csv(mutation_rates).values
        elif suffix == 'tsv':
            mutation_rates = pd.read_csv(mutation_rates, suffix='\t').values
        elif suffix.contains('xls'):
            mutation_rates = pd.read_excel(mutation_rates).values
        else:
            raise NotImplementedError
    else:
        mutation_rates = np.array(mutation_rates)
    assert mutation_rates.shape == (4,4), 'Mutation rate table must be a 4x4 table of mutation rates from ACGT(rows) to ACGT(columns)'
    return mutation_rates

def convert_to_onehot(seq, default_idx=(0,0,0,0)):
    u,inv = np.unique(seq, return_inverse = True)
    return np.array([tuple_dict.get(x,default_idx) for x in u], dtype=np.int16)[inv]

def convert_to_idx(seq, default_idx = -1):
    '''uses a fun trick from stackoverflow to quickly apply dict to np array'''
    u,inv = np.unique(seq, return_inverse = True)
    return np.array([nuc_dict.get(x,default_idx) for x in u], dtype=np.int16)[inv]

def create_combined_reference_Array(ref_fasta):
    with safe_open(ref_fasta, 'r') as fasta:
        refseq = [(id.split(' ')[0], seq.upper()) for id, seq in SeqIO.FastaIO.SimpleFastaParser(fasta)]
    # refseq = list(read_fasta_as_idx(ref_fasta))
    refseqArray, contig_starts, contig_coords = combine_contigs(refseq)
    refseqArray = convert_to_onehot(refseqArray)[np.newaxis,:,:]
    return refseqArray, contig_starts, contig_coords

def vcf_to_numpy_array_read_cts(vcf_file, refseqArray, contig_starts, maf=0, contig_coords=None):
    nucs = 'ACGTN'
    better_names_for_vcf_fields = {'calldata/AD':'AD', 'calldata/DP':'DP', 'calldata/RD':'RD',
                                    'variants/ALT':'alt', 'variants/CHROM':'contig', 'variants/POS':'pos', 'variants/REF':'ref'}
    fields_to_extract = list(better_names_for_vcf_fields.keys()) + ['samples']
    vcf = allel.read_vcf(vcf_file, fields=fields_to_extract,rename_fields=better_names_for_vcf_fields)#, fields=fields_to_extract, rename_fields=better_names_for_vcf_fields)
    sample_list = vcf['samples']
    var_pos = [pos+contig_starts[contig] for pos, contig in zip(vcf['pos'], vcf['contig'])]
    var_pos, var_pos_idx = np.unique(var_pos, return_index=True)
    read_cts = convert_to_onehot(vcf['ref'])[np.newaxis, :,:]
    vcf['RD'][vcf['RD'] == -1] = 0
    vcf['AD'][vcf['AD'] == -1] = 0
    vcf['RD'] = vcf['RD'].T[:,:,np.newaxis]
    vcf['AD'] = vcf['AD'].swapaxes(0,1)
    read_cts = read_cts*vcf['RD']#numba_mult(read_cts, vcf['RD'])
    for i, alts in enumerate(vcf['alt'].T):
        read_cts += convert_to_onehot(alts, default_idx = nucs.index('N'))[np.newaxis, :, :] * vcf['AD'][:,:,i:i+1]
    return read_cts, sample_list, var_pos, var_pos_idx

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
# q = numba.typed.List()
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
    vcf = pysam.VariantFile(vcf_file, threads = os.cpu_count()/2)
    test_record = next(vcf)
    sample_list = list(test_record.samples.keys())
    vcf.reset()
    num_records = get_num_var(vcf_file)
    read_cts = np.zeros((len(sample_list), num_records, 4),dtype=np.int16)
    var_pos = np.zeros(num_records)
    for i, r in enumerate(vcf):
        var_pos[i] = r.pos + contig_starts[r.contig]
        r_nuc_array = np.array([tuple_dict.get(x, (0,0,0,0)) for x in ((r.ref,)+r.alts)])
        read_cts[:,i,:] = np.stack([(np.array((s['RD'], s['AD']))[:,np.newaxis]*r_nuc_array[np.unique((0,)+tuple(a for a in s.allele_indices if a is not None)), :]).sum(0) for j,s in enumerate(r.samples.values())])
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
    #Combine contig sequences into one string, annotate contig locations within concatenated genome
    concatrefseq = "".join([seq[1] for seq in refseq])
    contigStarts = dict()
    contigCoords = dict()
    runningtally = 0
    for id, seq in refseq:
        contigStarts[id] = runningtally
        contigCoords[id] = (runningtally, runningtally+len(seq))
        runningtally += len(seq)
    refseqArray = np.array(list(concatrefseq))
    return refseqArray, contigStarts, contigCoords

def parseGTF(gtffile, contig_coords=None):
    '''given file location of gtf, and dictionary of starting locations
       of each chrom in a concatenated sequence, return dictionary of
       {gene product : numpy filter for concatenated sequence'''
    #Note to self: Stupid GTFs are stupid 1-indexed with stupid inclusive ends
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
        custom_items = {k:v for k,v in custom_items}
        mandatory_items = mandatory_items[:-1]
        contig_name = mandatory_items[0]
        if contig_name not in gene_coords.keys():
            gene_coords[contig_name] = OrderedDict()
        annotation_type = mandatory_items[2]
        start = int(mandatory_items[3]) - 1 # adding the -1 here to convert to 0 indexing
        stop = int(mandatory_items[4]) # not adding -1 because, while GTF is 1-indexed, its inclusive-ended. Converting to standard 0-indexing would mean 1-10 in GTF is equivelent to [0:10]
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
        gene_coords[chrom] = OrderedDict(sorted(gene_coords[chrom].items(), key=lambda item: item[1][0][0]))
    return gene_coords, transcript_to_gene_id, id_to_symbol