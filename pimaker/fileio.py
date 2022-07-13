"""
Input/output functions for PiMaker.

This module contains functions for obtaining data from reference fasta, gtf files, and vcf files.

"""


import numpy as np
import pandas as pd
import allel
import copy
import pathlib
import gzip
import contextlib
from collections import OrderedDict
from tqdm import tqdm
import glob
# from intervaltree import Interval, IntervalTree
import os

nucs = 'ACGTN'
nuc_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
nuc_array = np.concatenate((np.eye(4, dtype=np.int16), np.zeros((1, 4), dtype=np.int16)))
array_dict = {n: a for n, a in zip(nucs, nuc_array)}
nuc_tuple = tuple(map(tuple, nuc_array))
tuple_dict = {nucs[i]: t for i, t in enumerate(nuc_tuple)}


@contextlib.contextmanager
def safe_open(filepath, rw):
    '''Checks if file has .gz in suffix, and opens
       file with appropriate tool (gzip or std open)'''
    if '.gz' in pathlib.Path(filepath).suffixes:
        file = gzip.open(filepath, rw + 't')
    else:
        file = open(filepath, rw)
    try:
        yield file
    finally:
        file.close()


def setup_environment(output_file_prefix, cache_folder=None, log_file=None):
    """
    Ensures cache folder and output folder exists.

    For the cache folder and the output folder, this function initially
    checks to see if the output path exists. If it doesnt, it creates the
    output path.
    It repeats this for the folder that contains our cache.
    Finally, it also ensures the path to the log file exists.

    Args:
        output_file_prefix: path to output folder with common filename prefix
                            (eg outputfolder/prefix, to which '_genes.tsv' etc
                            will be attached).
        cache_folder: folder in which pre-processed input data is stored. Optional.
    Returns:
        output_file_prefix: new path to output folder w/ attached common outfile
                            prefix
        cache_folder: new path to cache_folder. Returns None if cache_folder not
                      in arguments.
    """

    ## TODO: handle case where output already exists. Overwrite? Create new folder?
    output_folder = os.path.dirname(output_file_prefix)
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    output_file_prefix = os.path.join(output_folder, os.path.basename(output_file_prefix))

    if cache_folder is None:
        cache_folder = os.path.join(output_file_prefix, 'tmp')

    if not os.path.isdir(cache_folder):
        os.makedirs(cache_folder)

    ## TODO: implement logging; make sure logger doesn't do this automatically
    # if log_file and not os.path.isdir(log_file):
    #     os.mkdir(log_file_dir)

    return output_file_prefix, cache_folder


def get_num_var(vcf_file, contig=False):
    """
    Wraps bcftools index function. Allows user to quickly obtain the
    number of variants in a vcf file. Especially useful if trying to know
    how many variants are left to process while iterating over the variant
    entries of the vcf file.

    Args:
        vcf_file: File name of vcf_file
        contig: If True, this function will return a dictionary of the number
                of variants per contig.
    Returns:
        The number of variant entries in the VCF file.
        If contig is true, returns a dictionary of
        contig:number of variants in contig
    """
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
    """
    Wraps bcftools query -l function.
    Args:
        vcf_file: filename of the vcf file of interest.
    Returns:
        list of the names of samples included in vcf_file
    """
    retval = os.popen(
        f'bcftools query -l {vcf_file}').read()
    return retval.strip().split('\n')


def SimpleFastaParser(handle):
    """
    Iterates over Fasta records as string tuples.
    Identical (indeed, copied) from Bio.SeqIO's SimpleFastaParser.

    Args:
        handle: input stream opened in text mode
    Yields:
        For each record a tuple of two strings is returned, the FASTA title
        line (without the leading '>' character), and the sequence (with any
        whitespace removed). The title line is not divided up into an
        identifier (the first word) and comment or description.

        e.g.
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


def read_mutation_rates(mutation_rates):
    """
    Handles accepting a variety of formats which a user could specify
    empirically defined mutation rates to use when calculating
    synon/nonsynon sites.

    No matter the format, mutation rates must either be None or a
    4 x 4 array of numerical rates of mutation representing relative
    frequency of mutation from ACGT(rows) to ACGT(columns).

    Args:
        mutation_rates: Either a path to a file containing a 4x4 array of
        numerical mutation rates, or an array-like object representing a
        4x4 array of numerical mutation rates.
    Returns:
        A 4x4 numpy array of np.float64 mutation rates.
    Raises:
        TypeError if input cannot be coaxed into a numpy array of mutation
        rates of dtype float64.
        AssertionError if the resulting numpy array is not a 4x4 array.
        Prints error message, then raises original exception.
    """
    # If not using mutation_rates, return None
    if mutation_rates is None:
        return None

    # Next, load mutation rates from input
    if type(mutation_rates) == str:  # if they gave a file, import file
        suffix = mutation_rates.split('.')[-1]
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
        # Attempt to convert input into 4x4 array of floats.
        try:
            mutation_rates = np.asarray(mutation_rates, dtype=np.float64)
            assert mutation_rates.shape == (4, 4)
        except (TypeError, AssertionError) as error:
            print ('Mutation rate table must be a 4x4 table of numerical' +
                   'mutation rates from ACGT(rows) to ACGT(columns).')
            raise error
    return mutation_rates


def convert_to_onehot(seq, default_idx=(0, 0, 0, 0)):
    """
    Converts an ACTGN genetic sequence to a one-hot np.int16 representation
    of the sequence of size 4 x sequence length.
    """
    u, inv = np.unique(seq, return_inverse=True)
    return np.array([tuple_dict.get(x, default_idx) for x in u], dtype=np.int16)[inv]


def load_references(ref_fasta):
    """
    Reads fasta file of reference sequence contigs into one long one-hot
    representation of the concatenated reference genome.

    Args:
        ref_fasta: A .fasta or .fasta.gz file containing reference sequences
        for every contig in the variant file. Ideally the same reference
        sequence that variants were called against.
    Returns:
        refseq
    """
    with safe_open(ref_fasta, 'r') as fasta:
        refseq = [(id.split(' ')[0], seq.upper()) for id, seq in SimpleFastaParser(fasta)]
    ref_seq_array, contig_coords = combine_contigs(refseq)
    ref_seq_array = convert_to_onehot(ref_seq_array)[np.newaxis, :, :]
    return ref_seq_array, contig_coords#, contig_interval_tree


def vcf_to_numpy_array_read_cts(vcf_file, contig_coords, region=None, maf=0):
    """
    Utilizing scikit-allel, reads a vcf file and generates a numpy array of
    read counts, optionally applying a region and minor frequency filter.

    Also returns the corresponding indicies of variants in a
    concatenated reference sequence. If the vcf has split multiallelic sites
    into separate variant entries, this function will merge the data at those
    sites. Optionally, applies a minor allele frequency cutoff value to the
    read count data; variants below that frequency are ignored.

    Args:
        vcf_file:
            Path to the vcf formatted file of variant calls. Should
            contain all samples/pools being measured.
        contig_coords:
            A dictionary of contig_ids (as named in the vcf_file) to
            (start, stop) indices of that contig in the concatenated reference
            sequence.
        region:
            Optional, default None. A tabix-style string of the form
            'contig_id:start-stop' specifying the genetic region you want to
            extract variants for. If None, will return all variants in the VCF.
            Important! Start and Stop coordinates are tabix-style, and so must
            be 1-based. For example: for the sequence AGTC, seq:1-3 will yield
            variants from the portions of that sequence whose reference alleles
            are AGT.

    Returns:
        A tuple containing:
            A (number of samples x number of variant sites x 4) array of the
            number of reads mapping to a given nucleotide site (ordered A/C/G/T
            on the third axis)
            An array of the index of each variant site within the concatenated
            reference sequence.
    """
    better_names_for_vcf_fields = {'calldata/AD': 'AD', 'calldata/DP': 'DP', 'calldata/RD': 'RD',
                                    'variants/ALT': 'alt', 'variants/CHROM': 'contig',
                                    'variants/POS': 'pos', 'variants/REF': 'ref'}
    fields_to_extract = list(better_names_for_vcf_fields.keys()) + ['samples']
    index = glob.glob(vcf_file + '.*')
    if len(index) == 0:
        raise AttributeError('VCF file must be indexed.')
    vcf = allel.read_vcf(vcf_file, fields=fields_to_extract, rename_fields=better_names_for_vcf_fields, region=region, tabix='tabix')
    if vcf is None:
        return None, None
    vcf['pos'] -= 1  # pos is 1-indexed in allel
    if region is None:
        vcf['pos'] = np.array([pos + contig_coords[contig][0] for contig, pos in zip(vcf['contig'], vcf['pos'])])
        og_var_pos = vcf['pos']
    else:
        og_var_pos = vcf['pos'] + contig_coords[0]

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
    read_cts = read_cts.astype(np.uint32)
    var_pos = var_pos.astype(np.uint64)
    return read_cts, var_pos


def merge_split_multiallelics(values, index, action):
    """
    Merges data from split multiallelic variants.

    Scikit-allel handles multiallelic variants very smoothly, but it does not
    deal well with alleles that are divided between two entries (e.g. A/T and
    A/G). If alleles are split, this function will find the unique variant
    positions and group values that are located at the same position. The
    action taken to group these variant values needs to be specified. Oddly
    enough, pandas handles this kind of groupby work very well, and so we
    employ it here.

    Args:
        values:
            Set of values from a allel vcf object (e.g., ref counts, or Alt
            nucleotides)
        index:
            Position of each variant entry within a concatenated reference
            allele.
        action:
            Method of merging two datapoints from the same genomic location.
            'sum' specifies adding the two values together.
            'reshape counts' specifies reshaping the n x 3 array of alt allele
            counts provided by allel into an n-duplicates x 3 array of merged
            alt counts.
            'reshape nucs' specifies merging the alt alleles output by allel
            into a tuple contining each alt nucleotide at that position.
    Returns:
        A numpy array of the appropriately formatted values.
    """
    if action not in ['sum', 'reshape counts', 'reshape nucs']:
        raise ValueError('merge_split_multiallelics requires a specified action to handle the merge')

    values_df = pd.DataFrame(values, index=index).reset_index().groupby('index')
    if action == 'sum':
        return values_df.apply(lambda x: x[x.columns[1:]].sum()).values
    elif action == 'reshape counts':
        df_of_arrays = values_df.apply(lambda x: np.array([tuple(x[c]) + (-1,) * (4 - len(x[c])) for c in x.columns[1:]]))
        return np.stack(df_of_arrays.values)
    elif action == 'reshape nucs':
        values_df = values_df.apply(lambda x: pd.Series([tuple(x[0].values) + ('',) * (3 - len(x))][0]))
        return values_df.values


def save_tmp_chunk_results(contig, chunk_id, chunk_sample_pi, chunk_gene_pi, chunk_site_pi, var_site_positions, cache_folder):
    """
    Saves chunk results to temp files on the HDD to save memory during data
    processing.

    Args:
        contig:
            Name of the contig that was processed. Used to generate temp file
            name.
        chunk_id:
            ID number of the chunk that was processed. Used to generate temp
            file name.
        chunk_sample_pi:
            A list of tuples containing the per-sample pi results.
        chunk_gene_pi:
            A list of tuples containing the per-gene pi results.
        chunk_site_pi:
            A list of tuples containing the per-site pi results.
        var_site_positions:
            The within-contig positions of the variant sites in this chunk.
            Used to annotate chunk_site_pi results.
        cache_folder:
            Folder to save temporary files in.
    Returns:
        A tuple of filenames pointing to the save locations of the temp sample
        data, gene data, and site data.
    """

    sample_pi_columns = list(['sample_id', 'contig', 'chunk_id', 'chunk_len', 'stat_name', 'stat'])
    gene_pi_columns = list(['sample_id', 'contig', 'chunk_id', 'transcript_id', 'transcript', 'gene_id', 'gene_symbol',
                            'transcript_len', 'transcript_len_no_overlap', 'pi', 'piN', 'piS', 'N_sites', 'S_sites',
                            'pi_no_overlap', 'piN_no_overlap', 'piS_no_overlap', 'N_sites_no_overlap', 'S_sites_no_overlap'])
    site_pi_columns = tuple(['sample_id', 'contig', 'stat_name'])

    chunk_sample_pi_df = pd.DataFrame(chunk_sample_pi, columns=sample_pi_columns)
    chunk_gene_pi_df = pd.DataFrame(chunk_gene_pi, columns=gene_pi_columns)
    chunk_site_pi_df = pd.DataFrame(chunk_site_pi, columns=(site_pi_columns + tuple(var_site_positions)))

    sample_filename = f'{cache_folder}/{contig}_{chunk_id}_sample.csv'
    gene_filename = f'{cache_folder}/{contig}_{chunk_id}_gene.csv'
    site_filename = f'{cache_folder}/{contig}_{chunk_id}_site.csv'

    chunk_sample_pi_df.to_csv(sample_filename, index=False)
    chunk_gene_pi_df.to_csv(gene_filename, index=False)
    chunk_site_pi_df.to_csv(site_filename, index=False)

    return sample_filename, gene_filename, site_filename


def combine_contigs(refseq):
    """
    Combines each contig's sequences into one string, recording the coordinates
    of each contig within the concatenated genome.

    Args:
        refseq:
            A list of tuples of the format (contig_name, sequence) containing
            the genetic sequence of each contig being analyzed.
    Returns:
        A tuple containing:
            - The concatenated genome in a char numpy array format
            - A dictionary of contig names to start positions of the contig
              within the concatenated genome
            - A dictionary of contig names to (start, end) tuples containing
              the coordinates of the contig within the concatenated genome
            - A intervaltree object containing the start-end coordinates of 
              every contig
    """
    '''Combine contig sequences into one string, annotate contig locations within concatenated genome'''
    concatrefseq = "".join([seq[1] for seq in refseq])
    contigCoords = dict()
    runningtally = 0
    # contig_interval_tree = IntervalTree()
    for id, seq in refseq:
        contigCoords[id] = (runningtally, runningtally + len(seq))
        # contig_interval_tree[runningtally:runningtally+len(seq)] = id
        runningtally += len(seq)
    ref_seq_array = np.array(list(concatrefseq))
    return ref_seq_array, contigCoords#, None #contig_interval_tree


def parse_gtf(gtffile, contig_coords=None):
    """
    Reads GTF formated file and returns a dictionary containing the coordinates
    of each exon in each coding region specified in the GTF.

    Args:
        gtffile:
            Path to GTF file.
        contig_coords:
            Optional, default None. Dictionary containing the start and end
            positions of each contig within a concatenated reference genome.
    Returns:
        Tuple containing:
            - A dictionary of the form contig : transcript_id : list of
              (start, stop) tuples containing the coordinates of each exon
              within the reference genome. If contig_coords are provided,
              exon coordinates are specified with regards to the concatenated
              reference sequence.
            - A dictionary of the form transcript_id : gene_id specifying each
              transcript's gene.
            - A dictionary of the form id : symbol specifying, for each gene or
              transcript id, the human-friendly gene or transcript symbol
              (aka the gene/transcript name).
            - transcript_interval_tree: tree of cds coordinates and the
              transcripts they belnog to.
            - gene_intervals: dictionary of transcripts to a tree of cds
              coordinates.
    """
    # Note to self: Stupid GTFs are stupid 1-indexed with stupid inclusive ends
    with safe_open(gtffile, 'r') as g:
        gtf = g.readlines()

    gene_coords = OrderedDict()
    # gene_intervals = OrderedDict()
    id_to_symbol = dict()
    transcript_to_gene_id = dict()
    # transcript_interval_tree = IntervalTree()

    if contig_coords is not None:
        for chrom in contig_coords.keys():
            gene_coords[chrom] = OrderedDict()
            # gene_intervals[chrom] = OrderedDict()

    for line in gtf:
        line = line.replace("/", "_")
        mandatory_items = line.split("\t")
        custom_items = [item.replace("\"", '').strip() for item in mandatory_items[-1].split(';') if not item.isspace() and len(item) != 0]
        custom_items = [tuple(item.split(' ')) for item in custom_items if (item != ('',)) and (item[0] != '#')]
        custom_items = {k: v for k, v in custom_items}
        mandatory_items = mandatory_items[:-1]
        contig_name = mandatory_items[0]
        if contig_name not in gene_coords.keys():
            gene_coords[contig_name] = OrderedDict()
            # gene_intervals[contig_name] = OrderedDict()

        annotation_type = mandatory_items[2]
        start = int(mandatory_items[3]) - 1  # adding the -1 here to convert to 0 indexing
        stop = int(mandatory_items[4])  # not adding -1 because, while GTF is 1-indexed, its inclusive-ended.
        if contig_coords is not None:   # Converting to standard 0-indexing would mean 1-10 in GTF is equivelent to [0:10]
            start += contig_coords[contig_name][0]
            stop += contig_coords[contig_name][0]

        strand = mandatory_items[6]
        # interval_start, interval_stop = start, stop
        # interval_strand = 1

        if strand == '-':
            start, stop = stop, start
            # interval_strand = -1

        if annotation_type.lower() == "cds":
            if custom_items['transcript_id'] not in gene_coords[contig_name].keys():
                gene_coords[contig_name][custom_items['transcript_id']] = tuple()
                # gene_intervals[contig_name][custom_items['transcript_id']] = IntervalTree()
            transcript_to_gene_id[custom_items['transcript_id']] = custom_items['gene_id']
            id_to_symbol[custom_items['gene_id']] = custom_items['gene_id']
            id_to_symbol[custom_items['transcript_id']] = custom_items['transcript_id']
            gene_coords[contig_name][custom_items['transcript_id']] += ((start, stop),)
            # transcript_interval_tree[interval_start:interval_stop] = custom_items['transcript_id']
            # gene_intervals[contig_name][custom_items['transcript_id']].add(Interval(interval_start, interval_stop, interval_strand))
    for chrom in gene_coords.keys():
        gene_coords[chrom] = OrderedDict(sorted(gene_coords[chrom].items(), key=lambda item: min(item[1][0][0], item[1][-1][-1])))
    return gene_coords, transcript_to_gene_id, id_to_symbol#, transcript_interval_tree, gene_intervals


# def retrieve_transcripts_in_region_interval_tree(start, stop, transcript_interval_tree, transcript_intervals):
#     # genes = OrderedDict()
#     # for gene, coords in gene_coords.items():
#     #     gene_start = min(coords[0][0], coords[-1][-1])
#     #     gene_end = max(coords[0][0], coords[-1][-1])
#     #     if gene_start > stop:  # if gene begins after region, stop iteration.
#     #         break
#     #     elif gene_start >= start:  # if gene begins after start of region and before end of region, record gene
#     #         genes[gene] = coords
#     #         if gene_end >= stop:  # if the gene begins within region but ends after region, adjust end of region.
#     #             stop = gene_end + 1
#     transcripts_in_region = IntervalTree(transcript_interval_tree[start:stop])
#     stop = transcripts_in_region.end()
#     transcript_coords = {t.data:transcript_intervals[t.data] for t in transcripts_in_region}
#     return transcript_coords, stop
#     # pass


def retrieve_transcripts_in_region(start, stop, gene_coords):
    """
    Returns the ids of the transcripts in a specified region.
    If the end of the region is between the 5' end and the 3' end of a
    transcript (including introns), also returns the nearest coordinate
    that is outside of any transcript.

    Used to define the start and end position of data chunks. By checking to
    see if a proposed end position is in the middle of a transcript, we ensure
    that all chunks contain all relevent sequence data for each transcript in
    that chunk.

    Args:
        start:
            Within-concatenated-sequence start position of chunk.
        stop:
            Within-concatenated-sequence end position of chunk.
        gene_coords:
            A dictionary of transcripts to lists of transcript exon coordinates.

    Returns:
        A tuple containing:
            - A dictionary of transcripts present between start and stop to the
              list of (start, stop) exon coordinates for that transcript.
            - A stop coordinate for the chunk that does not lie in the middle
              of a transcript.
    """
    genes = OrderedDict()
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


def pool_to_numpy(poolfile):
    """
    Converts a Popoolation pool file to a numpy array of read counts.
    Currently in development and untested. Do not use.
    """
    pool = pd.read_csv(poolfile, sep='\t')
    sample_list = pool.columns[2:]
    var_pos = pool.Position.astype(int)

    def col_to_nump(column):
        return np.stack(column.str.split(':').values).astype(int)

    read_cts = np.stack([col_to_nump(pool[sample])[:, :-2] for sample in sample_list])
    return read_cts, var_pos, sample_list


def vcf_to_sync(vcf_file, sync_file=None):
    """
    Converts a vcf file to a Popoolation pool file.
    Currently in development and untested. Do not use.
    """
    raise NotImplementedError
    import pysam
    if sync_file is None:
        sync_file = vcf_file.replace('.vcf', '.sync').replace('.bcf', '.sync').replace('.gz', '')
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
            contig_adjust = i  # assumes vcf sorted
        i = i - contig_adjust
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

        pool_fmt_list = [np.array2string(read_cts[contig][i, :, :].astype(int), separator=':', threshold=num_records[contig] * 2, edgeitems=num_records[contig] * 2).replace(' ', '').replace('[', '').replace(']', '').split(':\n') for i in range(len(sample_list))]
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
