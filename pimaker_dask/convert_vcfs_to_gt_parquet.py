import allel
import pandas as pd
import numpy as np
import sys
import os
import glob
import pyarrow as pa
import pyarrow.parquet as pq

# python script to extract genotypes


def convert_hdf5_to_parquet(h5_file, parquet_file, chunksize=100000):
    # Shamelessly stolen from https://stackoverflow.com/questions/46157709/converting-hdf5-to-parquet-without-loading-into-memory
    stream = pd.read_hdf(h5_file, chunksize=chunksize)

    for i, chunk in enumerate(stream):
        print("Chunk {}".format(i))

        if i == 0:
            # Infer schema and open parquet file on first chunk
            parquet_schema = pa.Table.from_pandas(df=chunk).schema
            parquet_writer = pq.ParquetWriter(parquet_file, parquet_schema, compression='snappy')

        table = pa.Table.from_pandas(chunk, schema=parquet_schema)
        parquet_writer.write_table(table)

    parquet_writer.close()


def get_vcf_as_df(vcf_file, region=None):
    vcf = allel.read_vcf(vcf_file, fields=fields_to_extract, rename_fields=better_names_for_vcf_fields,
                         tabix=vcf_file + '.tbi', log=sys.stderr, region=region)

    # PAR1 = {'chrX':(10001, 2781479), 'chrY':(10001, 2781479)}
    # PAR2 = {'chrX':(155701383, 156030895), 'chrY':(56887903, 57217415)}

    # par1_ix = (vcf['contig']=='chrX') & (vcf['pos'] >= PAR1['chrX'][0]) & (vcf['pos'] <= PAR1['chrX'][1])
    # par2_ix = (vcf['contig']=='chrX') & (vcf['pos'] >= PAR2['chrX'][0]) & (vcf['pos'] <= PAR2['chrX'][1])

    # par1_ix = ((vcf['contig']=='chrY') & (vcf['pos'] >= PAR1['chrY'][0]) & (vcf['pos'] <= PAR1['chrY'][1])) | par1_ix
    # par2_ix = ((vcf['contig']=='chrY') & (vcf['pos'] >= PAR2['chrY'][0]) & (vcf['pos'] <= PAR2['chrY'][1])) | par2_ix

    # par_ix = par1_ix | par2_ix

    # vcf['contig'][np.where(par_ix)[0]] = 'chrPAR'

    gts = np.sum(np.where(vcf['GT'] < 0, 0, vcf['GT']), axis=2)
    gts = pd.DataFrame(gts)
    gts.columns = vcf['samples']

    gts['snp_names'] = [':'.join((w.replace('chr', ''), x, y, z))
                        for w, x, y, z in zip(vcf['contig'], vcf['pos'].astype(str), vcf['ref'], vcf['alt'][:, 0])]
    gts[['chrom', 'pos', 'ref', 'alt']] = gts.snp_names.str.split(':', expand=True)
    gts = gts.set_index('snp_names')

    return gts


glob_template = '*_imputed_merged_no_multi.vcf.gz'
ref_file_loc = os.path.join('data', 'e1_samples')
vcf_files = glob.glob(os.path.join(ref_file_loc, glob_template))

better_names_for_vcf_fields = {'calldata/AD': 'AD',  # 'calldata/DP': 'DP',
                               'variants/ALT': 'alt', 'variants/CHROM': 'contig',
                               'variants/POS': 'pos', 'variants/REF': 'ref',
                               'calldata/GT': 'GT'}

fields_to_extract = list(better_names_for_vcf_fields.keys()) + ['samples']


hdf_file = glob_template.replace('*', 'all').replace('.vcf.gz', '.hdf')
parquet_file = glob_template.replace('*', 'all').replace('.vcf.gz', '.pq')

for vcf_file in vcf_files:
    print (os.path.basename(vcf_file).split('_')[0])
    gts = get_vcf_as_df(vcf_file)
    gts.to_hdf(os.path.join(ref_file_loc, glob_template.replace('*', 'all').replace('.vcf.gz', '.hdf')),
               key='test', append=True, mode='a', format='t')

# Convert to parquet:
convert_hdf5_to_parquet(hdf_file, parquet_file)

# Loading gts:
gts = pd.read_parquet('all_imputed_merged_no_multi.parquet')
gts = gts.set_index(['chrom', 'pos', 'ref', 'alt'])


# Notes:
# Probably want no Y chrom, just an X. However, male samples get the 2nd allele in PAR regions as a

# in_PAR = np.bitwise_and(vcf['pos'] >= PAR1['chrX'][0],  vcf['pos'] < PAR1['chrX'][1])
# in_PAR2 =np.bitwise_and(vcf['pos'] >= PAR2['chrX'][0],  vcf['pos'] < PAR2['chrX'][1])
# in_PAR = np.bitwise_or(in_PAR, in_PAR2).sum()
