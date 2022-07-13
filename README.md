# PiMaker
Tool to calculate nucleotide diversity statistics in large pooled population datasets.

## Installation instructions

To install, simply navigate to your project's folder and clone this repository using the following command:

    git clone https://github.com/JosephLalli/PiMaker

You can then import pimaker into your project as you wish.

If you would like to use the command line, please run:

    chmod +x /path/to/pimaker.py

Further instructions on use are under development, though you are encouraged to look through the code for function documentation.

### Example use:
    VCF="path/to/vcf/file/of/all/samples"
    REF="path/to/fasta/reference/VCF/was/called/against"
    GTF="path/to/GFF3/or/GTF/annotation/file"

    optional:
    RATES='path_to_csv_file/of_4x4_matrix_(ACGT ordered)/of_relative_all_to_all_mutation_rates.csv'

    run command:
    python pimaker.py -v $VCF -g $GTF -r $REF --mutation_rates $RATES -t 8

### Memory management
PiMaker is designed to load all data being computed into memory. If this causes out of memory errors, we suggest altering the chunksize value to reduce the memory usage of the data that each thread is processing. To do this, change the "--chunk_size" value. Default is 1000000 bases per chunk. Increasing this value will moderately increase speed. Reducing this value will reduce peak memory usage almost linearly.

