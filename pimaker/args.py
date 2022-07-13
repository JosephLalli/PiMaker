"""
Functions to handle command line arguments.

Args contains both a function to collect command line arguments (get_parser)
and a class stolen from stackoverflow to format the help function output
in a user-friendly manner.

"""

import argparse
import shutil
import os
from version import __version__


def get_parser():
    """
    Creates argparse parser object. This object gets and organizes command line
    arguments, and formats help text.
    """
    parser = argparse.ArgumentParser(description='Tool to calculate diversity statistics from sequenced populations.',
                                     formatter_class=MyHelpFormatter, add_help=False)
    parser.add_argument('--version', action='version', version=f'Pimaker {__version__}',
                        help='Show PiMaker\'s version number and exit')
    parser.add_argument('-h', '--help', action='help',
                        help='Show this help message and exit')
    parser.add_argument('-v', '--vcf', required=True, type=str,
                        help='''Combined VCF file containing called mutations from all sequencing runs''')
    parser.add_argument('-r', '--ref_fasta', required=True, type=str,
                        help='''Reference fasta file''')
    parser.add_argument('-g', '--gtf', required=True, type=str,
                        help='''GTF file containing locations of all genes and transcripts of interest in the
                                reference fasta file''')
    parser.add_argument('-o', '--output', required=True, type=str, default='pimaker/result',
                        help='''Location of output files. Sample, Gene, and Site specific results will each have
                        this prefix + \'_gene.csv\', etc. attached. defaults to \'pimaker/result\'''')
    # parser.add_argument('--FST', type=bool, action='store_true',
    #                     help='''Calculate pairwise FST, synonymous FST, and nonsynonymous FST for all sample pairs.''')
    parser.add_argument('--maf', type=float, default=0.0,
                        help='''Minimum intra-sample frequency of variants to be considered when calculating
                                diversity''')
    parser.add_argument('--mutation_rates', type=str, default=None,
                        help='''Location of csv file containing empirically defined mutation rates. In the
                                absence of specified mutation rates, PiMaker assumes all mutations are equally
                                likely to occur as in Nei & Gojobori, 1986''')
    # parser.add_argument('--rolling_window', type=int, default=0,
    #                     help='''How wide should the rolling window be for a rolling window analysis of pi/piN/piS''')
    parser.add_argument('--pi_only', action='store_true',
                        help='''Most of the processing time for this script is spent on calculating synonymous
                                and nonsynonymous-specific mutations. Use this flag to skip this analysis and just
                                calculate pi''')
    parser.add_argument('--include_stop_codons', action='store_true',
                        help='''Nei and Gojobori 1986 assume that mutations to stop codons cause a drop to zero fitness
                                and therefore may be ignored. For some populations (especially rapidly adaptive populations), this assumption may
                                not be valid. By default follows Nei and Gojobori rules; use this option to consider stop
                                codons as nonsynonymous mutations.''')
    parser.add_argument('--chunk_size', type=int,
                        help='''Number of nucleotides to process at a time per thread. Higher chunk_size equals
                                faster performance with more memory usage. Lower chunk_size results in (slightly) slower
                                performance with less memory usage. Default value is 1,000,000 nucleotides per analyzed
                                chunk. Please adjust this first if you are encoutering errors due to high memory usage''', default=int(1e6))
    parser.add_argument('-t', '--threads', type=int, default= 1,
                        help='''Number of processes to use while performing calculations''')
    return parser


class MyHelpFormatter(argparse.HelpFormatter):
    """
    This is a custom formatter class for argparse. It allows for some custom formatting,
    in particular for the help texts with multiple options (like bridging mode and verbosity level).
    Stolen from http://stackoverflow.com/questions/3853722.
    """
    def __init__(self, prog):
        terminal_width = shutil.get_terminal_size().columns
        os.environ['COLUMNS'] = str(terminal_width)
        max_help_position = min(max(24, terminal_width // 3), 40)
        super().__init__(prog, max_help_position=max_help_position)

    def _get_help_string(self, action):
        help_text = action.help
        if action.default != argparse.SUPPRESS and 'default' not in help_text.lower() and \
                action.default is not None:
            help_text += ' (default: ' + str(action.default) + ')'
        return help_text