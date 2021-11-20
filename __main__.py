# vcf_file, ref_fasta, gtf_file=None, maf=0.01, mutation_rates=None, 
#            rollingWindow=None, synon_nonsynon=False, include_stop_codons=False,
#            binsize=int(1e6), num_processes=1

import argparse
import shutil
import os, sys
from version import __version__


def get_parser():
    # determine command line arguments and get path
    parser = argparse.ArgumentParser(description='Tool to calculate diversity statistics from sequenced populations.',
                                     formatter_class=MyHelpFormatter, add_help=False)
    parser.add_argument('--version', action='version', version=f'Pimaker {__version__}',
                        help= 'Show PiMaker\'s version number and exit')
    parser.add_argument('-h', '--help', action='help',
                        help='Show this help message and exit')
    parser.add_argument('-v', '--vcf', required=True, type=str,
                        help='''Combined VCF file containing called mutations from all sequencing runs''')
    parser.add_argument('-r', '--ref_fasta', required=True, type=str,
                        help='''Reference fasta file''')
    parser.add_argument('-g', '--gtf',required=True, type=str,
                        help='''GTF file containing locations of all genes and transcripts of interest in the
                                reference fasta file''')
    parser.add_argument('--maf', type=float, default=0.0,
                        help='''Minimum intra-sample frequency of variants to be considered when calculating 
                                diversity''')
    parser.add_argument('--mutation_rates', type=str, default=None,
                        help='''Location of csv file containing empirically defined mutation rates. In the
                                absence of specified mutation rates, PiMaker assumes all mutations are equally
                                likely to occur as in Nei & Gojobori, 1986''')
    parser.add_argument('--rolling_window', type=int, default=0,
                        help='''How wide should the rolling window be for a rolling window analysis of pi/piN/piS''')
    parser.add_argument('--pi_only', type=bool, default=False,
                        help='''Most of the processing time for this script is spent on calculating synonymous
                        and nonsynonymous-specific mutations. Use this flag to skip this analysis and just
                        calculate pi''')
    parser.add_argument('--include_stop_codons', type=bool, default=False,
                        help='''Nei and Gojobori 1986 assume that mutations to stop codons do not randomly.
                        For some populations (especially rapidly adaptive populations), this assumption may
                        not be valid''')
    parser.add_argument('--binsize', type=int,
                        help='''Number of nucleotides to process at a time per thread. Higher binsize equals 
                        faster performance with more memory usage. Lower binsize results in (slightly) slower
                        performance with less memory usage. Default value is 1,000,000 nucleotides per analyzed
                        chunk. Please adjust this first if you are encoutering errors due to high memory usage''', default=int(1e6))
    parser.add_argument('-t', '--threads', type=int, default=os.cpu_count()-1,
                        help='''Number of processes to use while performing calculations''')
    return parser


class MyHelpFormatter(argparse.HelpFormatter):
    """
    This is a custom formatter class for argparse. It allows for some custom formatting,
    in particular for the help texts with multiple options (like bridging mode and verbosity level).
    http://stackoverflow.com/questions/3853722
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

# parameter checking
# def check_subsample_args(args):
#     if args.count < 2:
#         sys.exit('\nError: --count cannot be less than 2')
#     if args.count > 99:
#         sys.exit('\nError: --count cannot be greater than 99')


# def check_dotplot_args(args):
#     if args.res < 500 or args.res > 10000:
#         sys.exit('\nError: --res must be between 500 and 10000 (inclusive)')
#     if args.kmer < 8 or args.kmer > 100:
#         sys.exit('\nError: --res must be between 8 and 100 (inclusive)')

def main(args=None):
    print('Welcome to PiMaker!')
    parser = get_parser()
    arg_name_dict = {'vcf':'vcf_file', 'gtf':'gtf_file', 'threads':'num_processes'}
    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()
    else:
        args = parser.parse_args()
        args = vars(args)
        args = {arg_name_dict.get(k, k): v for k, v in args.items()}
    print (args)
    print ('\n')

    from timeit import default_timer as timer
    start = timer()
    from pimaker import calcPi
    results = calcPi(**args)
    stop = timer()
    print(stop - start)
    return None


if __name__ == '__main__':
    main()