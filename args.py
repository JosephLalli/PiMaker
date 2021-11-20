# vcf_file, ref_fasta, gtf_file=None, maf=0.01, mutation_rates=None, 
#            rollingWindow=None, synon_nonsynon=False, include_stop_codons=False,
#            binsize=int(1e6), num_processes=1

import argparse
import os


#determine command line arguments and get path
parser = argparse.ArgumentParser(description='Tool to calculate diversity statistics from sequenced populations')
parser.add_argument('-r',metavar='ref', type=str, help="Reference fasta file")
parser.add_argument('-v',metavar='vcf', type=str, help="Combined VCF file containing called mutations from all sequencing runs")
parser.add_argument('-g',metavar='gtf', type=str, help="GTF file containing locations of all genes and transcripts of interest in the reference fasta file")
parser.add_argument('--maf',metavar='minimum allele frequency', type=float, help="Minimum intra-sample frequency of variants to be considered when calculating diversity")
parser.add_argument('--mutation_rates', metavar='mutation rates', type=str, help="Location of csv file containing empirically defined mutation rates. In the absence of specified mutation rates, PiMaker assumes all mutations are equally likely to occur as in Nei & Gojobori, 1986")
parser.add_argument('--rolling_window', metavar='width in codons of rolling window', type=int, help="How wide should the rolling window be for a rolling window analysis of pi/piN/piS")
parser.add_argument('--pi_only', metavar='do not run coding status specific analysis', type=bool, help="Most of the processing time for this script is spent on calculating synonymous and nonsynonymous-specific mutations. Use this flag to skip this analysis and just calculate pi.")
parser.add_argument('--include_stop_codons', metavar='width in codons of rolling window', type=bool, help="How wide should the rolling window be for a rolling window analysis of pi/piN/piS")


parser.add_argument('-t',metavar='threads', type=int,help="Number of processes to use while performing calculations",default=os.cpu_count())

if len(sys.argv[1:]) == 0:
		parser.print_help()
		parser.exit()
args = parser.parse_args()
numThreads = args.t
configFile = args.c
if args.debug:
	startingstep = ["trim", "initial_mapping", "indexing", "norm_coverage", "consensus", "snpcalling", "annotation", "snpgenie"].index(args.debug)
else:
	startingstep = 0