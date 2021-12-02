To convert popoolation-style files from sys import dont_write_bytecode
from numpy.lib.function_base import piecewise
from pandas.tseries.offsets import BMonthBegin
from TB into sync files,

def fill_with_ref(x):
    e = ['0','0','0','0','0','0']
    e['ATCG'.index(x.Ref)] = '1'
    e= ":".join(e)
    return x.fillna(e)

x = pd.read_excel('TB_noheader_popoolation.xlsx')
p = [pd.read_excel('TB_noheader_popoolation.xlsx', sheet) for sheet in x.sheet_names]
x = x.sort_values('Position')
x = reduce(lambda d1, d2: merge_the_two(d1, d2), p[:5])
x = x.apply(lambda q: fill_with_ref(q), axis=1)
x.to_csv('TB.sync', sep='\t', index=False)

To run benchmarking script:
sudo /usr/bin/time -f `date +"%F,%T",`+$FORMAT -o pimaker/benchmarks.txt --append $CMD; cat pimaker/benchmarks.txt

FORMAT="%C,%U,%S,%E,%PCPU,%X,%D,%K,%M,%F,%R,%W"
# Benchmarking notes:
# popoolation will calculate variance at position, or synon/nonsynon at position, but not both
# popoolation requires pileup as input, which limits ability to filter SNPs before basecalling
# snpgenie will only calculate one contig at a time
# npstat only calculates one contig, one sample at a time

Drosophila sample-specific pileup:
PILEUP="pimaker/test_data/drosophilia/CA_tu_13_spring.mel_mpileup.txt"
PILEUP="/mnt/d/ORCHARDS/H1N1/ORCHARDS_run19H3N2/A_Singapore_INFIMH-16-0019_2016/map_to_consensus/A_Singapore_INFIMH-16-0019_2016.mpileup"

CMD="perl $POPOOLATION1DIR/variance-at-position.pl --pileup $PILEUP --gtf $GTF --output $OUTDIR/output.txt --snp-output $OUTDIR/variance_at_pos_snp.out --pool-size 500 --measure pi --dissable-corrections --min-count 1"
CMD="perl $POPOOLATION1DIR/syn-nonsyn/Syn-nonsyn-at-position.pl --pileup $PILEUP --gtf $GTF --output $FLYDIR/popoolation1/syn-non-output.txt --codon-table $POPOOLATION1DIR/syn-nonsyn/codon-table.txt --nonsyn-length-table $POPOOLATION1DIR/syn-nonsyn/nsl_p1.txt --output $FLYDIR/popoolation1/syn-non-output.txt --snp-output $FLYDIR/popoolation1/snp-syn-non-output.txt --region-output $FLYDIR/popoolation1/genes.txt --measure pi --pool-size 10000 --max-coverage 1000000000 --dissable-corrections --min-count 1"
CMD="perl snpgenie/snpgenie.pl --snpreport $VCF --fasta $FASTA --gtffile $GTF --outdir $OUTDIR/snpgenie_results --vcfformat 4" 

POPOOLATION1DIR="/mnt/d/projects/pimaker/popoolation/popoolation_1.2.2"
OUTDIR="test_data/influenza/popoolation1"
FLUDIR="/mnt/d/projects/pimaker/test_data/influenza"
FLYDIR="/mnt/d/projects/pimaker/test_data/drosophilia"
VCF="$FLUDIR/Orchards_H3N2_a.bcf.gz"
GTF="$FLUDIR/A_Singapore_INFIMH-16-0019_2016_antigenic_w_glycosylation.gtf"
FASTA="$FLUDIR/A_Singapore_INFIMH-16-0019_2016.fasta"
VCF="$FLYDIR/dest.PoolSeq.PoolSNP.001.50.10Nov2020.ann.vcf.gz"
GTF="$FLYDIR/dmel-all-r6.12.gtf"
FASTA="$FLYDIR/dmel-all-chromosome-r6.12.fasta.gz"
WORKINGDIR="$FLUDIR/snpgenie"
CMD_BEFORE_RUNNING_SNPGENIE="awk -F "|" '/^>/ {close(F); ID=$1; gsub("^>", "", ID); gsub("\r", "", ID); F=ID\".fasta\"} {print >> F}' $FASTA"
NPSTATDIR="/mnt/d/projects/pimaker/npstat"
NPSTATCMD="$NPSTATDIR/npstat -n 500 -l 1 -maxcov 200 -outgroup $FASTA $PILEUP"


CMD="$POPOOLATION1DIR/syn-nonsyn/Syn-nonsyn-at-position.pl --pileup $PILEUP --gtf $GTF --codon-table $POPOOLATION1DIR/syn-nonsyn/codon-table.txt --nonsyn-length-table $POPOOLATION1DIR/syn-nonsyn/nsl_p1.txt --output $OUTDIR/syn-non-output.txt --snp-output $OUTDIR/syn-non-output.txt --region-output $OUTDIR/genes.txt --measure pi --pool-size 10000 --max-coverage 1000000000 --dissable-corrections --min-count 1"
CMD="python3 pimaker -o pimaker/test_data/influenza/pimaker_results -v $VCF -r $FASTA -g $GTF --threads 8"
CMD="python3 pimaker -o pimaker/test_data/drosophilia/pimaker_results -v pimaker/test_data/drosophilia/dest.PoolSeq.PoolSNP.001.50.10Nov2020.ann.vcf.gz -r pimaker/test_data/drosophilia/dmel-all-chromosome-r6.12.fasta.gz -g pimaker/test_data/drosophilia/dmel-all-r6.12.gtf.gz --threads 6"
Plan:
run my stuff on flu, tb, drosophila
run snpgenie on flu, drosophila
run popoolation on flu, drosophila

only calc pi, piN, piS

Chart times, chart accuracy
toss in some flu stuff? manhattan plot of drosophila piN/piS?