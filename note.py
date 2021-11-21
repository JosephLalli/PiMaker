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
sudo /usr/bin/time -f `date +"%F,%T",`+$FORMAT -o benchmarks.txt --append $CMD; cat benchmarks.txt


# Benchmarking notes:
# popoolation will calculate variance at position, or synon/nonsynon at position, but not both
# popoolation requires pileup as input, which limits ability to filter SNPs before basecalling
# snpgenie will only calculate one contig at a time



CMD="perl popoolation/popoolation_1.2.2/variance-at-position.pl --pileup $PILEUP --gtf $GTF --output test_data/influenza/popoolation1/output.txt --pool-size 10000 --measure pi"
CMD="perl popoolation/popoolation_1.2.2/syn-nonsyn/Syn-nonsyn-at-position.pl --pileup $PILEUP --gtf $GTF --output test_data/influenza/popoolation1/output.txt --pool-size 10000 --measure pi"
CMD="perl snpgenie/snpgenie.pl --snpreport $VCF --fasta $FASTA --gtffile $GTF --outdir $FLUDIR/snpgenie_results" 

POPOOLATION1DIR="popoolation/popoolation_1.2.2"
OUTDIR="test_data/influenza/popoolation1"
FLUDIR="/mnt/d/projects/pimaker/test_data/influenza"
VCF="$FLUDIR/Orchards_H3N2_a.bcf.gz"
GTF="$FLUDIR/A_Singapore_INFIMH-16-0019_2016_antigenic_w_glycosylation.gtf"
FASTA="$FLUDIR/A_Singapore_INFIMH-16-0019_2016.fasta"
WORKINGDIR="$FLUDIR/snpgenie"
CMD_BEFORE_RUNNING_SNPGENIE="awk -F "|" '/^>/ {close(F); ID=$1; gsub("^>", "", ID); gsub("\r", "", ID); F=ID\".fasta\"} {print >> F}' $FASTA"

CMD="$POPOOLATION1DIR/syn-nonsyn/Syn-nonsyn-at-position.pl --pileup $PILEUP --gtf $GTF --codon-table $POPOOLATION1DIR/syn-nonsyn/codon-table.txt --nonsyn-length-table $POPOOLATION1DIR/syn-nonsyn/nsl_p1.txt --output $OUTDIR/syn-non-output.txt --snp-output $OUTDIR/syn-non-output.txt --region-output $OUTDIR/genes.txt --measure pi --pool-size 10000 --max-coverage 1000000000 --dissable-corrections --min-count 1"

CMD="python3 pimaker -o pimaker/test_data/drosophilia/pimaker_results -v pimaker/test_data/drosophilia/dest.PoolSeq.PoolSNP.001.50.10Nov2020.ann.vcf.gz -r pimaker/test_data/drosophilia/dmel-all-chromosome-r6.12.fasta.gz -g pimaker/test_data/drosophilia/dmel-all-r6.12.gtf.gz --threads 6"
Plan:
run my stuff on flu, tb, drosophila
run snpgenie on flu, drosophila
run popoolation on flu, drosophila

only calc pi, piN, piS

Chart times, chart accuracy
toss in some flu stuff? manhattan plot of drosophila piN/piS?