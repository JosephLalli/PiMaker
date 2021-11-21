#!/bin/bash

FLUDIR="/mnt/d/projects/pimaker/test_data/influenza"
VCF="$FLUDIR/Orchards_H3N2_a.bcf.gz"
GTF="$FLUDIR/A_Singapore_INFIMH-16-0019_2016_antigenic_w_glycosylation.gtf"
WORKINGDIR="$FLUDIR/snpgenie"


for FASTA in *.fasta
do
    CONTIG=${FASTA%.*}
    bcftools view $VCF $CONTIG > $WORKINGDIR/$CONTIG.vcf
    grep -w $CONTIG $GTF > $WORKINGDIR/$CONTIG.gtf
    mkdir $WORKINGDIR/$CONTIG
    CMD="perl snpgenie/snpgenie.pl --vcfformat 4 --snpreport $WORKINGDIR/$CONTIG.vcf --fasta $FASTA --gtffile $WORKINGDIR/$CONTIG.gtf --workdir $WORKINGDIR/$CONTIG"
    $CMD &
done