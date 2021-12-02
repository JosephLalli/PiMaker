#!/usr/bin/env python
# coding: utf-8

# What am I trying to do? Trying to create something that, given a VCF/DF of SNPs, a reference sequence, and a GTF,
# it will determine the PiN/PiS by GTF category
import numpy as np
from Bio import SeqIO
import pandas as pd
from itertools import combinations, chain

SNPGenie_1_rules = True
adjust_for_mutation_rates = True
mutation_rates={  #source: Pauley, Procario, Lauring 2017: A novel twelve class fluctuation test reveals higher than expected mutation rates for influenza A viruses
    'A':{'C':0.41, 'G':4.19, 'T':0.16},
    'C':{'A':0.21,'G':0.12,'T':0.57},
    'G':{'A':0.89,'C':0.34,'T':0.75},
    'T':{'A':0.06,'C':3.83,'G':0.45}
}

translate = {
    'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
    'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
    'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
    'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
    'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
    'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
    'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
    'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
    'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
    'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
    'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
    'TAC':'Y', 'TAT':'Y', 'TAA':'*', 'TAG':'*',
    'TGC':'C', 'TGT':'C', 'TGA':'*', 'TGG':'W'}

def makeNumSitesDict(ignore_stop_codons=False):
    nucs = ['A','T','C','G']
    numsitespercodon = {codon:{0:{'s':0,'n':0},1:{'s':0,'n':0},2:{'s':0,'n':0}, 'all':{'s':0,'n':0}} for codon in translate.keys()}
    for codon, AA in translate.items():
        if AA == '*' and ignore_stop_codons:
            numsitespercodon[codon][i]['s'] = 0
            numsitespercodon[codon][i]['n'] = 0
            continue
        else:
            for i in range(3):
                for nuc in nucs:
                    mut = list(codon)
                    mut[i] = nuc

                    if codon[i] == nuc:
                        continue
                    if ignore_stop_codons and translate[''.join(mut)] == '*':
                        continue
                    elif AA == translate[''.join(mut)]:
                        if adjust_for_mutation_rates:
                            numsitespercodon[codon][i]['s'] += mutation_rates[codon[i]][nuc]
                        else:
                            numsitespercodon[codon][i]['s'] += 1/3
                    else:
                        if adjust_for_mutation_rates:
                            numsitespercodon[codon][i]['n'] += mutation_rates[codon[i]][nuc]
                        else:
                            numsitespercodon[codon][i]['n'] += 1/3
                num_sites_assigned = numsitespercodon[codon][i]['n'] + numsitespercodon[codon][i]['s']

                if num_sites_assigned > 0:
                    numsitespercodon[codon][i]['n'] /= num_sites_assigned
                    numsitespercodon[codon][i]['s'] /= num_sites_assigned
        numsitespercodon[codon]['all']['s'] = sum([numsitespercodon[codon][i]['s'] for i in range(3)])
        numsitespercodon[codon]['all']['n'] = sum([numsitespercodon[codon][i]['n'] for i in range(3)])
    return numsitespercodon

def calc_overlaps(codingCoords, coords_reference='contig', ignore=['PA-X', 'PB1-F2', 'HA_antigenic','HA_nonantigenic']):
    overlaps = dict()
    for contig, genedict in codingCoords.items():
        length = len(genedict)
        if length == 1:
            continue
        elif any([gene in ignore for gene in genedict.keys()]):
            for gene in ignore:
                try:
                    del codingCoords[contig][gene]
                except KeyError:
                    continue
        else:
            gene_spans = list(genedict.values())
            one_gene_sites = set(range(0,5000))
            
            for spans in gene_spans:
                gene_range = set()
                for start, end in spans:
                    gene_range = gene_range.union(set(range(start, end+1)))
                one_gene_sites = one_gene_sites.intersection(gene_range)
            one_gene_sites = sorted(list(one_gene_sites))
            
            #identify overlaps in 
            
            
            for gene, spans in genedict.items():
                ranges = list()
                start_of_gene = spans[0][0]
                start = start_of_gene
                end = 0
                for i in range(len(one_gene_sites)-1):
                    if one_gene_sites[i]+1 == one_gene_sites[i+1]:
                        continue
                    else: #if there's a gap in one_gene_sites
                        end = one_gene_sites[i]
                        if coords_reference=='contig':
                            ranges.append((start, end+1))
                        elif coords_reference=='in_concat_gene':
                            ranges.append((start-start_of_gene, end-start_of_gene+1))
                        else:
                            ranges.append((start-start_of_gene, end-start_of_gene+1))
                        start = one_gene_sites[i+1]
                        end = 0
                end = one_gene_sites[-1]
                if coords_reference=='contig':
                    ranges.append((start, end+1))
                elif coords_reference=='in_concat_gene':
                    ranges.append((start-start_of_gene, end-start_of_gene+1))
                else:
                    ranges.append((start-start_of_gene, end-start_of_gene+1))
                overlaps[gene] = ranges
    return overlaps

def makeSynonNonsynonSiteDictionary(ignore_stop_codons=False):
    '''creates dictionary to assist with calculating synon/nonsynon sites.
       output:
       [codon]:(3,4) np array of floats. 
       eg, for the codon "AAT", the array would be the number of synon/nonsynon sites in:
       [[AAT, CAT, GAT, TAT],
        [AAT, ACT, AGT, ATT],
        [AAA, AAC, AAG, AAT]]
       That way, multiplying array by the SNP frequencies will yield the number of sites in that codon
       '''
    nucs = ['A','C','G','T']
    synonSiteCount = {codon:np.zeros((3,4)) for codon in translate.keys()}
    nonSynonSiteCount = {codon:np.zeros((3,4)) for codon in translate.keys()}
    for codon in translate.keys():
        if translate[codon] == '*' and ignore_stop_codons:
            continue
        else:
            for i, refNuc in enumerate(codon):
                for j, nuc in enumerate(nucs):
                    tmpcodon=list(codon)
                    tmpcodon[i] = nuc
                    tmpcodon=''.join(tmpcodon)
                    synonSiteCount[codon][i,j] = numSitesDict[tmpcodon][i]['s']
                    nonSynonSiteCount[codon][i,j] = numSitesDict[tmpcodon][i]['n']

    return synonSiteCount, nonSynonSiteCount

def generateCodonSynonMutationFilters():
    '''return dictionary of codons and 7x3 nonsynon filters'''
    nucs = ['A','C','G','T']
    mutationID = {'AC':1,'CA':1, 'AG':2,'GA':2, 'AT':3,'TA':3, 'CG':4,'GC':4, 'CT':5,'TC':5, 'GT':6,'TG':6}

    #generate dictionary of codon to 3x7 nonsynon filter arrays:
    synonPiTranslate = {codon:np.zeros((3,7)) for codon in translate.keys()}
    nonSynonPiTranslate = {codon:np.zeros((3,7)) for codon in translate.keys()}
    for codon, npfilter in translate.items():
        nonSynonPiTranslate[codon][:,0] = 1
        synonPiTranslate[codon][:,0] = 1

        for n, refNuc in enumerate(codon):
            for nuc in nucs:
                if refNuc == nuc:
                    continue
                altCodon = codon[:n]+nuc+codon[n+1:]
                if translate[codon] != translate[altCodon]:
                    nonSynonPiTranslate[codon][n,mutationID[refNuc+nuc]] = 1
                elif translate[codon] == translate[altCodon]:
                    synonPiTranslate[codon][n,mutationID[refNuc+nuc]] = 1
    return synonPiTranslate, nonSynonPiTranslate

numSitesDict = makeNumSitesDict(ignore_stop_codons=SNPGenie_1_rules)

synonSiteCount, nonSynonSiteCount = makeSynonNonsynonSiteDictionary(ignore_stop_codons=SNPGenie_1_rules)
synonPiTranslate, nonSynonPiTranslate = generateCodonSynonMutationFilters()


def range_overlap(r1, r2):
    overlap = (max(r1[0],r2[0]), min(r1[1],r2[1]))
    if overlap[0] > overlap[1]:
        return None
    else:
        return overlap

def coord_overlap(gene_coords):
    ranges = chain(*gene_coords)
    overlaps = list()
    for span_a, span_b in combinations(ranges, 2):
        overlap = range_overlap(span_a, span_b)
        if overlap:
            overlaps.append(overlap)
    return overlaps

def get_nonoverlapping_in_gene_locations(contig, gene, codingCoords):
    overlaps = coord_overlap(codingCoords[contig].values())
    coding_length=0
    non_overlaps = list()
    for region in codingCoords[contig][gene]:
        exon_overlap = coord_overlap([[region], overlaps])
        exon_overlap = range(exon_overlap[0][0], exon_overlap[0][1])
        non_overlaps.append(np.array(list(set(range(region[0],region[1])).symmetric_difference(set(exon_overlap))))-(region[0]-coding_length))
        coding_length = coding_length - region[0] + region[1]
    return np.concatenate(non_overlaps).astype(np.int32)


def one_hot_encode(ACGT_array):
    mapping = dict(zip("ACGT", range(4)))
    num_seq = np.vectorize(mapping.__getitem__)(ACGT_array)
    one_hot = np.eye(4)[num_seq]
    return one_hot

def remove_overlap_regions(overlaps, mask):
    '''given codingcoords, contigStarts, a filter and gene, return masks that will remove overlapping regions'''    
    for gene, regions in overlaps.items():
        for region in regions:
            mask[:, region[0]:region[1],:] = False
    return mask

def calcPerSamplePi(sitePi, length=None):
    '''given numpy array of pi values, returns average per sample pi'''
    if length is None:
        length = sitePi.shape[1]
    return np.nansum(sitePi,axis=1)/length


def zero_out_region(regions, array):
    '''zero out regions of array'''
    for region in regions:
        array[:, region[0]:region[1]] = 0
    return array


def reshape_by_codon(numpyarray):
    '''ensures that arrays that are about to undergo
    codon-level manipulation are in the proper shape first'''
    if type(numpyarray) != np.ndarray:
        numpyarray = np.array(numpyarray)
    if len(numpyarray.shape) != 3:
        numpyarray = numpyarray.reshape(len(numpyarray),-1,3)
    assert numpyarray.shape[-1] == 3
    return numpyarray

def generateSynonFilters(codingRefSeqs):
    '''given numpy array of char sequences, return:
       - #samples by #nucs by 7 synon filter array
       - same thing for nonsynon filter
       - array of synon/nonsynon counts (2 by nucs by samples)'''
    #for all potential codons, find indexes of all instances of codon
    #and put relevant filter in numpy array at that index
    codingRefSeqs = reshape_by_codon(codingRefSeqs)

    nonSynonFilter = np.zeros((len(codingRefSeqs),len(codingRefSeqs[0]),3,7))
    synonFilter = np.zeros((len(codingRefSeqs),len(codingRefSeqs[0]),3,7))

    nonSynonSites = np.zeros((len(codingRefSeqs),len(codingRefSeqs[0]),3,4))
    synonSites = np.zeros((len(codingRefSeqs),len(codingRefSeqs[0]),3,4))

    for codon in translate.keys():
        ixs = np.asarray(np.all(codingRefSeqs==tuple(codon),axis=2)).nonzero()
        nonSynonFilter[ixs[0],ixs[1],:,:] = nonSynonPiTranslate[codon][np.newaxis,np.newaxis,:,:]
        synonFilter[ixs[0],ixs[1],:,:] = synonPiTranslate[codon][np.newaxis,np.newaxis,:,:]
        nonSynonSites[ixs[0],ixs[1],:,:] = nonSynonSiteCount[codon][np.newaxis,np.newaxis,:,:]
        synonSites[ixs[0],ixs[1],:,:] = synonSiteCount[codon][np.newaxis,np.newaxis,:,:]

    nonSynonFilter = nonSynonFilter.reshape(len(codingRefSeqs),-1,7)
    synonFilter = synonFilter.reshape(len(codingRefSeqs),-1,7)
    nonSynonSites = nonSynonSites.reshape(len(codingRefSeqs),-1,4)
    synonSites = synonSites.reshape(len(codingRefSeqs),-1,4)
    return synonFilter, nonSynonFilter, synonSites, nonSynonSites

def get_num_sites(allRefs, readCts):
    '''given synon or nonsynon filter, returns
    number of synon/nonsynon sites at each nucleotide for each sample
    (return format: np.ndarray, 1D, numofsamples, values=number of sites in sample)'''
    if type(allRefs) != np.ndarray:
        allRefs = np.array(allRefs)
    if len(allRefs.shape) != 3:
        allRefs = codingRefSeqs.reshape(len(codingRefSeqs),-1,3)
    assert codingRefSeqs.shape[-1] == 3
    sites = np.sum(npfilter[:,:,1:], axis=2)/3

    if overlap_to_remove:
        overlaps = calc_overlaps(codingCoords)
        genes = overlaps.keys()
        if gene not in genes:
            return np.sum(sites,axis=1)
        ranges = overlaps[gene]
        for start, end in ranges:
            for start, end in ranges:
                sites[:,start:end] = 0
    return np.sum(sites,axis=1)

def read_cts_into_SNP_freqs(readCts, seqArray):
    one_hot_ref = one_hot_encode(seqArray)
    SNP_freqs = np.where(readCts.sum(axis=2, keepdims=True)==0, one_hot_ref, readCts)
    SNP_freqs = SNP_freqs/SNP_freqs.sum(axis=2, keepdims=True)
    return SNP_freqs

def turnDFintoReadCts(df, totalConcatLength):
    nucDict={'A':0,'C':1,'G':2,'T':3}
    df['ref_nuc'] = df.ref_nuc.map(nucDict)
    df['alt_nuc'] = df.alt_nuc.map(nucDict)
    readCtsDF = pd.pivot_table(df[['sampleID','inConcatPos','ref_nuc','RD']], columns='ref_nuc', values = 'RD', index=('sampleID', 'inConcatPos'))
    readCtsDF.update(pd.pivot_table(df[['sampleID','inConcatPos','alt_nuc','AD']], columns='alt_nuc', values = 'AD', index=('sampleID', 'inConcatPos')))

    readCtsDF = readCtsDF.unstack().fillna(0)
    # raise Exception
    positions = readCtsDF[0].columns

    readCts = np.zeros((readCtsDF.shape[0],totalConcatLength,4))
    readCts[:,positions,0] = readCtsDF[0].to_numpy()
    readCts[:,positions,1] = readCtsDF[1].to_numpy()
    readCts[:,positions,2] = readCtsDF[2].to_numpy()
    readCts[:,positions,3] = readCtsDF[3].to_numpy()
    samplelist = list(readCtsDF.index.get_level_values(0))

    return readCts, samplelist, list(positions)

def performPiCalc(readCts):
    piCalcs = np.zeros(shape=(readCts.shape[0:2]+(7,)))
    piCalcs[:,:,0] = np.sum(readCts, axis=2)
    piCalcs[:,:,1] = readCts[:,:,0]*readCts[:,:,1]
    piCalcs[:,:,2] = readCts[:,:,0]*readCts[:,:,2]
    piCalcs[:,:,3] = readCts[:,:,0]*readCts[:,:,3]
    piCalcs[:,:,4] = readCts[:,:,1]*readCts[:,:,2]
    piCalcs[:,:,5] = readCts[:,:,1]*readCts[:,:,3]
    piCalcs[:,:,6] = readCts[:,:,2]*readCts[:,:,3]
    return piCalcs

def calcPerSitePi(piCalcs):
    return np.sum(piCalcs[:,:,1:],axis=2)/((piCalcs[:,:,0]**2-piCalcs[:,:,0])/2)

def combine_contigs(refseq):
    #Combine contig sequences into one string, annotate contig locations within concatenated genome
    concatrefseq = "".join([str(seq[1]) for seq in refseq])
    contigStarts = dict()
    contigCoords = dict()
    runningtally = 0
    for contig, seq in refseq:
        contigStarts[contig] = runningtally
        contigCoords[contig] = (runningtally, runningtally+len(seq))
        runningtally += len(seq)
    refseqArray = np.array(list(concatrefseq))
    return refseqArray, contigStarts, contigCoords

def masksFromCodingCoordinates(codingCoords, contigCoords, refSeqArray):
    codingMasks = dict()
    for contig, (startLoc, end) in contigCoords.items():
        for gene, coords in codingCoords[contig].items():
            codingMasks[gene] = np.full(refSeqArray.shape, False)
            for coord in coords:
                codingMasks[gene][coord[0]+startLoc:coord[1]+startLoc] = True
    return codingMasks


def parseGTF(gtffile):
    '''given file location of gtf, and dictionary of starting locations
       of each chrom in a concatenated sequence, return dictionary of
       {gene product : numpy filter for concatenated sequence'''
    #Note to self: Stupid GTFs are stupid 1-indexed with stupid inclusive ends
    with open(gtffile, 'r') as g:
        gtf = g.readlines()

    coding_regions = {} 
    for line in gtf:
        line = line.replace("/", "_")
        lineitems = line.split("\t")
        contig_name = lineitems[0]
        annotation_type = lineitems[2]
        start = int(lineitems[3]) - 1  # adding the -1 here to convert to 0 indexing
        stop = int(lineitems[4]) # not adding -1 because, while GTF is 1-indexed, its inclusive-ended. Converting to standard 0-indexing would mean 1-10 in GTF is equivelent to [0:10]
        gene_name = lineitems[8]
        gene_name = gene_name.split(";")[0]
        gene_name = gene_name.replace("gene_id ","")
        gene_name = gene_name.replace("\"","")

        if annotation_type.lower() == "cds":
            if contig_name not in coding_regions:
                coding_regions[contig_name] = {}
                coding_regions[contig_name][gene_name] = [[start, stop]]
            elif contig_name in coding_regions and gene_name not in coding_regions[contig_name]:
                coding_regions[contig_name][gene_name] = [[start, stop]]
            elif gene_name in coding_regions[contig_name]:
                coding_regions[contig_name][gene_name].append([start, stop])

    return coding_regions

def getRefSeqs(read_cts, refSeq):
    '''given dataframe of VCF with variable sites and numpy refseq array,
    returns numpy array of refseqs with sample key'''

    samples = list(vcfDF['sampleID'].unique())

    refSeqs = np.broadcast_to(refSeq[np.newaxis:], (len(samples),)+refSeq.shape).copy()

    for sample, df in vcfDF.groupby('sampleID'):
        sampleIndex = samples.index(sample)
        for mut, mutdf in df.groupby('pos'):
            if mutdf['refAlleleFreq'].tolist()[0] < 0.5 :
                samplerefnuc = mutdf['alt_nuc'].mode()[0]
                position = mutdf['inConcatPos'].tolist()[0]
                refSeqs[sampleIndex,position] = samplerefnuc
            else:
                continue
    return refSeqs, samples

# refseqArray, contigStarts, contigCoords, readCts, samplelist, poslist
from pm_io import safe_open, SimpleFastaParser
def read_fasta(path):
    with safe_open(path, 'r') as fasta:
        refseq = [(id.split(' ')[0], seq) for id, seq in SimpleFastaParser(fasta)]
    return refseq

def calc_small_genome_pi(readCts, sampleKey, poslist, ref_file, contigCoords, gtf):

    def calcPerSamplePi(sitePi, length=None):
        '''given numpy array of pi values, returns average per sample pi'''
        if length is None:
            length = sitePi.shape[1]
        return np.nansum(sitePi,axis=1)/length
    refseq = read_fasta(ref_file)
    refSeqArray, contigStarts, contigCoords = combine_contigs(refseq)
    refSeqArrays, sampleKey = getRefSeqs(readCts, refseqArray)
    piMath = performPiCalc(readCts)
    perSitePi = calcPerSitePi(piMath)
    perSitePi = np.nan_to_num(perSitePi)
    perSitePiDF = pd.DataFrame(perSitePi, index=sampleKey).dropna(how='all',axis=1)
    # perSitePiDF.columns = poslist
    perSitePiDF = perSitePiDF.stack().reset_index().rename(columns={'level_0':'sampleID','level_1':'inConcatPos',0:'pi'})

    perSamplePi = calcPerSamplePi(perSitePi)
    perSamplePiDF = pd.DataFrame(perSamplePi, index=sampleKey, columns=['pi_sample'])


    #contig Pi
    perContigPiDF = pd.DataFrame(index=sampleKey)

    for contig, coords in contigCoords.items():
        perContigPiDF[contig] = pd.Series(calcPerSamplePi(perSitePi[:,coords[0]:coords[1]], length = coords[1]-coords[0]), index=sampleKey)
    perContigPiDF = perContigPiDF.stack().reset_index().rename(columns={'level_0':'sampleID','level_1':'contig',0:'pi_contig'})

    #coding region Pi/PiN/PiS
    codingCoords = parseGTF(gtf)
    codingMasks = masksFromCodingCoordinates(codingCoords, contigCoords, refSeqArray)
    overlaps = calc_overlaps(codingCoords)

    genePis = pd.DataFrame(index=sampleKey)
    genePiN = pd.DataFrame(index=sampleKey)
    genePiS = pd.DataFrame(index=sampleKey)
    genePiN_sites = pd.DataFrame(index=sampleKey)
    genePiS_sites = pd.DataFrame(index=sampleKey)
    sample_piN_piS = pd.DataFrame(index=sampleKey, columns=['piN_sum','piN_sites','piS_sum','piS_sites']).fillna(0)


    for gene, mask in codingMasks.items():
        # print ((readCts*mask[np.newaxis, :, np.newaxis]).sum())
        geneSeqMaskedArray = np.ma.array(refseqArray, mask=np.broadcast_to(abs(mask-1)[np.newaxis, :, np.newaxis], refSeqArrays.shape))
        geneSeqArray = geneSeqMaskedArray.compressed().reshape(len(sampleKey),-1)

        synonFilter, nonSynonFilter, synonSiteFilter, nonSynonSiteFilter = generateSynonFilters(geneSeqArray)

        gene_readCts = np.ma.array(readCts, mask=np.broadcast_to(abs(mask-1)[np.newaxis,:,np.newaxis], readCts.shape))
        gene_readCts = gene_readCts.compressed().reshape(synonSiteFilter.shape)

        SNP_freqs = read_cts_into_SNP_freqs(gene_readCts, geneSeqArray)
        # remove start codon from math
        if SNPGenie_1_rules:
            nonSynonSiteFilter[:,0:3,:] = 1
            synonSiteFilter[:,0:3,:] = 0

        nonsynon_sites = (SNP_freqs*nonSynonSiteFilter).sum(axis=2)
        synon_sites = (SNP_freqs*synonSiteFilter).sum(axis=2)

        piMathMaskedArray = np.ma.array(piMath, mask=np.broadcast_to(abs(mask-1)[np.newaxis,:,np.newaxis], piMath.shape))
        genePiMathArray = piMathMaskedArray.compressed().reshape(synonFilter.shape)
        genePerSitePi = calcPerSitePi(genePiMathArray)
        nonSynonPerSitePi = calcPerSitePi(genePiMathArray*nonSynonFilter)
        synonPerSitePi = calcPerSitePi(genePiMathArray*synonFilter)
        genePis[gene] = calcPerSamplePi(genePerSitePi)
        genePiN[gene] = calcPerSamplePi(nonSynonPerSitePi, length=nonsynon_sites.sum(axis=1))
        genePiS[gene] = calcPerSamplePi(synonPerSitePi, length=synon_sites.sum(axis=1))
        genePiN_sites[gene] = nonsynon_sites.sum(axis=1)
        genePiS_sites[gene] = synon_sites.sum(axis=1)

        #And now do the same thing w/o overlapping regions to accurately determine whole-sample piN/piS
        if gene in overlaps.keys(): # Still best done w/ new metho
            synonFilter_no_overlap = remove_overlap_regions(overlaps, synonFilter)
            nonSynonFilter_no_overlap = remove_overlap_regions(overlaps, nonSynonFilter)
            contig = [contig for contig in codingCoords.keys() if gene in codingCoords[contig]][0]
            keepers = get_nonoverlapping_in_gene_locations(contig, gene, codingCoords)

            sample_piN_piS['piS_sites'] += synon_sites[:, keepers].sum(axis=1)
            sample_piN_piS['piN_sites'] += nonsynon_sites[:, keepers].sum(axis=1)
            nonSynonPerSitePi_no_overlap = calcPerSitePi(genePiMathArray*nonSynonFilter_no_overlap)
            synonPerSitePi_no_overlap = calcPerSitePi(genePiMathArray*synonFilter_no_overlap)

            sample_piN_piS['piS_sum'] += calcPerSamplePi(synonPerSitePi_no_overlap, length=1)
            sample_piN_piS['piN_sum'] += calcPerSamplePi(nonSynonPerSitePi_no_overlap, length=1)
        else:
            if gene not in ['PA-X', 'PB1-F2', 'HA_antigenic', 'HA_nonantigenic']:
                sample_piN_piS['piS_sites'] += synon_sites.sum(axis=1)
                sample_piN_piS['piN_sites'] += nonsynon_sites.sum(axis=1)
                sample_piN_piS['piS_sum'] += calcPerSamplePi(synonPerSitePi, length=1)
                sample_piN_piS['piN_sum'] += calcPerSamplePi(nonSynonPerSitePi, length=1)


    sample_piN_piS['piN_sample'] = sample_piN_piS['piN_sum']/sample_piN_piS['piN_sites']
    sample_piN_piS['piS_sample'] = sample_piN_piS['piS_sum']/sample_piN_piS['piS_sites']
    perSamplePiDF = perSamplePiDF.join(sample_piN_piS)

    genePis = genePis.stack().reset_index().rename(columns={'level_0':'sampleID','level_1':'product',0:'pi'})
    genePiN = genePiN.stack().reset_index().rename(columns={'level_0':'sampleID','level_1':'product',0:'pi'})
    genePiS = genePiS.stack().reset_index().rename(columns={'level_0':'sampleID','level_1':'product',0:'pi'})
    genePiN_sites = genePiN_sites.stack().reset_index().rename(columns={'level_0':'sampleID','level_1':'product',0:'sites'})
    genePiS_sites = genePiS_sites.stack().reset_index().rename(columns={'level_0':'sampleID','level_1':'product',0:'sites'})
    genePiN = genePiN.merge(genePiN_sites, on=['sampleID','product'], how='left')
    genePiS = genePiS.merge(genePiS_sites, on=['sampleID','product'], how='left')

    genePis['type'] = 'pi'
    genePiN['type'] = 'piN'
    genePiS['type'] = 'piS'

    genePis = genePis.append(genePiN).append(genePiS)

    # Finally, just because somehow it hasn't happened yet in this code,
    # I'm going to make sure the contig is attached to the gene DF

    contigDict = {gene:contig.split('_')[-1] for contig in codingCoords.keys() for gene in codingCoords[contig]}
    genePis['contig'] = genePis['product'].map(contigDict)

    return perSamplePiDF, perContigPiDF, genePis, perSitePiDF

def calcPi(vcfDF, gtf=None, length = None, refseq = None, maf = 0.01, rollingWindow=None, synon_nonsynon=False): 
    # Main body of program
    # refseq = list(SeqIO.parse(refseq, 'fasta'))
    # refseqArray, contigStarts, contigCoords = combine_contigs(refseq)

    vcfDF = vcfDF.rename(columns={'sample':'sampleID'})
    #Now that we have our concatenated sequence with dictionary of start of each contig,
    #we can assemble a numpy representation of read counts aligned with our references
    # piSNPs = vcfDF[['sampleID','product','contig','inGenePos','ref_nuc','alt_nuc','RD','AD','AAtype','pos']]
    # piSNPs = piSNPs.sort_values(['sampleID','product','inGenePos'])

    # piSNPs['pos'] = piSNPs.pos-1 #Apparantly this is taken straight from vcf, which is 1-indexed

    # piSNPs = piSNPs.rename(columns={'pos':'inContigPos'})
    # piSNPs['refAlleleFreq'] = piSNPs.RD/(piSNPs.RD+piSNPs.AD)

    # piSNPs = piSNPs.loc[(piSNPs.refAlleleFreq >= maf)&(piSNPs.refAlleleFreq <= (1-maf))]
    # Adjust SNPs to remove reads that contribute to minor variants below maf cutoff:
    # piSNPs.loc[piSNPs.refAlleleFreq < maf, 'RD'] = 0
    # piSNPs.loc[piSNPs.refAlleleFreq > (1-maf), 'AD'] = 0
    # piSNPs['pos'] = piSNPs['product'] + ' '+ piSNPs['inGenePos'].astype(str)
    piSNPs[['sampleID','contig','pos','ref_nuc','alt_nuc','RD','AD','AAtype','inContigPos','refAlleleFreq']].set_index(['sampleID','pos'])
    piSNPs['inConcatPos'] = piSNPs.inContigPos

    for contig, offset in contigStarts.items():
        contig = contig.split('_')[-1]
        piSNPs.loc[piSNPs.contig == contig,'inConcatPos'] += offset

    # equivelent to consensus_arrays and samle_list
    refSeqArrays, sampleKey = getRefSeqs(piSNPs, refseqArray)
    # equivelent to read_cts, sample_list, var_pos
    readCts, samplelist, poslist = turnDFintoReadCts(piSNPs, len(refseqArray))

    piMath = performPiCalc(readCts)

    perSitePi = calcPerSitePi(piMath)

    def calcPerSamplePi(sitePi, length=None):
        '''given numpy array of pi values, returns average per sample pi'''
        if length is None:
            length = sitePi.shape[1]
        return np.nansum(sitePi,axis=1)/length

    perSitePi = np.nan_to_num(perSitePi)
    perSitePiDF = pd.DataFrame(perSitePi, index=sampleKey).dropna(how='all',axis=1)
    # perSitePiDF.columns = poslist
    perSitePiDF = perSitePiDF.stack().reset_index().rename(columns={'level_0':'sampleID','level_1':'inConcatPos',0:'pi'})
    perSitePiDF = piSNPs.merge(perSitePiDF, on=['sampleID','inConcatPos'])

    perSamplePi = calcPerSamplePi(perSitePi)
    perSamplePiDF = pd.DataFrame(perSamplePi, index=sampleKey, columns=['pi_sample'])


    #contig Pi
    perContigPiDF = pd.DataFrame(index=sampleKey)

    for contig, coords in contigCoords.items():
        perContigPiDF[contig] = pd.Series(calcPerSamplePi(perSitePi[:,coords[0]:coords[1]], length = coords[1]-coords[0]), index=sampleKey)
    perContigPiDF = perContigPiDF.stack().reset_index().rename(columns={'level_0':'sampleID','level_1':'contig',0:'pi_contig'})

    #coding region Pi/PiN/PiS
    codingCoords = parseGTF(gtf, contigStarts)
    codingMasks = masksFromCodingCoordinates(codingCoords, contigStarts, refSeqArrays)
    overlaps = calc_overlaps(codingCoords)

    genePis = pd.DataFrame(index=sampleKey)
    genePiN = pd.DataFrame(index=sampleKey)
    genePiS = pd.DataFrame(index=sampleKey)
    genePiN_sites = pd.DataFrame(index=sampleKey)
    genePiS_sites = pd.DataFrame(index=sampleKey)
    sample_piN_piS = pd.DataFrame(index=sampleKey, columns=['piN_sum','piN_sites','piS_sum','piS_sites']).fillna(0)


    for gene, mask in codingMasks.items():
        # print ((readCts*mask[np.newaxis, :, np.newaxis]).sum())
        geneSeqMaskedArray = np.ma.array(refSeqArrays, mask=np.broadcast_to(abs(mask-1), refSeqArrays.shape))
        geneSeqArray = geneSeqMaskedArray.compressed().reshape(len(samplelist),-1)

        synonFilter, nonSynonFilter, synonSiteFilter, nonSynonSiteFilter = generateSynonFilters(geneSeqArray)

        gene_readCts = np.ma.array(readCts, mask=np.broadcast_to(abs(mask-1)[np.newaxis,:,np.newaxis], readCts.shape))
        gene_readCts = gene_readCts.compressed().reshape(synonSiteFilter.shape)

        SNP_freqs = read_cts_into_SNP_freqs(gene_readCts, geneSeqArray)
        # remove start codon from math
        if SNPGenie_1_rules:
            nonSynonSiteFilter[:,0:3,:] = 1
            synonSiteFilter[:,0:3,:] = 0

        nonsynon_sites = (SNP_freqs*nonSynonSiteFilter).sum(axis=2)
        synon_sites = (SNP_freqs*synonSiteFilter).sum(axis=2)

        piMathMaskedArray = np.ma.array(piMath, mask=np.broadcast_to(abs(mask-1)[np.newaxis,:,np.newaxis], piMath.shape))
        genePiMathArray = piMathMaskedArray.compressed().reshape(synonFilter.shape)
        genePerSitePi = calcPerSitePi(genePiMathArray)
        nonSynonPerSitePi = calcPerSitePi(genePiMathArray*nonSynonFilter)
        synonPerSitePi = calcPerSitePi(genePiMathArray*synonFilter)
        genePis[gene] = calcPerSamplePi(genePerSitePi)
        genePiN[gene] = calcPerSamplePi(nonSynonPerSitePi, length=nonsynon_sites.sum(axis=1))
        genePiS[gene] = calcPerSamplePi(synonPerSitePi, length=synon_sites.sum(axis=1))
        genePiN_sites[gene] = nonsynon_sites.sum(axis=1)
        genePiS_sites[gene] = synon_sites.sum(axis=1)

        #And now do the same thing w/o overlapping regions to accurately determine whole-sample piN/piS
        if gene in overlaps.keys(): # Still best done w/ new metho
            synonFilter_no_overlap = remove_overlap_regions(overlaps, synonFilter)
            nonSynonFilter_no_overlap = remove_overlap_regions(overlaps, nonSynonFilter)
            contig = [contig for contig in codingCoords.keys() if gene in codingCoords[contig]][0]
            keepers = get_nonoverlapping_in_gene_locations(contig, gene, codingCoords)

            sample_piN_piS['piS_sites'] += synon_sites[:, keepers].sum(axis=1)
            sample_piN_piS['piN_sites'] += nonsynon_sites[:, keepers].sum(axis=1)
            nonSynonPerSitePi_no_overlap = calcPerSitePi(genePiMathArray*nonSynonFilter_no_overlap)
            synonPerSitePi_no_overlap = calcPerSitePi(genePiMathArray*synonFilter_no_overlap)

            sample_piN_piS['piS_sum'] += calcPerSamplePi(synonPerSitePi_no_overlap, length=1)
            sample_piN_piS['piN_sum'] += calcPerSamplePi(nonSynonPerSitePi_no_overlap, length=1)
        else:
            if gene not in ['PA-X', 'PB1-F2', 'HA_antigenic', 'HA_nonantigenic']:
                sample_piN_piS['piS_sites'] += synon_sites.sum(axis=1)
                sample_piN_piS['piN_sites'] += nonsynon_sites.sum(axis=1)
                sample_piN_piS['piS_sum'] += calcPerSamplePi(synonPerSitePi, length=1)
                sample_piN_piS['piN_sum'] += calcPerSamplePi(nonSynonPerSitePi, length=1)


    sample_piN_piS['piN_sample'] = sample_piN_piS['piN_sum']/sample_piN_piS['piN_sites']
    sample_piN_piS['piS_sample'] = sample_piN_piS['piS_sum']/sample_piN_piS['piS_sites']
    perSamplePiDF = perSamplePiDF.join(sample_piN_piS)

    genePis = genePis.stack().reset_index().rename(columns={'level_0':'sampleID','level_1':'product',0:'pi'})
    genePiN = genePiN.stack().reset_index().rename(columns={'level_0':'sampleID','level_1':'product',0:'pi'})
    genePiS = genePiS.stack().reset_index().rename(columns={'level_0':'sampleID','level_1':'product',0:'pi'})
    genePiN_sites = genePiN_sites.stack().reset_index().rename(columns={'level_0':'sampleID','level_1':'product',0:'sites'})
    genePiS_sites = genePiS_sites.stack().reset_index().rename(columns={'level_0':'sampleID','level_1':'product',0:'sites'})
    genePiN = genePiN.merge(genePiN_sites, on=['sampleID','product'], how='left')
    genePiS = genePiS.merge(genePiS_sites, on=['sampleID','product'], how='left')

    genePis['type'] = 'pi'
    genePiN['type'] = 'piN'
    genePiS['type'] = 'piS'

    genePis = genePis.append(genePiN).append(genePiS)

    # Finally, just because somehow it hasn't happened yet in this code,
    # I'm going to make sure the contig is attached to the gene DF

    contigDict = {gene:contig.split('_')[-1] for contig in codingCoords.keys() for gene in codingCoords[contig]}
    genePis['contig'] = genePis['product'].map(contigDict)

    return perSamplePiDF, perContigPiDF, genePis, perSitePiDF
