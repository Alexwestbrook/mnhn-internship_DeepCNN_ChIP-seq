#!/bin/bash
threads=18
frac=0.1
input_bam='/home/alex/shared_folder/Judith-H3K9me3/results/alignments/T2T-CHM13v2.0/HN00205099_RawFASTQ_RPE1_WTH3K9me3_paired.sorted.bam'

source activate align
samtools collate -@ $threads $input_bam -o shuffled.bam
samtools view -@ $threads --subsample $frac shuffled.bam -bo subsample.bam
conda deactivate

source activate picard
picard SplitSamByNumberOfReads I=shuffled.bam OUTPUT=. SPLIT_TO_N_FILES=2 OUT_PREFIX=split
conda deactivate