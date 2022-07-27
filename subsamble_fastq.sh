#!/bin/bash

sequences=$1
final_fastq=$2
shift
shift
fastq_files=$@
# extract only sequence from fasta
awk '/^>/ {printf("%s",N>0?"\n":"");N++;next;}{printf("%s",$0);}' $sequences > $sequences'_tmp'
# linearize fastq file
awk 'NR%4==1 {printf("%s%s\t",(N>0?"\n":""),$0);N++;next;} {printf("%s\t",$0);} END {printf("\n")}' $fastq_files > $final_fastq'_tmp'
# extract fastq lines corresponding to subsampled sequences
awk -F '\t' 'FNR==NR{seqs[$1];next;} ($2 in seqs) {printf("%s\n%s\n%s\n%s\n",$1,$2,$3,$4); }' $sequences'_tmp' $final_fastq'_tmp' > $final_fastq
rm $sequences'_tmp'
rm $final_fastq'_tmp'
