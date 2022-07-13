#!/bin/bash


data='H3K27ac'
label1='IP'
label2='Control'
access1='ENCFF098LPP'
access2='ENCFF226MBW'
# access3='ENCFF494PEG'
# access4='ENCFF923IOZ'
paired_end=true

in1=$data/raw_data/$label1/$access1.fastq
in2=$data/raw_data/$label1/$access2.fastq
out1=$data/raw_data/$label1/$access1'_deduped'.fastq
out2=$data/raw_data/$label1/$access2'_deduped'.fastq

# in3=$data/raw_data/$label2/$access3.fastq
# in4=$data/raw_data/$label2/$access4.fastq
# out3=$data/raw_data/$label2/$access3'_deduped'.fastq
# out4=$data/raw_data/$label2/$access4'_deduped'.fastq

if [ $paired_end = true ]
then
    bash bbmap/clumpify.sh in=$in1 in2=$in2 out=$out1 out2=$out2 -subs=0 dedupe
    # bash bbmap/clumpify.sh in=$in3 in2=$in4 out=$out3 out2=$out4 -subs=0 dedupe
else
    bash bbmap/clumpify.sh in=$in1 out=$out1 -subs=0 dedupe
    bash bbmap/clumpify.sh in=$in2 out=$out2 -subs=0 dedupe
    bash bbmap/clumpify.sh in=$in3 out=$out3 -subs=0 dedupe
    bash bbmap/clumpify.sh in=$in4 out=$out4 -subs=0 dedupe
fi

# from https://www.biostars.org/p/9764/

# out1=$data/raw_data/$label/$access1'_deduped_shuf'.fastq
# out2=$data/raw_data/$label/$access2'_deduped_shuf'.fastq
# # paired end reads, doesn't work though
# paste <(zcat $mid1) <(zcat $mid2) | paste - - - - | shuf | awk -F'\t' '{OFS="\n"; print $1,$3,$5,$7 > $out1; print $2,$4,$6,$8 > $out2}'

# # single end reads
# awk '{OFS="\t"; getline seq; \
#                 getline sep; \
#                 getline qual; \
#                 print $0,seq,sep,qual}' $mid1 | \
# shuf | \
# awk '{OFS="\n"; print $1,$2,$3,$4}' \
# > $out1
