#!/bin/bash
dir='../shared_folder/Judith-H3K4me3/raw_data/Control'
root_name=$dir/'D1145C42_trimmed'
fq_file=$root_name'.R1.fastq'
id_file=$root_name'.ids'

# get ids in fastq
awk 'NR%4 == 1 {print $0}' $fq_file > $id_file

# get number of ids
n=`wc -l < $id_file`
# subsample ids
for d in 2 4 8
do
    shuf $id_file | awk -v num=$(expr $n / $d) 'NR <= num {print $0}' > $root_name'_divby'$d'.ids'
done
rm $id_file
