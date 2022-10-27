#!/bin/bash

# usage:
#   shuffle_fastq.sh -o output_fastq_file -f input_fastq_file

while getopts "o:f:" option; do
    case $option in
        o) # output file prefix
            final_name=$OPTARG;;
        f) # fastq files to sample from
            fastq_file=$OPTARG;;
        \?) # Invalid option
            echo "Error: Invalid option, shuffle_fastq.sh -o [OUTPUT] -f [FASTQ_FILE]"
            exit;;
    esac
done
# from https://www.biostars.org/p/9764/
awk '{OFS="\t"; getline seq; getline sep; getline qual; print $0,seq,sep,qual}' $fastq_file | shuf | awk -F '\t' '{OFS="\n"; print $1,$2,$3,$4}' > $final_name

