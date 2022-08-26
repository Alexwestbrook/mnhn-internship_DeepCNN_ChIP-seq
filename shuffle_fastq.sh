#!/bin/bash

remove_Ns=false
while getopts "o:f:r" option; do
    case $option in
        o) # output file prefix
            final_name=$OPTARG;;
        f) # fastq files to sample from
            fastq_file=$OPTARG;;
        r) # indicate to remove Ns
            remove_Ns=true;;
        \?) # Invalid option
            echo "Error: Invalid option, shuffle_fastq.sh -o [OUTPUT] -f [FASTQ_FILE] [OPTIONS]"
            echo "-r : remove reads with Ns"
            exit;;
    esac
done
# from https://www.biostars.org/p/9764/
if [ $remove_Ns = true ]
then
    echo 'remove sequences with Ns'
    awk '{OFS="\t"; getline seq; getline sep; getline qual; \
                    if (index(seq, "N")==0) {print $0,seq,sep,qual}}' $fastq_file > $final_name'_tmp'
else
    awk '{OFS="\t"; getline seq; getline sep; getline qual; \
                    print $0,seq,sep,qual}' $fastq_file > $final_name'_tmp'
fi
echo 'shuffle fastq'
shuf $final_name'_tmp' | awk -F '\t' '{OFS="\n"; print $1,$2,$3,$4}' > $final_name
rm $final_name'_tmp'
