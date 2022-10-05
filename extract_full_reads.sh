#!/bin/bash

# This script removes reads too small or containing Ns

# usage:
#   subsample_fastq.sh -l read_length -o output_file_prefix -f "file1.fastq file2.fastq ..." [options]
# options:
#   -p  indicates that reads are paired-end, requires number of files to be exactly 2
# (Paired-end assumes there is one space in id line, and no spaces elsewhere. It is untested)

paired_end=false
while getopts "l:o:f:p" option; do
    case $option in
        l) # length of reads to extract
            length=$OPTARG;;
        o) # output file prefix
            final_name=$OPTARG;;
        f) # fastq files to sample from
            fastq_files=$OPTARG;;
        p) # indicate paired-end
            paired_end=true;;
        \?) # Invalid option
            echo "Error: Invalid option"
            exit;;
    esac
done
if [ $paired_end = true ]
then
    echo 'processing as paired-end...'
    fastq_1=${fastq_files[0]}
    fastq_2=${fastq_files[1]}
    # linearize both fastq files
    awk 'NR%4==1 {printf("%s%s\t",(N>0?"\n":""),$0);N++;next;} {printf("%s\t",$0);} END {printf("\n")}' $fastq_1 > $fastq_1'_tmp'
    awk 'NR%4==1 {printf("%s%s\t",(N>0?"\n":""),$0);N++;next;} {printf("%s\t",$0);} END {printf("\n")}' $fastq_2 > $fastq_2'_tmp'
    # extract ids of reads matching subsampled sequence from at least one end
    awk -v l=$length -F ' |\t' '(length($3)==l && index($3, "N")==0) {if (FNR==NR) {ids1[$1];next;} if ($1 in ids1) {ids2[$1]}} END {for (id in ids) {print id}}' $fastq_1'_tmp' $fastq_2'_tmp' > $sequences'_ids'
    # extract fastq lines with corresponding ids in each file
    awk -F ' |\t' 'FNR==NR{ids[$1];next;} ($1 in ids) {printf("%s %s\n%s\n%s\n%s\n",$1,$2,$3,$4,$5); }' $sequences'_ids' $fastq_1'_tmp' > $final_name'_1.fastq'
    awk -F ' |\t' 'FNR==NR{ids[$1];next;} ($1 in ids) {printf("%s %s\n%s\n%s\n%s\n",$1,$2,$3,$4,$5); }' $sequences'_ids' $fastq_2'_tmp' > $final_name'_2.fastq'
    rm $sequences'_ids'
    rm $fastq_1'_tmp'
    rm $fastq_2'_tmp'
else
    echo 'processing as single-end...'
    # merge and linearize fastq files
    awk 'NR%4==1 {printf("%s%s\t",(N>0?"\n":""),$0);N++;next;} {printf("%s\t",$0);} END {printf("\n")}' $fastq_files > $final_name'_tmp'
    # extract fastq lines with sequence of correct length
    awk -v l=$length -F '\t' '(length($2)==l && index($2, "N")==0) {printf("%s\n%s\n%s\n%s\n",$1,$2,$3,$4); }' $final_name'_tmp' > $final_name.fastq
    rm $final_name'_tmp'
fi
echo 'done'
