#!/bin/bash

# usage:
#   alignment.sh -i path_to_index_prefix -d data_directory -f fastq_prefix [options]
# options
#   -w  writing directory, if data_directory cannot be written in
#   -p  indicates paired-end reads, fastq files must be fastq_prefix_1.fastq and fastq_prefix_2.fastq
#   -t  number of threads to use to speed up computation

paired_end=false
threads=1
while getopts "i:d:w:f:pt:" option; do
    case $option in
        i) # index of reference genome
            index=$OPTARG;;
        d) # directory containing the sequences
            data_dir=$OPTARG
            writing_dir=$data_dir;;
        w) # directory in which to write files,
           # if no writing access in data_dir
            writing_dir=$OPTARG;;
        f) # prefix of fastq files containing reads to align
            fastq_prefix=$OPTARG;;
        p) # indicate paired-end
            paired_end=true;;
        t) # number of threads to use
            threads=$OPTARG;;
        \?) # Invalid option
            echo "Error: Invalid option"
            exit;;
    esac
done

# aligning sequences to human genome

if [ $paired_end = true ]
then
    out_prefix=$writing_dir/$fastq_prefix'_paired'
    bowtie2 -p $threads -x $index -1 $data_dir/$fastq_prefix'.R1.fastq' -2 $data_dir/$fastq_prefix'.R2.fastq' -S $out_prefix.sam
else
    out_prefix=$writing_dir/$fastq_prefix
    bowtie2 -p $threads -x $index -U $data_dir/$fastq_prefix.fastq -S $out_prefix.sam
fi
samtools view -bS $out_prefix.sam > $out_prefix.bam
samtools sort $out_prefix.bam -o $out_prefix.sorted.bam
samtools index $out_prefix.sorted.bam


