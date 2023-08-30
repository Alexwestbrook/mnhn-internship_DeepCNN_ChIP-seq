#!/bin/bash

# usage:
#   alignment.sh -x <path_to_index_prefix> -d <data_directory> -1 <fastq_file1> -2 <fastq_file2> -o <output_file_prefix> [options]
# options
#   -w <writing directory>   writing directory, if data_directory cannot be written in
#   -b  indicates paired-end reads, fastq files must be fastq_prefix.R1.fastq and fastq_prefix.R2.fastq
#   -p <threads>  number of threads to use to speed up computation

paired_end=false
threads=1
while getopts "x:d:w:1:2:o:bp:" option; do
    case $option in
        x) # index of reference genome
            index=$OPTARG;;
        d) # directory containing the sequences
            data_dir=$OPTARG
            writing_dir=$data_dir;;
        w) # directory in which to write files,
           # otherwise in data_dir
            writing_dir=$OPTARG;;
        1) # fastq file containing reads to align
            fastq_file1=$OPTARG;;
        2) # fastq file of paired reads containing reads to align
            fastq_file2=$OPTARG;;
        o) # prefix of output_file
            out_prefix=$OPTARG;;
        b) # indicate paired-end
            paired_end=true;;
        p) # number of threads to use
            threads=$OPTARG;;
        \?) # Invalid option
            echo "Error: Invalid option"
            exit;;
    esac
done

# aligning sequences to genome

if [ $paired_end = true ]
then
    out_prefix=$writing_dir/$out_prefix'_paired'
    bowtie2 -p $threads -x $index -1 $data_dir/$fastq_file1 -2 $data_dir/$fastq_file2 -S $out_prefix.sam
else
    out_prefix=$writing_dir/$out_prefix
    bowtie2 -p $threads -x $index -U $data_dir/$fastq_file1 -S $out_prefix.sam
fi
samtools view -bS $out_prefix.sam | samtools sort -o $out_prefix.sorted.bam
rm $out_prefix.sam
samtools index $out_prefix.sorted.bam