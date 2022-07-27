#!/bin/bash

data_dir='../shared_folder'
writing_dir='../shared_folder'
# aligning sequences to human genome
data='H3K27ac'
model_name='model_inception2.2'
sequences_file='seqs_H3K27ac_2.2_0.84-_all'
bowtie2 -p 20 -x $data_dir/Human/assembly/GRCh38_index -U $data_dir/$data/results/$model_name/$sequences_file.fastq -S $writing_dir/$data/results/$model_name/$sequences_file.sam
samtools view -bS $writing_dir/$data/results/$model_name/$sequences_file.sam > $writing_dir/$data/results/$model_name/$sequences_file.bam
samtools sort $writing_dir/$data/results/$model_name/$sequences_file.bam -o $writing_dir/$data/results/$model_name/$sequences_file.sorted.bam
samtools index $writing_dir/$data/results/$model_name/$sequences_file.sorted.bam


