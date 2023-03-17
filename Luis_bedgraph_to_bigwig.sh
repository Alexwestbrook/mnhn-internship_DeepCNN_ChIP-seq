#!/bin/bash

bedgraph_dir='../shared_folder/MMusculus/data/MNase/multimappers/A'
genome_dir='../shared_folder/MMusculus/genome'

# subsample Luis' bedgraphs and sort them
awk '/^chr[0-9]+/ {print}' MNase_A_16_1.bedgraph | sort -k1,1 -k2,2n > MNase_A_16_1_sorted.bedgraph
awk '/^chr[0-9]+/ {print}' MNase_A_16_2.bedgraph | sort -k1,1 -k2,2n > MNase_A_16_2_sorted.bedgraph
awk '/^chr[0-9]+/ {print}' MNase_A_16_3.bedgraph | sort -k1,1 -k2,2n > MNase_A_16_3_sorted.bedgraph
# convert to bigwig
bedGraphToBigWig $bedgraph_dir/MNase_A_16_1_sorted.bedgraph $genome_dir/mm10.chrom.sizes $bedgraph_dir/MNase_A_16_1.bw
bedGraphToBigWig $bedgraph_dir/MNase_A_16_2_sorted.bedgraph $genome_dir/mm10.chrom.sizes $bedgraph_dir/MNase_A_16_2.bw
bedGraphToBigWig $bedgraph_dir/MNase_A_16_3_sorted.bedgraph $genome_dir/mm10.chrom.sizes $bedgraph_dir/MNase_A_16_3.bw