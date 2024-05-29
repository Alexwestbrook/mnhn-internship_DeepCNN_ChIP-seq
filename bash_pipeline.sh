#!/bin/bash

scripts_dir='/home/alex/mnhn-internship_DeepCNN_ChIP-seq'
data_dir='/home/alex/shared_folder'
writing_dir=$data_dir

# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/test49 --start_seqs $data_dir/SCerevisiae/generated/test9/start_seqs_first10.npy --steps 500 -temp 0.0005 --stride 128 --seed 0 -w 1 1 1 0.01
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/test50 --start_seqs $data_dir/SCerevisiae/generated/test9/start_seqs_first10.npy --steps 500 -temp 0.0005 --stride 128 --seed 0 -w 1 1 1 0.005
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/test51 --start_seqs $data_dir/SCerevisiae/generated/test9/start_seqs_first10.npy --steps 500 -temp 0.0005 --stride 128 --seed 0 -w 1 1 1 0.002
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/test52 --start_seqs $data_dir/SCerevisiae/generated/test9/start_seqs_first10.npy --steps 500 -temp 0.0005 --stride 128 --seed 0 -w 1 1 1 0.001
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/test53 --start_seqs $data_dir/SCerevisiae/generated/test9/start_seqs_first10.npy --steps 500 -temp 0.0005 --stride 128 --seed 0 -w 1 1 1 0.0005
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/test54 --start_seqs $data_dir/SCerevisiae/generated/test9/start_seqs_first10.npy --steps 500 -temp 0.0005 --stride 128 --seed 0 -w 1 1 1 0.0001
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/test55 --start_seqs $data_dir/SCerevisiae/generated/test9/start_seqs_first10.npy --steps 500 -temp 0.0005 --stride 128 --seed 0 -w 1 1 1 0.00005
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/test56 --start_seqs $data_dir/SCerevisiae/generated/test9/start_seqs_first10.npy --steps 500 -temp 0.0005 --stride 128 --seed 0 -w 1 1 1 0.00001

# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/test73 -m $data_dir/JB_seqdes/weight_CNN_RNA_seq_2001_12_8_4_SRR7131299.hdf5 -ord ATGC -nseq 2 -l 5000 --steps 50 -temp 0.1 --stride 100 --seed 0 -b 1024
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/test74 -m $data_dir/JB_seqdes/weight_CNN_RNA_seq_2001_12_8_4_SRR7131299.hdf5 -ord ATGC -nseq 2 -l 5000 --steps 50 -temp 0.01 --stride 100 --seed 0 -b 1024
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/test75 -m $data_dir/JB_seqdes/weight_CNN_RNA_seq_2001_12_8_4_SRR7131299.hdf5 -ord ATGC -nseq 2 -l 5000 --steps 50 -temp 0.001 --stride 100 --seed 0 -b 1024
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/test76 -m $data_dir/JB_seqdes/weight_CNN_RNA_seq_2001_12_8_4_SRR7131299.hdf5 -ord ATGC -nseq 2 -l 5000 --steps 50 -temp 0.0001 --stride 100 --seed 0 -b 1024
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/test77 -m $data_dir/JB_seqdes/weight_CNN_RNA_seq_2001_12_8_4_SRR7131299.hdf5 -ord ATGC -nseq 2 -l 5000 --steps 50 -temp 0.00001 --stride 100 --seed 0 -b 1024

# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/lowpol_4kb_temp2e-4 -n 2 -l 4000 --steps 500 -t 0.0002 -s 128 --seed 1 -v
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/lowpol_4kb_temp1e-4 -n 2 -l 4000 --steps 500 -t 0.0001 -s 128 --seed 1 -v
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/lowpol_4kb_temp5e-5 -n 2 -l 4000 --steps 500 -t 0.00005 -s 128 --seed 1 -v
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/lowpol_4kb_temp2e-4_mid -n 2 -l 4000 --steps 500 -t 0.0002 -s 128 -mid --seed 1 -v
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/lowpol_4kb_temp1e-4_mid -n 2 -l 4000 --steps 500 -t 0.0001 -s 128 -mid --seed 1 -v
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/lowpol_4kb_temp5e-5_mid -n 2 -l 4000 --steps 500 -t 0.00005 -s 128 -mid --seed 1 -v
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/highpol_4kb_temp2e-4 -n 2 -l 4000 --steps 500 -t 0.0002 -s 128 -targ 1 --seed 1 -v
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/highpol_4kb_temp1e-4 -n 2 -l 4000 --steps 500 -t 0.0001 -s 128 -targ 1 --seed 1 -v
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/highpol_4kb_temp5e-5 -n 2 -l 4000 --steps 500 -t 0.00005 -s 128 -targ 1 --seed 1 -v
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/highpol_4kb_temp2e-4_mid -n 2 -l 4000 --steps 500 -t 0.0002 -s 128 -mid -targ 1 --seed 1 -v
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/highpol_4kb_temp1e-4_mid -n 2 -l 4000 --steps 500 -t 0.0001 -s 128 -mid -targ 1 --seed 1 -v
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/highpol_4kb_temp5e-5_mid -n 2 -l 4000 --steps 500 -t 0.00005 -s 128 -mid -targ 1 --seed 1 -v

# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/test89 -n 2 -l 4000 --steps 500 -t 0.0001 -s 128 --seed 1 -v --flanks $data_dir/SCerevisiae/genome/W303_Mmmyco_random1kbflanks_ACGTidx.npz
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/test90 -n 2 -l 4000 --steps 500 -t 0.0001 -s 128 -mid --seed 1 -v --flanks $data_dir/SCerevisiae/genome/W303_Mmmyco_random1kbflanks_ACGTidx.npz

# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/test91 -m $data_dir/JB_seqdes/weight_CNN_RNA_seq_2001_12_8_4_SRR7131299.hdf5 -ord ATGC -n 2 -l 4000 --steps 50 -t 0.0001 -s 100 --seed 1 -v
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/test92 -m $data_dir/JB_seqdes/weight_CNN_RNA_seq_2001_12_8_4_SRR7131299.hdf5 -ord ATGC -n 2 -l 4000 --steps 500 -t 0.0001 -s 100 --seed 1 -v --flanks random
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/test93 -n 1 -l 4000 --steps 500 -t 0.0001 -s 128 --seed 1 -v --flanks random
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/test94 -n 2 -l 4000 --steps 500 -t 0.0001 -s 128 --seed 1 -v --target_file $data_dir/SCerevisiae/generated/targets/4kb_midpeak1kb.npz
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/test95 -m $data_dir/SCerevisiae/Trainedmodels/model_myco_nuc_2/model -n 2 -l 4000 --steps 500 -t 0.0001 -s 100 --seed 1 -v --target_file $data_dir/SCerevisiae/generated/targets/4kb_sharpnuc147_linker20.npz
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/test96 -m $data_dir/SCerevisiae/Trainedmodels/model_myco_nuc_2/model -n 2 -l 4000 --steps 10 -t 0.0001 -s 100 --seed 1 -v --target_file $data_dir/SCerevisiae/generated/targets/4kb_sharpnuc147_linker20.npz
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/test97 -m $data_dir/SCerevisiae/Trainedmodels/model_myco_nuc_2/model -n 2 -l 4000 --steps 10 -t 0.0001 -s 16 --seed 1 -v --target_file $data_dir/SCerevisiae/generated/targets/4kb_sharpnuc147_linker20.npz
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/test98 -m $data_dir/SCerevisiae/Trainedmodels/model_myco_nuc_2/model -n 2 -l 4000 --steps 10 -t 0.0001 -s 16 --seed 1 -v --target_file $data_dir/SCerevisiae/generated/targets/4kb_sharpnuc147_linker20_middepleted1kb.npz

# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/4kb_lowpol_10seq -n 10 -l 4000 --steps 500 -t 0.0001 -s 128 -mid --seed 2 -v
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/4kb_lowpol_10seq_randomflanks -n 10 -l 4000 --steps 500 -t 0.0001 -s 128 -mid --seed 3 -v --flanks random
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/4kb_highpol_10seq -n 10 -l 4000 --steps 500 -t 0.0001 -s 128 -mid --seed 4 -v  -targ 1
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/4kb_highpol_10seq_randomflanks -n 10 -l 4000 --steps 500 -t 0.0001 -s 128 -mid --seed 5 -v -targ 1 --flanks random
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/4kb_peakpol200bp_10seq -n 10 -l 4000 --steps 500 -t 0.0001 -s 128 -mid --seed 6 -v --target_file $data_dir/SCerevisiae/generated/targets/4kb_midpeak200b.npz
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/4kb_peakpol200bp_10seq_randomflanks -n 10 -l 4000 --steps 500 -t 0.0001 -s 128 -mid --seed 7 -v --target_file $data_dir/SCerevisiae/generated/targets/4kb_midpeak200b.npz --flanks random

# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/test99 -m $data_dir/SCerevisiae/Trainedmodels/model_myco_nuc_2/model -n 2 -l 4000 --steps 500 -t 0.0001 -s 20 --seed 1 -v --target_file $data_dir/SCerevisiae/generated/targets/4kb_sharpnuc147_linker20.npz
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/test100 -m $data_dir/SCerevisiae/Trainedmodels/model_myco_nuc_2/model -n 2 -l 4000 --steps 500 -t 0.0001 -s 20 --seed 1 -v --target_file $data_dir/SCerevisiae/generated/targets/4kb_sharpnuc147_linker20_middepleted1kb.npz

# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/test101 -n 2 -l 4000 --steps 500 -t 0.0001 -s 128 -mid --seed 8 -v --target_file $data_dir/SCerevisiae/generated/targets/4kb_midpeak500bp.npz
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/test102 -n 2 -l 4000 --steps 100 -t 0.0001 -s 128 -mid --seed 8 -v --target_file $data_dir/SCerevisiae/generated/targets/4kb_midpeaklin500bp.npz
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/test103 -n 2 -l 4000 --steps 500 -t 0.0001 -s 128 -mid --seed 9 -v -plen 500
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/test104 -n 2 -l 4000 --steps 500 -t 0.0001 -s 128 -mid --seed 9 -v -plen 500 -pshape gaussian
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/test105 -n 2 -l 4000 --steps 500 -t 0.0001 -s 128 -mid --seed 9 -v -plen 500 -pshape gaussian -stdf 0.167
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/test106 -n 2 -l 4000 --steps 500 -t 0.0001 -s 128 -mid --seed 9 -v -plen 200
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/test107 -n 2 -l 4000 --steps 500 -t 0.0001 -s 128 -mid --seed 9 -v -plen 200 -pshape gaussian
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/test108 -n 2 -l 4000 --steps 500 -t 0.0001 -s 128 -mid --seed 9 -v -plen 200 -pshape gaussian -stdf 0.167
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/test109 -n 2 -l 4000 --steps 500 -t 0.0001 -s 128 -mid --seed 9 -v -plen 1000
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/test110 -n 2 -l 4000 --steps 500 -t 0.0001 -s 128 -mid --seed 9 -v -plen 1000 -pshape gaussian
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/test111 -n 2 -l 4000 --steps 500 -t 0.0001 -s 128 -mid --seed 9 -v -plen 1000 -pshape gaussian -stdf 0.167
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/test112 -n 2 -l 4000 --steps 500 -t 0.0001 -s 16 -mid --seed 9 -v -plen 200
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/test113 -n 2 -l 4000 --steps 500 -t 0.0001 -s 32 -mid --seed 9 -v -plen 200
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/test114 -n 2 -l 4000 --steps 500 -t 0.0001 -s 64 -mid --seed 9 -v -plen 200

# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/test115 -n 2 -l 4000 --steps 500 -t 0.0001 -s 128 -mid -v -ilen 200 -ishape sigmoid -bg low high --seed 9
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/test116 -n 2 -l 4000 --steps 500 -t 0.0001 -s 128 -mid -v -ilen 200 -ishape sigmoid -bg high low --seed 9
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/test117 -n 2 -l 4000 --steps 500 -t 0.0001 -s 128 -mid -v -ilen 500 -ishape sigmoid -bg low high --seed 9
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/test118 -n 2 -l 4000 --steps 500 -t 0.0001 -s 128 -mid -v -ilen 500 -ishape sigmoid -bg high low --seed 9
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/test119 -n 2 -l 4000 --steps 500 -t 0.0001 -s 128 -mid -v -ilen 1000 -ishape sigmoid -bg low high --seed 9
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/test120 -n 2 -l 4000 --steps 500 -t 0.0001 -s 128 -mid -v -ilen 1000 -ishape sigmoid -bg high low --seed 9
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/test121 -m $data_dir/SCerevisiae/Trainedmodels/model_myco_nuc_2/model -n 2 -l 4000 --steps 500 -t 0.0001 -s 20 -v -ilen 0 -per 167 -plen 147 -pshape gaussian --seed 9
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/test122 -m $data_dir/SCerevisiae/Trainedmodels/model_myco_nuc_2/model -n 2 -l 4000 --steps 500 -t 0.0001 -s 20 -v -ilen 200 -per 167 -plen 147 -pshape gaussian --seed 9
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/test123 -m $data_dir/SCerevisiae/Trainedmodels/model_myco_nuc_2/model -n 2 -l 4000 --steps 500 -t 0.0001 -s 20 -v -ilen 500 -per 167 -plen 147 -pshape gaussian --seed 9
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/test124 -m $data_dir/SCerevisiae/Trainedmodels/model_myco_nuc_2/model -n 2 -l 4000 --steps 500 -t 0.0001 -s 20 -v -ilen 1000 -per 167 -plen 147 -pshape gaussian --seed 9
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/test125 -n 2 -l 4000 --steps 500 -t 0.0001 -s 128 -mid -v -targ 0 --flanks self --seed 9
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/4kb_lowpol_10seq_selfflank -n 10 -l 4000 --steps 500 -t 0.0001 -s 128 -mid -v -targ 0 --flanks self --seed 10

# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/test126 --start_seqs $data_dir/SCerevisiae/generated/test121/designed_seqs/mut_seqs_step499.npy --steps 500 -t 0.0001 -s 128 -mid -v -targ 0 --seed 0
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/test127 --start_seqs $data_dir/SCerevisiae/generated/test121/designed_seqs/mut_seqs_step499.npy --steps 500 -t 0.0001 -s 128 -mid -v -ilen 500 -ishape gaussian --seed 0
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/test128 --start_seqs $data_dir/SCerevisiae/generated/test121/designed_seqs/mut_seqs_step499.npy --steps 500 -t 0.0001 -s 128 -mid -v -ilen 500 -ishape sigmoid -bg low high --seed 0
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/test129 --start_seqs $data_dir/SCerevisiae/generated/test122/designed_seqs/mut_seqs_step499.npy --steps 500 -t 0.0001 -s 128 -mid -v -targ 0 --seed 0
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/test130 --start_seqs $data_dir/SCerevisiae/generated/test122/designed_seqs/mut_seqs_step499.npy --steps 500 -t 0.0001 -s 128 -mid -v -ilen 500 -ishape gaussian --seed 0
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/test131 --start_seqs $data_dir/SCerevisiae/generated/test122/designed_seqs/mut_seqs_step499.npy --steps 500 -t 0.0001 -s 128 -mid -v -ilen 500 -ishape sigmoid -bg low high --seed 0
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/test132 --start_seqs $data_dir/SCerevisiae/generated/test123/designed_seqs/mut_seqs_step499.npy --steps 500 -t 0.0001 -s 128 -mid -v -targ 0 --seed 0
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/test133 --start_seqs $data_dir/SCerevisiae/generated/test123/designed_seqs/mut_seqs_step499.npy --steps 500 -t 0.0001 -s 128 -mid -v -ilen 500 -ishape gaussian --seed 0
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/test134 --start_seqs $data_dir/SCerevisiae/generated/test123/designed_seqs/mut_seqs_step499.npy --steps 500 -t 0.0001 -s 128 -mid -v -ilen 500 -ishape sigmoid -bg low high --seed 0

# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/test135 --start_seqs $data_dir/SCerevisiae/generated/test121/designed_seqs/mut_seqs_step499.npy --steps 3 -t 0.0001 -s 128 -mid -v -targ 0 --seed 0
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/test136 --start_seqs $data_dir/SCerevisiae/generated/test121/designed_seqs/mut_seqs_step499.npy --steps 3 -t 0.0001 -s 128 -mid -v -ilen 500 -ishape gaussian --seed 0
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/test137 --start_seqs $data_dir/SCerevisiae/generated/test121/designed_seqs/mut_seqs_step499.npy --steps 3 -t 0.0001 -s 128 -mid -v -ilen 500 -ishape sigmoid -bg low high --seed 0

# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/4kb_lowpol_10seq_randomflanks_v2 -n 10 -l 4000 --steps 500 -t 0.0001 -s 128 -mid --seed 4 -v --flanks random
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/4kb_lowpol_10seq_selfflanks_v2 -n 10 -l 4000 --steps 500 -t 0.0001 -s 128 -mid --seed 5 -v --flanks self
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/test138 -m $data_dir/JB_seqdes/weight_CNN_RNA_seq_2001_12_8_4_SRR7131299.hdf5 -ord ATGC -n 2 -l 4000 --steps 500 -t 0.0001 -s 100 --seed 6 -v --flanks random
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/test139 -m $data_dir/JB_seqdes/weight_CNN_RNA_seq_2001_12_8_4_SRR7131299.hdf5 -ord ATGC -n 2 -l 4000 --steps 500 -t 0.0001 -s 100 --seed 6 -v --flanks random -targ 1 -targ_rev 0
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/test140 -m $data_dir/JB_seqdes/weight_CNN_RNA_seq_2001_12_8_4_SRR7131299.hdf5 -ord ATGC -n 2 -l 4000 --steps 500 -t 0.0001 -s 100 --seed 6 -v --flanks random -targ 0 -targ_rev 1
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/test141 -m $data_dir/JB_seqdes/weight_CNN_RNA_seq_2001_12_8_4_SRR7131299.hdf5 -ord ATGC -n 2 -l 4000 --steps 500 -t 0.0001 -s 100 --seed 7 -v --flanks random -targ 5 -targ_rev 0
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/test142 -m $data_dir/JB_seqdes/weight_CNN_RNA_seq_2001_12_8_4_SRR7131299.hdf5 -ord ATGC -n 2 -l 4000 --steps 500 -t 0.0001 -s 100 --seed 7 -v --flanks random -targ 0 -targ_rev 5
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/4kb_lowpol_10seq_flanksInt2_v2 -n 10 -l 4000 --steps 500 -t 0.0001 -s 128 -mid --seed 11 -v --flanks $data_dir/SCerevisiae/data/S288c_siteManon_Int2_1kbflanks_ACGTidx.npz
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/4kb_lowpol_10seq_flanksInt4_v2 -n 10 -l 4000 --steps 500 -t 0.0001 -s 128 -mid --seed 12 -v --flanks $data_dir/SCerevisiae/data/S288c_siteManon_Int4_1kbflanks_ACGTidx.npz
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/4kb_lowpol_10seq_flanksInt5_v2 -n 10 -l 4000 --steps 500 -t 0.0001 -s 128 -mid --seed 13 -v --flanks $data_dir/SCerevisiae/data/S288c_siteManon_Int5_1kbflanks_ACGTidx.npz
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/4kb_lowpol_10seq_flanksInt6_v2 -n 10 -l 4000 --steps 500 -t 0.0001 -s 128 -mid --seed 14 -v --flanks $data_dir/SCerevisiae/data/S288c_siteManon_Int6_1kbflanks_ACGTidx.npz
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/4kb_lowpol_10seq_flanksInt7_v2 -n 10 -l 4000 --steps 500 -t 0.0001 -s 128 -mid --seed 15 -v --flanks $data_dir/SCerevisiae/data/S288c_siteManon_Int7_1kbflanks_ACGTidx.npz
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/4kb_lowpol_10seq_flanksInt8_v2 -n 10 -l 4000 --steps 500 -t 0.0001 -s 128 -mid --seed 16 -v --flanks $data_dir/SCerevisiae/data/S288c_siteManon_Int8_1kbflanks_ACGTidx.npz
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/4kb_lowpol_10seq_flanksInt9_v2 -n 10 -l 4000 --steps 500 -t 0.0001 -s 128 -mid --seed 17 -v --flanks $data_dir/SCerevisiae/data/S288c_siteManon_Int9_1kbflanks_ACGTidx.npz
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/4kb_lowpol_10seq_flanksInt10_v2 -n 10 -l 4000 --steps 500 -t 0.0001 -s 128 -mid --seed 18 -v --flanks $data_dir/SCerevisiae/data/S288c_siteManon_Int10_1kbflanks_ACGTidx.npz

# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/4kb_regnuc_2seq_randomflanks -m $data_dir/SCerevisiae/Trainedmodels/model_myco_nuc_2/model -n 2 -l 4000 --steps 500 -t 0.0001 -s 20 --flanks random -ilen 0 -per 167 -plen 147 -pshape gaussian --seed 10 -v
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/4kb_regnuc_gc0.2_2seq_randomflanks -m $data_dir/SCerevisiae/Trainedmodels/model_myco_nuc_2/model -n 2 -l 4000 --steps 500 -t 0.0001 -s 20 --flanks random -ilen 0 -per 167 -plen 147 -pshape gaussian --seed 11 -v -targ_gc 0.2

# python $scripts_dir/kMC_sequence_design_v2.py -o $data_dir/SCerevisiae/generated/test143 \
#     -n 2 -l 4000 --steps 500 -t 0.0001 -s 128 -mid --seed 0 -v --flanks random -p 3
# python $scripts_dir/kMC_sequence_design_v2.py -o $data_dir/SCerevisiae/generated/test144 \
#     -m $data_dir/JB_seqdes/weight_CNN_RNA_seq_2001_12_8_4_SRR7131299.hdf5 -ord ATGC \
#     -n 2 -l 4000 --steps 100 -t 0.0001 -s 100 --seed 0 -v --flanks random -p 3 -gclen 4000
# python $scripts_dir/kMC_sequence_design_v2.py -o $data_dir/SCerevisiae/generated/test145 \
#     -n 1 -l 50000 --steps 500 -t 0.0001 -s 128 -mid --seed 0 -v --flanks random -p 3
# python $scripts_dir/kMC_sequence_design_v2.py -o $data_dir/SCerevisiae/generated/test146 \
#     -n 2 -l 4000 --steps 100 -t 0.0001 -s 128 -mid --seed 4 -v -p 10 -gclen 4000
# python $scripts_dir/kMC_sequence_design_v2.py -o $data_dir/SCerevisiae/generated/test147 \
#     -n 2 -l 4000 --steps 100 -t 0.0001 -s 128 -mid --seed 4 -v --flanks random -p 10 -gclen 4000
# python $scripts_dir/kMC_sequence_design_v2.py -o $data_dir/SCerevisiae/generated/test148 \
#     --flanks $data_dir/SCerevisiae/data/S288c_siteManon_Int2_1kbflanks_ACGTidx.npz \
#     -n 2 -l 4000 --steps 100 -t 0.0001 -s 128 -mid --seed 4 -v -p 10 -gclen 4000
# python $scripts_dir/kMC_sequence_design_v2.py -o $data_dir/SCerevisiae/generated/test149 \
#     --flanks $data_dir/SCerevisiae/data/S288c_siteManon_Int2_1kbflanks_ACGTidx.npz \
#     -n 2 -l 6000 --steps 100 -t 0.0001 -s 128 -mid --seed 4 -v -p 10 -gclen 4000
# python $scripts_dir/kMC_sequence_design_v2.py -o $data_dir/SCerevisiae/generated/test150 \
#     --flanks $data_dir/SCerevisiae/data/S288c_siteManon_Int2_1kbflanks_ACGTidx.npz \
#     -n 2 -l 6000 --steps 100 -t 0.0001 -s 128 -mid --seed 4 -v -p 10 -gclen 4000
# python $scripts_dir/kMC_sequence_design_v2.py -o $data_dir/SCerevisiae/generated/test151 \
#     --flanks $data_dir/SCerevisiae/data/S288c_siteManon_Int2_1kbflanks_ACGTidx.npz \
#     -n 2 -l 6000 --steps 100 -t 0.0001 -s 128 -mid --seed 0 -v -p 10 -gclen 4000

# python $scripts_dir/kMC_sequence_design_v2.py -o $data_dir/SCerevisiae/generated/50kb_lowpol_1seq_flanksInt2 \
#     --flanks $data_dir/SCerevisiae/data/S288c_siteManon_Int2_1kbflanks_ACGTidx.npz \
#     -n 1 -l 50000 --steps 500 -t 0.0001 -s 128 -mid --seed 0 -v -p 10 -gclen 4000
# python $scripts_dir/kMC_sequence_design_v2.py -o $data_dir/SCerevisiae/generated/50kb_lowpol_1seq_flanksInt4 \
#     --flanks $data_dir/SCerevisiae/data/S288c_siteManon_Int4_1kbflanks_ACGTidx.npz \
#     -n 1 -l 50000 --steps 500 -t 0.0001 -s 128 -mid --seed 1 -v -p 10 -gclen 4000
# python $scripts_dir/kMC_sequence_design_v2.py -o $data_dir/SCerevisiae/generated/50kb_lowpol_1seq_flanksInt5 \
#     --flanks $data_dir/SCerevisiae/data/S288c_siteManon_Int5_1kbflanks_ACGTidx.npz \
#     -n 1 -l 50000 --steps 500 -t 0.0001 -s 128 -mid --seed 2 -v -p 10 -gclen 4000
# python $scripts_dir/kMC_sequence_design_v2.py -o $data_dir/SCerevisiae/generated/50kb_lowpol_1seq_flanksInt6 \
#     --flanks $data_dir/SCerevisiae/data/S288c_siteManon_Int6_1kbflanks_ACGTidx.npz \
#     -n 1 -l 50000 --steps 500 -t 0.0001 -s 128 -mid --seed 3 -v -p 10 -gclen 4000
# python $scripts_dir/kMC_sequence_design_v2.py -o $data_dir/SCerevisiae/generated/50kb_lowpol_1seq_flanksInt7 \
#     --flanks $data_dir/SCerevisiae/data/S288c_siteManon_Int7_1kbflanks_ACGTidx.npz \
#     -n 1 -l 50000 --steps 500 -t 0.0001 -s 128 -mid --seed 4 -v -p 10 -gclen 4000
# python $scripts_dir/kMC_sequence_design_v2.py -o $data_dir/SCerevisiae/generated/50kb_lowpol_1seq_flanksInt8 \
#     --flanks $data_dir/SCerevisiae/data/S288c_siteManon_Int8_1kbflanks_ACGTidx.npz \
#     -n 1 -l 50000 --steps 500 -t 0.0001 -s 128 -mid --seed 5 -v -p 10 -gclen 4000
# python $scripts_dir/kMC_sequence_design_v2.py -o $data_dir/SCerevisiae/generated/50kb_lowpol_1seq_flanksInt9 \
#     --flanks $data_dir/SCerevisiae/data/S288c_siteManon_Int9_1kbflanks_ACGTidx.npz \
#     -n 1 -l 50000 --steps 500 -t 0.0001 -s 128 -mid --seed 6 -v -p 10 -gclen 4000
# python $scripts_dir/kMC_sequence_design_v2.py -o $data_dir/SCerevisiae/generated/50kb_lowpol_1seq_flanksInt10 \
#     --flanks $data_dir/SCerevisiae/data/S288c_siteManon_Int10_1kbflanks_ACGTidx.npz \
#     -n 1 -l 50000 --steps 500 -t 0.0001 -s 128 -mid --seed 7 -v -p 10 -gclen 4000
# python $scripts_dir/kMC_sequence_design_v2.py -o $data_dir/SCerevisiae/generated/4kb_regnuc_1seq_flanksInt2 \
#     -m $data_dir/SCerevisiae/Trainedmodels/model_myco_nuc_2/model \
#     -ilen 0 -per 167 -plen 147 -pshape gaussian \
#     --flanks $data_dir/SCerevisiae/data/S288c_siteManon_Int2_1kbflanks_ACGTidx.npz \
#     -n 1 -l 4000 --steps 500 -t 0.0001 -s 20 --seed 0 -v -p 10 -gclen 4000
# python $scripts_dir/kMC_sequence_design_v2.py -o $data_dir/SCerevisiae/generated/4kb_regnuc_gc0.2_1seq_flanksInt2 \
#     -m $data_dir/SCerevisiae/Trainedmodels/model_myco_nuc_2/model \
#     -ilen 0 -per 167 -plen 147 -pshape gaussian \
#     --flanks $data_dir/SCerevisiae/data/S288c_siteManon_Int2_1kbflanks_ACGTidx.npz \
#     -n 1 -l 4000 --steps 500 -t 0.0001 -s 20 --seed 1 -v -p 10 -gclen 4000 -targ_gc 0.2
# python $scripts_dir/kMC_sequence_design_v2.py -o $data_dir/SCerevisiae/generated/4kb_regnuc_gc0.6_1seq_flanksInt2 \
#     -m $data_dir/SCerevisiae/Trainedmodels/model_myco_nuc_2/model \
#     -ilen 0 -per 167 -plen 147 -pshape gaussian \
#     --flanks $data_dir/SCerevisiae/data/S288c_siteManon_Int2_1kbflanks_ACGTidx.npz \
#     -n 1 -l 4000 --steps 500 -t 0.0001 -s 20 --seed 2 -v -p 10 -gclen 4000 -targ_gc 0.6
# python $scripts_dir/kMC_sequence_design_v2.py -o $data_dir/SCerevisiae/generated/4kb_regnuc_gc0.8_1seq_flanksInt2 \
#     -m $data_dir/SCerevisiae/Trainedmodels/model_myco_nuc_2/model \
#     -ilen 0 -per 167 -plen 147 -pshape gaussian \
#     --flanks $data_dir/SCerevisiae/data/S288c_siteManon_Int2_1kbflanks_ACGTidx.npz \
#     -n 1 -l 4000 --steps 500 -t 0.0001 -s 20 --seed 3 -v -p 10 -gclen 4000 -targ_gc 0.8

# python $scripts_dir/kMC_sequence_design_v2.py -o $data_dir/SCerevisiae/generated/50kb_lowpol_1seq_flanksInt2_p50 \
#     --flanks $data_dir/SCerevisiae/data/S288c_siteManon_Int2_1kbflanks_ACGTidx.npz \
#     -n 1 -l 50000 --steps 500 -t 0.0001 -s 128 -mid --seed 8 -v -p 50 -gclen 4000
# python $scripts_dir/kMC_sequence_design_v2.py -o $data_dir/SCerevisiae/generated/50kb_lowpol_2seq_flanksInt2_p50 \
#     --flanks $data_dir/SCerevisiae/data/S288c_siteManon_Int2_1kbflanks_ACGTidx.npz \
#     -n 2 -l 50000 --steps 500 -t 0.0001 -s 128 -mid --seed 9 -v -p 50 -gclen 4000
# python $scripts_dir/kMC_sequence_design_v2.py -o $data_dir/SCerevisiae/generated/50kb_lowpol_1seq_flanksInt2_p50_nmutstep50k \
#     --flanks $data_dir/SCerevisiae/data/S288c_siteManon_Int2_1kbflanks_ACGTidx.npz \
#     -n 1 -l 50000 --steps 500 -t 0.0001 -s 128 -mid --seed 10 -v -p 50 -gclen 4000 --nmut_step 50000

# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/4kb_regnuc_2seq_randomflanks -m $data_dir/SCerevisiae/Trainedmodels/model_myco_nuc_2/model -n 2 -l 4000 --steps 500 -t 0.0001 -s 20 --flanks random -ilen 0 -per 167 -plen 147 -pshape gaussian --seed 10 -v
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/4kb_regnuc_gc0.2_2seq_randomflanks -m $data_dir/SCerevisiae/Trainedmodels/model_myco_nuc_2/model -n 2 -l 4000 --steps 500 -t 0.0001 -s 20 --flanks random -ilen 0 -per 167 -plen 147 -pshape gaussian --seed 11 -v -targ_gc 0.2
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/4kb_regnuc_2seq_randomflanks_nrl163 -m $data_dir/SCerevisiae/Trainedmodels/model_myco_nuc_2/model -n 1 -l 4000 --steps 100 -t 0.0001 -s 20 --flanks random -ilen 0 -per 163 -plen 147 -pshape gaussian --seed 12 -v
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/4kb_regnuc_2seq_randomflanks_nrl165 -m $data_dir/SCerevisiae/Trainedmodels/model_myco_nuc_2/model -n 1 -l 4000 --steps 100 -t 0.0001 -s 20 --flanks random -ilen 0 -per 165 -plen 147 -pshape gaussian --seed 12 -v
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/4kb_regnuc_2seq_randomflanks_nrl169 -m $data_dir/SCerevisiae/Trainedmodels/model_myco_nuc_2/model -n 1 -l 4000 --steps 100 -t 0.0001 -s 20 --flanks random -ilen 0 -per 169 -plen 147 -pshape gaussian --seed 12 -v
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/4kb_regnuc_2seq_randomflanks_nrl171 -m $data_dir/SCerevisiae/Trainedmodels/model_myco_nuc_2/model -n 1 -l 4000 --steps 100 -t 0.0001 -s 20 --flanks random -ilen 0 -per 171 -plen 147 -pshape gaussian --seed 12 -v
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/4kb_regnuc_2seq_randomflanks_nrl173 -m $data_dir/SCerevisiae/Trainedmodels/model_myco_nuc_2/model -n 1 -l 4000 --steps 100 -t 0.0001 -s 20 --flanks random -ilen 0 -per 173 -plen 147 -pshape gaussian --seed 12 -v
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/4kb_regnuc_2seq_randomflanks_nrl175 -m $data_dir/SCerevisiae/Trainedmodels/model_myco_nuc_2/model -n 1 -l 4000 --steps 100 -t 0.0001 -s 20 --flanks random -ilen 0 -per 175 -plen 147 -pshape gaussian --seed 12 -v
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/4kb_regnuc_2seq_randomflanks_nrl177 -m $data_dir/SCerevisiae/Trainedmodels/model_myco_nuc_2/model -n 1 -l 4000 --steps 100 -t 0.0001 -s 20 --flanks random -ilen 0 -per 177 -plen 147 -pshape gaussian --seed 12 -v
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/4kb_regnuc_2seq_randomflanks_amp0.8 -m $data_dir/SCerevisiae/Trainedmodels/model_myco_nuc_2/model -n 1 -l 4000 --steps 100 -t 0.0001 -s 20 --flanks random -ilen 0 -per 167 -plen 147 -amp 0.8 -pshape gaussian --seed 12 -v
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/4kb_regnuc_2seq_randomflanks_amp0.6 -m $data_dir/SCerevisiae/Trainedmodels/model_myco_nuc_2/model -n 1 -l 4000 --steps 100 -t 0.0001 -s 20 --flanks random -ilen 0 -per 167 -plen 147 -amp 0.6 -pshape gaussian --seed 12 -v

# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/20kb_lowpol_1seq_flanksInt2 -n 1 -l 20000 --steps 100 -t 0.0001 -s 128 -mid --seed 20 -v --flanks $data_dir/SCerevisiae/data/S288c_siteManon_Int2_1kbflanks_ACGTidx.npz
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/20kb_lowpol_1seq_flanksInt2_longer --start_seqs $data_dir/SCerevisiae/generated/20kb_lowpol_1seq_flanksInt2/designed_seqs/start_seqs.npy --steps 600 -t 0.0001 -s 128 -mid --seed 20 -v --flanks $data_dir/SCerevisiae/data/S288c_siteManon_Int2_1kbflanks_ACGTidx.npz
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/20kb_lowpol_1seq_flanksInt2_longer_continued --start_seqs $data_dir/SCerevisiae/generated/20kb_lowpol_1seq_flanksInt2_longer/designed_seqs/mut_seqs_step599.npy --steps 600 -t 0.0001 -s 128 -mid --seed 20 -v --flanks $data_dir/SCerevisiae/data/S288c_siteManon_Int2_1kbflanks_ACGTidx.npz
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/20kb_lowpol_1seq_flanksInt2_longer_continued2 --start_seqs $data_dir/SCerevisiae/generated/20kb_lowpol_1seq_flanksInt2_longer_continued/designed_seqs/mut_seqs_step599.npy --steps 200 -t 0.0001 -s 128 -mid --seed 20 -v --flanks $data_dir/SCerevisiae/data/S288c_siteManon_Int2_1kbflanks_ACGTidx.npz
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/20kb_lowpol_1seq_flanksInt2_longer_continued3 --start_seqs $data_dir/SCerevisiae/generated/20kb_lowpol_1seq_flanksInt2_longer_continued2/designed_seqs/mut_seqs_step199.npy --steps 700 -t 0.0001 -s 128 -mid --seed 20 -v --flanks $data_dir/SCerevisiae/data/S288c_siteManon_Int2_1kbflanks_ACGTidx.npz
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/20kb_lowpol_1seq_flanksInt2_longer_continued4 --start_seqs $data_dir/SCerevisiae/generated/20kb_lowpol_1seq_flanksInt2_longer_continued3/designed_seqs/mut_seqs_step699.npy --steps 100 -t 0.0001 -s 128 -mid --seed 20 -v --flanks $data_dir/SCerevisiae/data/S288c_siteManon_Int2_1kbflanks_ACGTidx.npz

# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/test153 --steps 5 -t 0.0001 -s 128 -mid --seed 20 -v --flanks $data_dir/SCerevisiae/data/S288c_siteManon_Int2_1kbflanks_ACGTidx.npz
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/test154 -targ_gc 0.2 --steps 5 -t 0.0001 -s 128 -mid --seed 20 -v --flanks $data_dir/SCerevisiae/data/S288c_siteManon_Int2_1kbflanks_ACGTidx.npz
# python $scripts_dir/kMC_sequence_design.py -o $data_dir/SCerevisiae/generated/test155 -kfile $data_dir/SCerevisiae/genome/W303/W303_3mer_freq.csv --steps 5 -t 0.0001 -s 128 -mid --seed 20 -v --flanks $data_dir/SCerevisiae/data/S288c_siteManon_Int2_1kbflanks_ACGTidx.npz
# python $scripts_dir/kMC_sequence_design.py \
#     -o $data_dir/SCerevisiae/generated/20kb_lowpol_1seq_from_concat4kb_first5_flanksInt2 \
#     --start_seqs $data_dir/SCerevisiae/generated/4kb_lowpol_10seq_flanksInt2_v2/designed_seqs/concat_first5.npy \
#     -kfile $data_dir/SCerevisiae/genome/W303/W303_3mer_freq.csv \
#     --steps 500 -t 0.0001 -s 128 -mid --seed 20 -v \
#     --flanks $data_dir/SCerevisiae/data/S288c_siteManon_Int2_1kbflanks_ACGTidx.npz
# python $scripts_dir/kMC_sequence_design.py \
#     -o $writing_dir/SCerevisiae/generated/4kb_regnuc_2seq_randomflanks_W303_3mer \
#     -kfile $data_dir/SCerevisiae/genome/W303/W303_3mer_freq.csv \
#     -m $data_dir/SCerevisiae/Trainedmodels/model_myco_nuc_2/model \
#     --loss mae_cor \
#     -n 2 -l 4000 --steps 100 -t 0.0001 -s 20 --flanks random -ilen 0 -per 167 -plen 147 -pshape gaussian --seed 12 -v
# python $scripts_dir/kMC_sequence_design.py \
#     -o $data_dir/SCerevisiae/generated/19kb_lowpol_1seq_from_concat4kb_first5_withoutfirstkb_flanksforInt2 \
#     --start_seqs $data_dir/SCerevisiae/generated/4kb_lowpol_10seq_flanksInt2_v2/designed_seqs/concat_first5_withoutfirstkb.npy \
#     --flanks $data_dir/SCerevisiae/generated/4kb_lowpol_10seq_flanksInt2_v2/designed_seqs/lowpolforInt2_1kbflanks_ACGTidx.npz \
#     --steps 100 -t 0.0001 -s 128 -mid --seed 20 -v \
# python $scripts_dir/kMC_sequence_design.py \
#     -o $data_dir/SCerevisiae/generated/4kb_lowpol_10seq_randomflanks_v4 \
#     -kfile $data_dir/SCerevisiae/genome/W303/W303_3mer_freq.csv \
#     -n 10 -l 4000 --steps 500 -t 0.0001 -s 128 -mid --seed 101 -v \
#     --flanks random
# python $scripts_dir/kMC_sequence_design.py \
#     -o $data_dir/SCerevisiae/generated/4kb_lowpol_10seq_randomflanks_v5 \
#     -kfile $data_dir/SCerevisiae/genome/W303/W303_3mer_freq.csv \
#     -n 10 -l 4000 --steps 500 -t 0.0001 -s 128 -mid --seed 102 -v \
#     --flanks random
# python $scripts_dir/kMC_sequence_design.py \
#     -o $data_dir/SCerevisiae/generated/4kb_lowpol_10seq_randomflanks_v6 \
#     -kfile $data_dir/SCerevisiae/genome/W303/W303_3mer_freq.csv \
#     -n 10 -l 4000 --steps 500 -t 0.0001 -s 128 -mid --seed 103 -v \
#     --flanks random
# python $scripts_dir/kMC_sequence_design.py \
#     -o $data_dir/SCerevisiae/generated/4kb_lowpol_10seq_randomflanks_v7 \
#     -kfile $data_dir/SCerevisiae/genome/W303/W303_3mer_freq.csv \
#     -n 10 -l 4000 --steps 500 -t 0.0001 -s 128 -mid --seed 104 -v \
#     --flanks random
# python $scripts_dir/kMC_sequence_design.py \
#     -o $data_dir/SCerevisiae/generated/4kb_lowpol_10seq_randomflanks_v8 \
#     -kfile $data_dir/SCerevisiae/genome/W303/W303_3mer_freq.csv \
#     -n 10 -l 4000 --steps 500 -t 0.0001 -s 128 -mid --seed 105 -v \
#     --flanks random
    

# writing_dir='../'
# bbmap_dir='../bbmap'

# #### downloads
# data='NONO'
# # IP
# IP_file1='ENCFF767SQO'
# IP_file2='ENCFF527AMZ'
# wget https://www.encodeproject.org/files/$IP_file1/@@download/$IP_file1.fastq.gz -P $data_dir/$data/raw_data/IP
# wget https://www.encodeproject.org/files/$IP_file2/@@download/$IP_file2.fastq.gz -P $data_dir/$data/raw_data/IP
# # Control
# Control_file1='ENCFF002DUF'
# Control_file2='ENCFF002EGX'
# wget https://www.encodeproject.org/files/$Control_file1/@@download/$Control_file1.fastq.gz -P $data_dir/$data/raw_data/Control
# wget https://www.encodeproject.org/files/$Control_file2/@@download/Control_file2.fastq.gz -P $data_dir/$data/raw_data/Control
# # Unzip
# gunzip $data_dir/$data/raw_data/*/*.gz


# #### remove optical and PCR duplicates from fastq
# data='CENPT'
# label1='IP'
# label2='Control'
# access1='ENCFF098LPP'
# access2='ENCFF226MBW'
# access3='ENCFF494PEG'
# access4='ENCFF923IOZ'
# paired_end=false

# in1=$data_dir/$data/raw_data/$label1/$access1.fastq
# in2=$data_dir/$data/raw_data/$label1/$access2.fastq
# out1=$data_dir/$data/raw_data/$label1/$access1'_deduped'.fastq
# out2=$data_dir/$data/raw_data/$label1/$access2'_deduped'.fastq

# in3=$data_dir/$data/raw_data/$label2/$access3.fastq
# in4=$data_dir/$data/raw_data/$label2/$access4.fastq
# out3=$data_dir/$data/raw_data/$label2/$access3'_deduped'.fastq
# out4=$data_dir/$data/raw_data/$label2/$access4'_deduped'.fastq
# if [ $paired_end = true ]
# then
#     bash $bbmap_dir/clumpify.sh in=$in1 in2=$in2 out=$out1 out2=$out2 -subs=0 dedupe
#     bash $bbmap_dir/clumpify.sh in=$in3 in2=$in4 out=$out3 out2=$out4 -subs=0 dedupe
# else
#     bash $bbmap_dir/clumpify.sh in=$in1 out=$out1 -subs=0 dedupe
#     bash $bbmap_dir/clumpify.sh in=$in2 out=$out2 -subs=0 dedupe
#     bash $bbmap_dir/clumpify.sh in=$in3 out=$out3 -subs=0 dedupe
#     bash $bbmap_dir/clumpify.sh in=$in4 out=$out4 -subs=0 dedupe
# fi


# #### dataset creation

# # extract fasta
# data='Bombyx'
# X_file1='p50_dissect_D2-24h_cluster_X_chunck_300bp.fasta'
# nonX_file1='p50_dissect_D2-24h_cluster_non-X_chunck_300bp.fasta'
# python $scripts_dir/fastq_to_npz.py $writing_dir/$data/raw_data/X_reads $data_dir/$data/raw_data/X/$X_file1
# python $scripts_dir/fastq_to_npz.py $writing_dir/$data/raw_data/nonX_reads $data_dir/$data/raw_data/nonX/$nonX_file1

# # extract fastq
# data='NONO'
# IP_file1='ENCFF767SQO.fastq'
# IP_file2='ENCFF527AMZ.fastq'
# Control_file1='ENCFF002DUF.fastq'
# Control_file2='ENCFF002EGX.fastq'
# python $scripts_dir/fastq_to_npz.py $writing_dir/$data/raw_data/IP_reads $data_dir/$data/raw_data/IP/$IP_file1 $data_dir/$data/raw_data/IP/$IP_file2
# python $scripts_dir/fastq_to_npz.py $writing_dir/$data/raw_data/Control_reads $data_dir/$data/raw_data/Control/$Control_file1 $data_dir/$data/raw_data/Control/$Control_file2


# # build dataset
# data='CENPT'
# dataset='dataset'
# python $scripts_dir/build_dataset.py -ip $data_dir/$data/raw_data/IP_reads.npz -ctrl $data_dir/$data/raw_data/Control_reads.npz -out $data_dir/$data/$dataset -bal




# #### Training and evaluating

# data='CENPT'
# dataset='dataset'
# model_name='model_inception'
# architecture='inception_dna_v1'
# # create output directory
# mkdir $writing_dir/$data/Trainedmodels/$model_name/
# mkdir $writing_dir/$data/results/$model_name/
# # train model
# python $scripts_dir/Train_model.py -arch $architecture -d $data_dir/$data/$dataset.npz -out $writing_dir/$data/Trainedmodels/$model_name/ -ee -dist


# # save test predictions
# # data='Bombyx'
# # dataset='dataset'
# # model_name='model_Yann_original2'
# res_name='distrib_'$model_name'_test'
# python $scripts_dir/evaluate_model.py -m $writing_dir/$data/Trainedmodels/$model_name/model -d $data_dir/$data/$dataset.npz -out $writing_dir/$data/results/$model_name/$res_name


# # save train predictions
# # data='Bombyx'
# # dataset='dataset'
# # model_name='model_Yann_original2'
# res_name='distrib_'$model_name'_train'
# python $scripts_dir/evaluate_model.py -m $writing_dir/$data/Trainedmodels/$model_name/model -d $data_dir/$data/$dataset.npz -out $writing_dir/$data/results/$model_name/$res_name -data 'train'


# #### Relabeling and re-training
# data='CENPT'
# dataset='dataset'
# model_name='model_inception'
# new_dataset='dataset_rel55'
# python $scripts_dir/relabel_data.py -m $writing_dir/$data/Trainedmodels/$model_name/model -d $data_dir/$data/$dataset.npz -out $data_dir/$data/$new_dataset -t 0.55


# data='CENPT'
# dataset='dataset'
# new_labels='dataset_rel5'
# model_name='model_inception_rel5'
# architecture='inception_dna_v1'
# # create output directory
# mkdir $writing_dir/$data/Trainedmodels/$model_name/
# mkdir $writing_dir/$data/results/$model_name/
# # train model
# python $scripts_dir/Train_model.py -arch $architecture -d $data_dir/$data/$dataset.npz -out $writing_dir/$data/Trainedmodels/$model_name/ -rel $data_dir/$data/$new_labels.npz
# # save test predictions
# res_name='distrib_'$model_name'_test_rel'
# python $scripts_dir/evaluate_model.py -m $writing_dir/$data/Trainedmodels/$model_name/model -d $data_dir/$data/$dataset.npz -out $writing_dir/$data/results/$model_name/$res_name -rel $data_dir/$data/$new_labels.npz
# # save train predictions
# res_name='distrib_'$model_name'_train_rel'
# python $scripts_dir/evaluate_model.py -m $writing_dir/$data/Trainedmodels/$model_name/model -d $data_dir/$data/$dataset.npz -out $writing_dir/$data/results/$model_name/$res_name -data 'train' -rel $data_dir/$data/$new_labels.npz



# python $scripts_dir/saturation_analysis.py \
#     -o /home/alex/shared_folder/Judith-H3K9me3/results/alignments/T2T-CHM13v2.0/saturation_analysis/saturation_IP2_vs_INPUT_variousfracs \
#     -i /home/alex/shared_folder/Judith-H3K9me3/results/alignments/T2T-CHM13v2.0/HN00205099_RawFASTQ_RPE1_WTH3K9me3_paired_mid_points.bw \
#     -c /home/alex/shared_folder/Judith-H3K9me3/results/alignments/T2T-CHM13v2.0/D1145C44_trimmed_paired_T2T_mid_points.bw \
#     -b 10000 -f 0.01 0.05 0.1 0.2 0.5 0.75 1

# source activate align
# python $scripts_dir/bamCoverage_custom.py -p 18 \
#     -b /home/alex/shared_folder/Judith-H3K4me3/results/alignments/T2T-CHM13v2.0/D1145C41_trimmed_paired_T2T.sorted.bam \
#     -o /home/alex/shared_folder/Judith-H3K4me3/results/alignments/T2T-CHM13v2.0/D1145C41_trimmed_paired_T2T_nodup_mid_points.bw \
#     --binSize 1 --MidPoints --ignoreDuplicates
# python $scripts_dir/bamCoverage_custom.py -p 18 \
#     -b /home/alex/shared_folder/Judith-H3K4me3/results/alignments/T2T-CHM13v2.0/D1145C42_trimmed_paired_T2T.sorted.bam \
#     -o /home/alex/shared_folder/Judith-H3K4me3/results/alignments/T2T-CHM13v2.0/D1145C42_trimmed_paired_T2T_nodup_mid_points.bw \
#     --binSize 1 --MidPoints --ignoreDuplicates
# conda deactivate
# python $scripts_dir/saturation_analysis.py \
#     -o /home/alex/shared_folder/Judith-H3K4me3/results/alignments/T2T-CHM13v2.0/saturation_analysis/saturation_H3K4me3_vs_INPUT_powersof10 \
#     -i /home/alex/shared_folder/Judith-H3K4me3/results/alignments/T2T-CHM13v2.0/D1145C41_trimmed_paired_T2T_nodup_mid_points.bw \
#     -c /home/alex/shared_folder/Judith-H3K4me3/results/alignments/T2T-CHM13v2.0/D1145C42_trimmed_paired_T2T_nodup_mid_points.bw \
#     -b 100 1000 10000 100000 1000000 -f 0.001 0.01 0.1 1

# python $scripts_dir/saturation_analysis.py \
#     -o /home/alex/shared_folder/Judith-H3K4me3/results/alignments/T2T-CHM13v2.0/saturation_analysis/saturation_H3K4me3_vs_INPUT_powersof10_onlyip \
#     -i /home/alex/shared_folder/Judith-H3K4me3/results/alignments/T2T-CHM13v2.0/D1145C41_trimmed_paired_T2T_nodup_mid_points.bw \
#     -c /home/alex/shared_folder/Judith-H3K4me3/results/alignments/T2T-CHM13v2.0/D1145C42_trimmed_paired_T2T_nodup_mid_points.bw \
#     -b 100 1000 10000 100000 1000000 -f 0.001 0.01 0.1 1 -only_ip
# python $scripts_dir/saturation_analysis.py \
#     -o /home/alex/shared_folder/Judith-H3K9me3/results/alignments/T2T-CHM13v2.0/saturation_analysis/saturation_IP2_vs_INPUT_powersof10_onlyip \
#     -i /home/alex/shared_folder/Judith-H3K9me3/results/alignments/T2T-CHM13v2.0/HN00205099_RawFASTQ_RPE1_WTH3K9me3_paired_mid_points.bw \
#     -c /home/alex/shared_folder/Judith-H3K9me3/results/alignments/T2T-CHM13v2.0/D1145C44_trimmed_paired_T2T_mid_points.bw \
#     -b 100 1000 10000 100000 1000000 -f 0.001 0.01 0.1 1 -only_ip

# python $scripts_dir/saturation_analysis.py \
#     -o /home/alex/shared_folder/Judith-H3K4me3/results/alignments/T2T-CHM13v2.0/saturation_analysis/saturation_H3K4me3_vs_INPUT_powersof10_onlyip_b100 \
#     -i /home/alex/shared_folder/Judith-H3K4me3/results/alignments/T2T-CHM13v2.0/D1145C41_trimmed_paired_T2T_nodup_mid_points.bw \
#     -c /home/alex/shared_folder/Judith-H3K4me3/results/alignments/T2T-CHM13v2.0/D1145C42_trimmed_paired_T2T_nodup_mid_points.bw \
#     -b 100 -f 0.001 0.01 0.1 1 -only_ip