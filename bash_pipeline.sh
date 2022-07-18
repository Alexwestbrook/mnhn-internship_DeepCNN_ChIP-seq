#!/bin/bash

scripts_dir='.'
data_dir='../shared_folder'
writing_dir='../'
bbmap_dir='../bbmap'

#### downloads
data='NONO'
# IP
IP_file1='ENCFF767SQO'
IP_file2='ENCFF527AMZ'
wget https://www.encodeproject.org/files/$IP_file1/@@download/$IP_file1.fastq.gz -P $data_dir/$data/raw_data/IP
wget https://www.encodeproject.org/files/$IP_file2/@@download/$IP_file2.fastq.gz -P $data_dir/$data/raw_data/IP
# Control
Control_file1='ENCFF002DUF'
Control_file2='ENCFF002EGX'
wget https://www.encodeproject.org/files/$Control_file1/@@download/$Control_file1.fastq.gz -P $data_dir/$data/raw_data/Control
wget https://www.encodeproject.org/files/$Control_file2/@@download/Control_file2.fastq.gz -P $data_dir/$data/raw_data/Control
# Unzip
gunzip $data_dir/$data/raw_data/*/*.gz


#### remove optical and PCR duplicates from fastq
data='CENPT'
label1='IP'
label2='Control'
access1='ENCFF098LPP'
access2='ENCFF226MBW'
access3='ENCFF494PEG'
access4='ENCFF923IOZ'
paired_end=false

in1=$data_dir/$data/raw_data/$label1/$access1.fastq
in2=$data_dir/$data/raw_data/$label1/$access2.fastq
out1=$data_dir/$data/raw_data/$label1/$access1'_deduped'.fastq
out2=$data_dir/$data/raw_data/$label1/$access2'_deduped'.fastq

in3=$data_dir/$data/raw_data/$label2/$access3.fastq
in4=$data_dir/$data/raw_data/$label2/$access4.fastq
out3=$data_dir/$data/raw_data/$label2/$access3'_deduped'.fastq
out4=$data_dir/$data/raw_data/$label2/$access4'_deduped'.fastq
if [ $paired_end = true ]
then
    bash $bbmap_dir/clumpify.sh in=$in1 in2=$in2 out=$out1 out2=$out2 -subs=0 dedupe
    bash $bbmap_dir/clumpify.sh in=$in3 in2=$in4 out=$out3 out2=$out4 -subs=0 dedupe
else
    bash $bbmap_dir/clumpify.sh in=$in1 out=$out1 -subs=0 dedupe
    bash $bbmap_dir/clumpify.sh in=$in2 out=$out2 -subs=0 dedupe
    bash $bbmap_dir/clumpify.sh in=$in3 out=$out3 -subs=0 dedupe
    bash $bbmap_dir/clumpify.sh in=$in4 out=$out4 -subs=0 dedupe
fi


#### dataset creation

# extract fastq
data='NONO'
IP_file1='ENCFF767SQO.fastq'
IP_file2='ENCFF527AMZ.fastq'
Control_file1='ENCFF002DUF.fastq'
Control_file2='ENCFF002EGX.fastq'
python $scripts_dir/fastq_to_npz.py $writing_dir/$data/raw_data/IP_reads $data_dir/$data/raw_data/IP/$IP_file1 $data_dir/$data/raw_data/IP/$IP_file2
python $scripts_dir/fastq_to_npz.py $writing_dir/$data/raw_data/Control_reads $data_dir/$data/raw_data/Control/$Control_file1 $data_dir/$data/raw_data/Control/$Control_file2


# build dataset
data='CENPT'
dataset='dataset'
python $scripts_dir/build_dataset.py -ip $data_dir/$data/raw_data/IP_reads.npz -ctrl $data_dir/$data/raw_data/Control_reads.npz -out $data_dir/$data/$dataset -bal -20




#### Training and evaluating

data='CENPT'
dataset='dataset'
model_name='model_inception'
architecture='inception_dna_v1'
# create output directory
mkdir $writing_dir/$data/Trainedmodels/$model_name/
mkdir $writing_dir/$data/results/$model_name/
# train model
python $scripts_dir/Train_model.py -arch $architecture -d $data_dir/$data/$dataset.npz -out $writing_dir/$data/Trainedmodels/$model_name/ -ee


# save test predictions
# data='Bombyx'
# dataset='dataset'
# model_name='model_Yann_original2'
res_name='distrib_'$model_name'_test'
python $scripts_dir/evaluate_model.py -m $writing_dir/$data/Trainedmodels/$model_name/model -d $data_dir/$data/$dataset.npz -out $writing_dir/$data/results/$model_name/$res_name


# save train predictions
# data='Bombyx'
# dataset='dataset'
# model_name='model_Yann_original2'
res_name='distrib_'$model_name'_train'
python $scripts_dir/evaluate_model.py -m $writing_dir/$data/Trainedmodels/$model_name/model -d $data_dir/$data/$dataset.npz -out $writing_dir/$data/results/$model_name/$res_name -data 'train'


#### Relabeling and re-training
data='CENPT'
dataset='dataset'
model_name='model_inception'
new_dataset='dataset_rel55'
python $scripts_dir/relabel_data.py -m $writing_dir/$data/Trainedmodels/$model_name/model -d $data_dir/$data/$dataset.npz -out $data_dir/$data/$new_dataset -t 0.55


data='CENPT'
dataset='dataset'
new_labels='dataset_rel5'
model_name='model_inception_rel5'
architecture='inception_dna_v1'
# create output directory
mkdir $writing_dir/$data/Trainedmodels/$model_name/
mkdir $writing_dir/$data/results/$model_name/
# train model
python $scripts_dir/Train_model.py -arch $architecture -d $data_dir/$data/$dataset.npz -out $writing_dir/$data/Trainedmodels/$model_name/ -rel $data_dir/$data/$new_labels.npz
# save test predictions
res_name='distrib_'$model_name'_test_rel'
python $scripts_dir/evaluate_model.py -m $writing_dir/$data/Trainedmodels/$model_name/model -d $data_dir/$data/$dataset.npz -out $writing_dir/$data/results/$model_name/$res_name -rel $data_dir/$data/$new_labels.npz
# save train predictions
res_name='distrib_'$model_name'_train_rel'
python $scripts_dir/evaluate_model.py -m $writing_dir/$data/Trainedmodels/$model_name/model -d $data_dir/$data/$dataset.npz -out $writing_dir/$data/results/$model_name/$res_name -data 'train' -rel $data_dir/$data/$new_labels.npz
