#!/bin/bash

#### downloads

# data='NONO'
# # IP
# IP_file1='ENCFF767SQO'
# IP_file2='ENCFF527AMZ'
# wget https://www.encodeproject.org/files/$IP_file1/@@download/$IP_file1.fastq.gz -P $data/raw_data/IP
# wget https://www.encodeproject.org/files/$IP_file2/@@download/$IP_file2.fastq.gz -P $data/raw_data/IP
# # Control
# Control_file1='ENCFF002DUF'
# Control_file2='ENCFF002EGX'
# wget https://www.encodeproject.org/files/$Control_file1/@@download/$Control_file1.fastq.gz -P $data/raw_data/Control
# wget https://www.encodeproject.org/files/$Control_file2/@@download/Control_file2.fastq.gz -P $data/raw_data/Control
# # Unzip
# gunzip $data/raw_data/*/*.gz



#### dataset creation

# # extract fastq
# data='NONO'
# IP_file1='ENCFF767SQO.fastq'
# IP_file2='ENCFF527AMZ.fastq'
# Control_file1='ENCFF002DUF.fastq'
# Control_file2='ENCFF002EGX.fastq'
# python fastq_to_npz.py $data/raw_data/IP_reads $data/raw_data/IP/$IP_file1 $data/raw_data/IP/$IP_file2
# python fastq_to_npz.py $data/raw_data/Control_reads $data/raw_data/Control/$Control_file1 $data/raw_data/Control/$Control_file2


# build dataset
# data='reads_test_mutasome'
# dataset='dataset'
# python build_dataset.py -ip $data/raw_data/IP_reads.npz -ctrl $data/raw_data/Control_reads.npz -out $data/$dataset -bal -20




#### Training and evaluating

# data='CENPT'
# dataset='dataset'
# model_dir='model_inception'
# architecture='Inception.py'
# # create output directory
# mkdir $data/Trainedmodels/$model_dir/
# mkdir $data/results/$model_dir/
# # train model
# python shared_folder/Train_model.py -arch $architecture -d shared_folder/$data/$dataset.npz -out $data/Trainedmodels/$model_dir/


# # save test predictions
# # data='Bombyx'
# # dataset='dataset'
# # model_dir='model_Yann_original2'
# res_name='distrib_'$model_dir'_test'
# python shared_folder/evaluate_model.py -m $data/Trainedmodels/$model_dir/model -d shared_folder/$data/$dataset.npz -out $data/results/$model_dir/$res_name


# # save train predictions
# # data='Bombyx'
# # dataset='dataset'
# # model_dir='model_Yann_original2'
# res_name='distrib_'$model_dir'_train'
# python shared_folder/evaluate_model.py -m $data/Trainedmodels/$model_dir/model -d shared_folder/$data/$dataset.npz -out $data/results/$model_dir/$res_name -data 'train'


#### Relabeling and re-training
# data='CENPT'
# dataset='dataset'
# model_dir='model_inception'
# new_dataset='dataset_rel55'
# python shared_folder/relabel_data.py -m $data/Trainedmodels/$model_dir/model -d shared_folder/$data/$dataset.npz -out $data/$new_dataset -t 0.55


# data='CENPT'
# dataset='dataset'
# new_labels='dataset_rel5'
# model_dir='model_inception_rel5'
# architecture='Inception.py'
# # create output directory
# mkdir $data/Trainedmodels/$model_dir/
# mkdir $data/results/$model_dir/
# # train model
# python shared_folder/Train_model.py -arch $architecture -d $data/$dataset.npz -out shared_folder/$data/Trainedmodels/$model_dir/ -rel $data/$new_labels.npz
# # save test predictions
# res_name='distrib_'$model_dir'_test_rel'
# python shared_folder/evaluate_model.py -m $data/Trainedmodels/$model_dir/model -d shared_folder/$data/$dataset.npz -out $data/results/$model_dir/$res_name -rel $data/$new_labels.npz
# # save train predictions
# res_name='distrib_'$model_dir'_train_rel'
# python shared_folder/evaluate_model.py -m $data/Trainedmodels/$model_dir/model -d shared_folder/$data/$dataset.npz -out $data/results/$model_dir/$res_name -data 'train' -rel $data/$new_labels.npz
