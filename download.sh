#!/bin/bash

#### CTCF
# # IP
# wget https://www.encodeproject.org/files/ENCFF976CAW/@@download/ENCFF976CAW.fastq.gz
# wget https://www.encodeproject.org/files/ENCFF130AIS/@@download/ENCFF130AIS.fastq.gz
# # Control
# wget https://www.encodeproject.org/files/ENCFF007ATP/@@download/ENCFF007ATP.fastq.gz
# wget https://www.encodeproject.org/files/ENCFF056XEJ/@@download/ENCFF056XEJ.fastq.gz

#### H3K9me3
# # IP
# wget https://www.encodeproject.org/files/ENCFF782FNB/@@download/ENCFF782FNB.fastq.gz
# wget https://www.encodeproject.org/files/ENCFF026CAT/@@download/ENCFF026CAT.fastq.gz
# # Control
# wget https://www.encodeproject.org/files/ENCFF494PEG/@@download/ENCFF494PEG.fastq.gz
# wget https://www.encodeproject.org/files/ENCFF923IOZ/@@download/ENCFF923IOZ.fastq.gz

####H3K27me3
# # IP
# wget https://www.encodeproject.org/files/ENCFF080NOR/@@download/ENCFF080NOR.fastq.gz
# wget https://www.encodeproject.org/files/ENCFF346WYR/@@download/ENCFF346WYR.fastq.gz
# # Control
# same control as H3K9me3

####H3K27ac
# # IP
# wget https://www.encodeproject.org/files/ENCFF226MBW/@@download/ENCFF226MBW.fastq.gz -P H3K27ac/raw_data/IP/
# wget https://www.encodeproject.org/files/ENCFF098LPP/@@download/ENCFF098LPP.fastq.gz -P H3K27ac/raw_data/IP/
# # Control
# same control as CTCF
#
# try ENFF704XKC, ENCFF707CNX vs ENCFF273KUS, ENCFF466WDC

# CENPT bombyx
# # IP
# prefetch SRR12762381
# fasterq-dump SRR12762381 --outdir ../shared_folder/CENPT/raw_data/IP/
# prefetch SRR12762382
# fasterq-dump SRR12762382 --outdir ../shared_folder/CENPT/raw_data/IP/
# # Control
# prefetch SRR12762384
# fasterq-dump SRR12762384 --outdir ../shared_folder/CENPT/raw_data/Control/
# prefetch SRR12762385
# fasterq-dump SRR12762385 --outdir ../shared_folder/CENPT/raw_data/Control/


#### A549G4
# IP
prefetch SRR9603955
fasterq-dump SRR9603955 --outdir ../shared_folder/A549G4/raw_data/IP/
prefetch SRR9603953
fasterq-dump SRR9603953 --outdir ../shared_folder/A549G4/raw_data/IP/
fasterq-dump SRR9603955 --outdir ../shared_folder/A549G4/raw_data/IP/
# Control (Input)
prefetch SRR9603956
fasterq-dump SRR9603956 --outdir ../shared_folder/A549G4/raw_data/Control/
prefetch SRR9603954
fasterq-dump SRR9603954 --outdir ../shared_folder/A549G4/raw_data/Control/

#### HEK293-ZFAT
# # IP
# prefetch SRR9723246
# fasterq-dump SRR9723246 --outdir ../shared_folder/HEK293-ZFAT/raw_data/IP/
# # Control (Mock)
# prefetch SRR9723245
# fasterq-dump SRR9723245 --outdir ../shared_folder/HEK293-ZFAT/raw_data/Control/

#### NONO
# # IP
# wget https://www.encodeproject.org/files/ENCFF767SQO/@@download/ENCFF767SQO.fastq.gz
# wget https://www.encodeproject.org/files/ENCFF527AMZ/@@download/ENCFF527AMZ.fastq.gz
# # Control
# wget https://www.encodeproject.org/files/ENCFF002DUF/@@download/ENCFF002DUF.fastq.gz
# wget https://www.encodeproject.org/files/ENCFF002EGX/@@download/ENCFF002EGX.fastq.gz
