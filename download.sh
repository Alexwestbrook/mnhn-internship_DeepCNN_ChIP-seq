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
# wget s3://sra-pub-src-9/SRR12762381/V369C5.R1.fastq.gz.1
# wget s3://sra-pub-src-12/SRR12762382/A914C09.R1.fastq.gz.1
# # Control
# wget s3://sra-pub-src-13/SRR12762384/V369C4.R1.fastq.gz.1
# wget s3://sra-pub-src-13/SRR12762385/A914C10.R1.fastq.gz.1
# # method
# fastq-dump IP/SRR12762381.1
# fastq-dump IP/SRR12762382.1
# fastq-dump Control/SRR12762384.1
# fastq-dump Control/SRR12762385.1

#### A549G4
# IP
wget gs://sra-pub-run-13/SRR9603955/SRR9603955.1
wget gs://sra-pub-run-13/SRR9603953/SRR9603953.1
# Control
wget gs://sra-pub-run-13/SRR9603956/SRR9603956.1
wget gs://sra-pub-run-13/SRR9603954/SRR9603954.1

#### NONO
# # IP
# wget https://www.encodeproject.org/files/ENCFF767SQO/@@download/ENCFF767SQO.fastq.gz
# wget https://www.encodeproject.org/files/ENCFF527AMZ/@@download/ENCFF527AMZ.fastq.gz
# # Control
# wget https://www.encodeproject.org/files/ENCFF002DUF/@@download/ENCFF002DUF.fastq.gz
# wget https://www.encodeproject.org/files/ENCFF002EGX/@@download/ENCFF002EGX.fastq.gz
