# find recursively files with specific name
find . -mindepth 1 -type f -name "*.fastq"
find . -mindepth 1 -type f -name "*.sam" -exec du -ch {} +

# bash arithmetic on wc
expr `wc -l < file` / 4

# check read lengths in fastq
awk 'NR%4 == 2 {lengths[length($0)]++ ; counter++} ENDFILE {for (l in lengths) {print l, lengths[l]}; print "total reads: " counter; delete lengths; counter=0}' $fastqs

# linearize, shuffle and rebuild fastq
awk '{OFS="\t"; getline seq; getline sep; getline qual; print $0,seq,sep,qual}' $fastq | shuf | awk -F '\t' '{OFS="\n"; print $1,$2,$3,$4}' > $new_fastq

# unwrap fasta
awk 'BEGIN {RS=">";FS="\n";OFS=""} NR>1 {print ">"$1; $1=""; print}' $fasta > $new_fasta

# count base occurence in fasta
awk '/^>/ {next} {for(i=1;i<=length($0);i++) {array[substr($1,i,1)]++}} END {for(key in array) {print key ": " array[key]}}' $fasta

# extract chromosome from fasta
awk 'BEGIN {RS=">";FS="\n"} /^CHRNAME/ {printf("%s%s",">",$0)}' $fasta > $new_fasta

# launch bash command, while saving the command and its terminal output to a file
bash_command="bash my_script.sh"
echo $bash_command > $log_file
script -a -c "$bash_command" $log_file
# In terminal use exit to stop scripting

# get chromsizes
samtools faidx $fasta
cut -f1,2 $fasta.fai > $chromsizes

# bam to BigWig
bamCoverage -p $threads -b $file.bam -o $file.bw --binSize 1 --extendReads

# check for adapter in compressed fastq (TruSeq example)
gunzip -c $fastq1.gz | grep AGATCGGAAGAGCACACGTCTGAACTCCAGTCA
gunzip -c $fastq2.gz | grep AGATCGGAAGAGCGTCGTGTAGGGAAAGAGTGT

# Dealing with background processes
jobs # see background processes
disown -p $pid # perd l'ownership du process
# Ctrl + Z stops current terminal process (pause) then type
bg %1 # bg is enough, 1 is the job number
# to resume as background process
