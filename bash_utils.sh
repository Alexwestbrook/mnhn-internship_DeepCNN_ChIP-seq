# find recursively files with specific name
find . -mindepth 2 -type f -name "*.fastq"

# bash arithmetic on wc
expr `wc -l < file` / 4

# check read lengths in fastq
awk 'NR%4 == 2 {lengths[length($0)]++ ; counter++} END {for (l in lengths) {print l, lengths[l]}; print "total reads: " counter}' $fastq

# linearize, shuffle and rebuild fastq
awk '{OFS="\t"; getline seq; getline sep; getline qual; print $0,seq,sep,qual}' $fastq | shuf | awk -F '\t' '{OFS="\n"; print $1,$2,$3,$4}' > $new_fastq

# unwrap fasta
awk 'BEGIN {RS=">";FS="\n";OFS=""} NR>1 {print ">"$1; $1=""; print}' $fasta > $new_fasta

# count base occurence in fasta
awk '/^>/ {next} {for(i=1;i<=length($0);i++) {array[substr($1,i,1)]++}} END {for(key in array) {print key ": " array[key]}}' $fasta