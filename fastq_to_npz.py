#!/usr/bin/env python

# This code extracts the sequences in fastq files and stores them in a numpy
# binary archive
#
# To execute this code run
# fastq_to_npz.py <output_file> <input_file1> <input_file2> ...
#
# parameters :
# - output_file : name of the file to write in
# - input_file : fastq files from which to extract data, must be in typical
#       | name     | format
#       | sequence |
#       | strand   |
#       | quality  |
#       if multiple input files are given, all the sequence are stored in the
#       same array

import sys
import os
import numpy as np

# Get the arguments from the command line and handle exceptions
try:
    output_file = sys.argv[1]
except IndexError as ie:
    raise SystemError("Error: Specify output file as 'fastq_to_npz.py"
                      " <output_file> <input_file1> <input_file2> ...'\n")

try:
    input_file = sys.argv[2]
except IndexError as ie:
    raise SystemError("Error: Specify at least one input file as"
                      " 'fastq_to_npz.py <output_file> <input_file1>"
                      " <input_file2> ...'\n")

# loop over input files to check if they exist
cur_arg = 2
while cur_arg < len(sys.argv):
    if not os.path.exists(sys.argv[cur_arg]):
        raise SystemError("Error: File ", sys.argv[cur_arg],
                          " does not exist\n")
    cur_arg += 1

# loop over input files to extract sequences
cur_arg = 2
while cur_arg < len(sys.argv):
    # list of all sequences
    reads = []
    with open(sys.argv[cur_arg], 'r') as f:
        # loop over file lines
        i = 0
        pos, neg, other = 0, 0, 0
        while True:
            id_line = f.readline()
            if not id_line:
                break
            seq_line = f.readline().rstrip()  # remove space at the end
            strand_line = f.readline().rstrip()
            quality_line = f.readline()
            reads.append(seq_line)
            if strand_line == '+':
                pos += 1
            elif strand_line == '-':
                neg += 1
            i += 1
        print(f'file {cur_arg-1} contains {pos} + reads and {neg} - reads')
    cur_arg += 1
reads = np.array(reads)
# convert the list to array and store in a numpy binary file
np.savez_compressed(output_file,
                    reads=reads)
