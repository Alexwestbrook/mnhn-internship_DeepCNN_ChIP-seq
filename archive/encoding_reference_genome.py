#!/usr/bin/env python

# This code takes a genome file in fasta format and encodes it as one_hot.
# The file is stored as a compressed npz archive
#
# To execute this code run
#   encoding_reference_genome.py -g <genome_file> -o <output_name>
# arguments
#   -g <genome_file>    must be in fasta format
#   -o <output_file>    the extension .npz will be appended to the file name,
#                       if the file already exists, it will be overriden
# options
#   -b <int>    specify batch size to use for encoding
#   -oht <type> specify type to use for one_hot_encoding

import sys
import argparse
import time
import numpy as np
from pathlib import Path
from Modules import utils


def parsing():
    """
    Parse the command-line arguments.

    Arguments
    ---------
    python command-line

    Returns
    -------
    args : object containing all arguments in parser
    """
    # Declaration of expexted arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-g", "--genome_files",
        help="genome in fasta format.",
        type=str,
        nargs="+",
        required=True)
    parser.add_argument(
        "-o", "--outputs",
        help="Path to the output directory and file name.",
        type=str,
        nargs="+",
        required=True)
    parser.add_argument(
        "-b", "--batch_size",
        help="size of batch of base pairs to encode at once, default to 1Mb.",
        default=1024*1024,
        type=int)
    parser.add_argument(
        "-oht", "--one_hot_type",
        help="type to use for one hot encoding, default to bool.",
        default=bool,
        type=type)
    parser.add_argument(
        "-m", "--method",
        help="method to use for encoding, 0 for all batches at once which "
             "will add Ns at the end of the genome, 1 for iterating over "
             "batches which will maintain genome length, default to 1",
        default=1,
        type=int)
    parser.add_argument(
        "-time", "--timing",
        action="store_true",
        help="indicates to time operations")
    parser.add_argument(
        "-test", "--testing",
        action="store_true",
        help="indicates to test results")
    args = parser.parse_args()
    # Check if the input data is valid
    if len(args.genome_files) != len(args.outputs):
        sys.exit("Please specify as many genome files as output files.")
    for genome_file in args.genome_files:
        if not Path(genome_file).is_file():
            sys.exit(f"{genome_file} does not exist.\n"
                     "Please enter valid genome file paths.")
    if args.method not in {1, 2}:
        sys.exit(f"{args.method} is not a valid method.")
    return args


# get arguments
args = parsing()
batch_size = args.batch_size

for genome_file, output in zip(args.genome_files, args.outputs):
    print(f"Writing {genome_file} to {output}.npz")
    if args.timing:
        times = {}
        t0 = time.time()
    # parse genome file into batches
    with open(genome_file, 'r') as f:
        genome = []
        n_seqs = 0
        sequence = ''
        for line in f:
            if line[0] == '>':  # First line header, discard this line
                if n_seqs == 1:
                    print("This implementation doesn't support multiple "
                          "sequences in fasta, all sequences will be "
                          "concatenated into one")
                n_seqs += 1
            else:
                sequence += line.rstrip()
                # split sequence into batches
                batches = [sequence[start:start+batch_size]
                           for start in range(0, len(sequence), batch_size)]
                if len(batches[-1]) < batch_size:
                    # if last batch is incomplete, forward it to next line
                    sequence = batches.pop(-1)
                else:
                    sequence = ''
                # add all complete batches
                genome += batches
        # Add remaining incomplete batch, unless it is empty
        if len(sequence) != 0:
            genome.append(sequence)
        # If genome is empty, raise an error
        if len(genome) == 0:
            sys.exit(f"{genome_file} doesn't contain any sequence or isn't in"
                     " fasta format")
        # compute genome length in bases
        genome_bases = (len(genome)-1)*batch_size + len(genome[-1])
        print(f'Processing {n_seqs} sequence{utils.s_plural(n_seqs)} '
              f'with {genome_bases} base{utils.s_plural(genome_bases)} '
              f'into {len(genome)} batche{utils.s_plural(len(genome))}')
    if args.timing:
        times['parsing'] = time.time() - t0

    # one-hot encoding
    if args.method == 0:
        # process all batches at once, adding N to complete final batch
        one_hot_batches = utils.one_hot_encoding(
            genome,
            read_length=batch_size,
            one_hot_type=args.one_hot_type)
        # reshape into a single genome
        one_hot_genome = one_hot_batches.reshape((len(genome)*batch_size, 4))
    elif args.method == 1:
        # initialze empty array
        one_hot_genome = np.empty((genome_bases, 4), dtype=args.one_hot_type)
        # process batches individually and write them in one_hot_genome
        for i, batch in enumerate(genome[:-1]):
            one_hot_batch = utils.one_hot_encoding(
                [batch],
                read_length=batch_size,
                one_hot_type=args.one_hot_type)[0]
            one_hot_genome[i*batch_size:(i+1)*batch_size] = one_hot_batch
        # process last batch seperately as it can have different length
        one_hot_batch = utils.one_hot_encoding(
            [genome[-1]],
            read_length=len(genome[-1]),
            one_hot_type=args.one_hot_type)[0]
        one_hot_genome[(len(genome)-1)*batch_size:] = one_hot_batch
    else:
        print("This method is not supported")
    if args.timing:
        times['encoding'] = time.time() - t0

    # test that reconversion gives the same sequence
    if args.testing:
        print('testing...')
        sequence = utils.one_hot_to_seq(np.array([one_hot_genome]))[0]
        if args.timing:
            times['decoding'] = time.time() - t0

        flat_genome = "".join(genome).upper()
        print(len(sequence)-len(flat_genome), 'extra bases in encoded genome')
        try:
            assert all(char == 'N' for char in sequence[len(flat_genome):])
        except AssertionError:
            print("Encoded genome extra length contains information")
        else:
            print("Encoded genome extra length is just Ns")
        if args.timing:
            times['extra length checking'] = time.time() - t0

        expected_char_set = {'N', 'A', 'C', 'G', 'T'}
        char_set = set()
        for char in flat_genome:
            if char not in char_set:
                char_set.add(char)
        try:
            assert char_set.issubset(expected_char_set)
        except AssertionError:
            unexpected_char_set = char_set - expected_char_set
            print(f'Got some unexpected bases: {unexpected_char_set}\n'
                  'They will be converted to N')
            for char in unexpected_char_set:
                flat_genome = flat_genome.replace(char, 'N')
        if args.timing:
            times['base checking'] = time.time() - t0

        assert (flat_genome == sequence[:len(flat_genome)])
        if args.timing:
            times['match checking'] = time.time() - t0

    # save the one-hot encoded genome into a numpy binary file
    np.savez_compressed(output, one_hot_genome=one_hot_genome)
    if args.timing:
        times['compression and saving'] = time.time() - t0
        start = 0
        for key in times.keys():
            print(key, '\t', times[key] - start)
            start = times[key]
