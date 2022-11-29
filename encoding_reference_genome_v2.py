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
from warnings import WarningMessage
import numpy as np
import json
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
        "-g", "--genome_file",
        help="genome in fasta format.",
        type=str,
        required=True)
    parser.add_argument(
        "-o", "--output",
        help="Path to the output directory and file name.",
        type=str,
        required=True)
    parser.add_argument(
        "-k", "--key_names",
        help='Dictionnary is json format {"key1":"value1"} mapping assembly '
             'identifiers for chromosomes and their desired key names in the '
             'npz archive. If specified, only these chromosomes are '
             'processed, otherwise all sequences are.',
        type=json.loads)
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
    if not Path(args.genome_file).is_file():
        sys.exit(f"{args.genome_file} does not exist.\n"
                 "Please enter a valid genome file path.")
    if args.method not in {1, 2}:
        sys.exit(f"{args.method} is not a valid method.")
    return args


# get arguments
args = parsing()
batch_size = args.batch_size

print(f"Writing {args.genome_file} to {args.output}.npz")
if args.timing:
    times = {}
    t0 = time.time()
# parse genome file into batches
with open(args.genome_file, 'r') as f:
    genome = {}
    n_seqs = 0
    sequence = ''
    skip = False
    for line in f:
        if line[0] == '>':  # First line header, discard this line
            # Save remaining sequence of previous chromosome
            if n_seqs >= 1 and len(sequence) != 0:
                genome[id].append(sequence)
                sequence = ''
            # Get new chromosome id
            id, *_ = line.split()
            id = id[1:]
            # Maybe convert to input key names
            if args.key_names:
                if id in args.key_names.keys():
                    id = args.key_names[id]
                    skip = False
                else:
                    skip = True
                    continue
            genome[id] = []
            n_seqs += 1
        else:
            if skip:
                continue
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
            genome[id] += batches
    # Add remaining incomplete batch, unless it is empty
    if len(sequence) != 0:
        genome[id].append(sequence)
    # If genome is empty, raise an error
    if len(genome) == 0:
        sys.exit(f"{args.genome_file} doesn't contain any sequence or isn't "
                 "in fasta format")
    for id, batches in genome.items():
        if len(batches) == 0:
            raise RuntimeWarning(f"{id} didn't match any sequence in "
                                 f"{args.genome_file}")
    # compute genome length in bases
    genome_bases = sum([(len(batches)-1)*batch_size + len(batches[-1])
                        for batches in genome.values()])
    nb_batches = sum(len(batches) for batches in genome.values())
    print(f'Processing {n_seqs} sequences '
          f'with {genome_bases} bases '
          f'into {nb_batches} batches')
if args.timing:
    times['parsing'] = time.time() - t0

# one-hot encoding
one_hot_genome = {}
if args.method == 0:
    for id, batches in genome.items():
        # process all batches at once, adding N to complete final batch
        one_hot_batches = utils.one_hot_encoding(
            batches,
            read_length=batch_size,
            one_hot_type=args.one_hot_type)
        # reshape into a single genome
        one_hot_genome[id] = one_hot_batches.reshape(
            (len(batches)*batch_size, 4))
elif args.method == 1:
    for id, batches in genome.items():
        # initialize empty array
        id_bases = (len(batches)-1)*batch_size + len(batches[-1])
        one_hot_batches = np.empty((id_bases, 4), dtype=args.one_hot_type)
        # process batches individually and write them in one_hot_genome
        for i, batch in enumerate(batches[:-1]):
            one_hot_batch = utils.one_hot_encoding(
                [batch],
                read_length=batch_size,
                one_hot_type=args.one_hot_type)[0]
            one_hot_batches[i*batch_size:(i+1)*batch_size] = one_hot_batch
        # process last batch seperately as it can have different length
        one_hot_batch = utils.one_hot_encoding(
            [batches[-1]],
            read_length=len(batches[-1]),
            one_hot_type=args.one_hot_type)[0]
        one_hot_batches[(len(batches)-1)*batch_size:] = one_hot_batch
        one_hot_genome[id] = one_hot_batches
else:
    print("This method is not supported")
if args.timing:
    times['encoding'] = time.time() - t0

# test that reconversion gives the same sequence
if args.testing:
    print('testing...')
    for id, one_hot_batches in one_hot_genome.items():
        decoded = utils.one_hot_to_seq(np.array([one_hot_batches]))[0]
        if args.timing:
            times['decoding'] = time.time() - t0

        flat_sequence = "".join(genome[id]).upper()
        print(len(decoded)-len(flat_sequence),
              'extra bases in encoded genome')
        try:
            assert all(char == 'N' for char in decoded[len(flat_sequence):])
        except AssertionError:
            print("Encoded genome extra length contains information")
        else:
            print("Encoded genome extra length is just Ns")
        if args.timing:
            times['extra length checking'] = time.time() - t0

        expected_char_set = {'N', 'A', 'C', 'G', 'T'}
        char_set = set()
        for char in flat_sequence:
            if char not in char_set:
                char_set.add(char)
        try:
            assert char_set.issubset(expected_char_set)
        except AssertionError:
            unexpected_char_set = char_set - expected_char_set
            print(f'Got some unexpected bases: {unexpected_char_set}\n'
                  'They will be converted to N')
            for char in unexpected_char_set:
                flat_sequence = flat_sequence.replace(char, 'N')
        if args.timing:
            times['base checking'] = time.time() - t0

        assert (flat_sequence == sequence[:len(flat_sequence)])
        if args.timing:
            times['match checking'] = time.time() - t0

# save the one-hot encoded genome into a numpy binary file
np.savez_compressed(args.output, **one_hot_genome)
if args.timing:
    times['compression and saving'] = time.time() - t0
    start = 0
    for key in times.keys():
        print(key, '\t', times[key] - start)
        start = times[key]
