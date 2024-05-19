#!/usr/bin/env python
import argparse
import pyBigWig

parser = argparse.ArgumentParser(description="Convert the chromosome names of a bigWig file.")
parser.add_argument("conv", metavar="conversion.txt", help="Text file with two columns, the first a chromosome name and the second the converted chromosome name.")
parser.add_argument("input", metavar="input.bigWig", help="Input bigWig file")
parser.add_argument("output", metavar="output.bigWig", help="Output bigWig file name")
args = parser.parse_args()

# read in the name map
d = {}
f = open(args.conv)
for line in f:
    cols = line.strip().split("\t")
    if len(cols) < 2 or cols[1] == "":
         continue
    d[cols[0]] = cols[1]
f.close()

bw = pyBigWig.open(args.input)
# Make a new header
hdr = [(d[chrom], length) for chrom, length in bw.chroms().items() if chrom in d]
bwOutput = pyBigWig.open(args.output, "w")
bwOutput.addHeader(hdr)
for chrom, length in bw.chroms().items():
    ints = bw.intervals(chrom, 0, length)
    if len(ints):
        bwOutput.addEntries([d[chrom]] * len(ints),
                            [x[0] for x in ints],
                            ends=[x[1] for x in ints],
                            values=[x[2] for x in ints])
bw.close()
bwOutput.close()