#!/bin/bash

awk 'index($0, "tensorflow/core/common_runtime")==0&&index($0, "/job:localhost")==0&&index($0, "tensorflow/core/grappler/optimizers")==0&&index($0, "options.experimental_distribute.auto_shard_policy")==0 {print $0}' nohup.out > temp.txt
awk '/^(  |\}|attr|op:|input:)/ {next} {print $0}' temp.txt > parsed.txt
rm temp.txt