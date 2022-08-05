#!/bin/bash

awk 'index($0, "tensorflow/core/common_runtime")==0&&index($0, "/job:localhost") {print $0}' nohup.out > parsed.txt
awk '/^(  |\t|attr|op:|input:)/ {next} {print $0}' parsed.txt > parsed.txt