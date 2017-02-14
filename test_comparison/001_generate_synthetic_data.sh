#!/bin/bash

opts="--firing_rate_min=0.5 --firing_rate_max=3 --M=8 --duration=1200"

./generate_synth_examples.node.js $opts --dsname=example1 --K=3 --noise_level=1
./generate_synth_examples.node.js $opts --dsname=example2 --K=8 --noise_level=1
./generate_synth_examples.node.js $opts --dsname=example3 --K=16 --noise_level=1
