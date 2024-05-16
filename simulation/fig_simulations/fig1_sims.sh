#!/bin/bash
export seed_str='1000000 1000001 1000002 1000003 1000004 1000005 1000006 1000007 1000008 1000009'

# MAKE DATASETS (only need to do this once)
python3 tasks/synth_dataset.py
seeds=($seed_str)
for seed in ${seeds[@]}
do
    python3 tasks/rotate_dataset.py $seed
done

