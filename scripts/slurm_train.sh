#!/bin/bash
job=${JOB:-'pascal_unimatch_v2_dinov2_small_366'}

# modify these augments if you want to try other datasets, splits or methods
# dataset: ['pascal', 'cityscapes', 'ade20k', 'coco']
# method: ['unimatch_v2', 'unimatch_v2_peft', 'scalematch', 'scalematch_peft', 'fixmatch', 'supervised']
# exp: just for specifying the 'save_path'
# split: ['92', '1_16', ...]. Please check directory './splits/$dataset' for concrete splits
dataset=${DATASET:-'pascal'}
method=${METHOD:-'unimatch_v2'}
exp=${EXP:-'dinov2_small'}
split=${SPLIT:-'366'}

config=${CONFIG:-configs/${dataset}.yaml}
labeled_id_path=splits/$dataset/$split/labeled.txt
unlabeled_id_path=splits/$dataset/$split/unlabeled.txt
save_path=exp/$dataset/$method/$exp/$split

mkdir -p $save_path

srun --mpi=pmi2 -p $3 -n $1 --gres=gpu:$1 --ntasks-per-node=$1 --job-name=$job \
    --open-mode=append -o $save_path/out.log --quotatype=reserved \
    python3 -u $method.py \
    --config=$config --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path \
    --save-path $save_path --port $2
