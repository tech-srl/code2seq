#!/usr/bin/env bash

data_dir=$1
data_name=$(basename "${data_dir}")
data=${data_dir}/${data_name}
test=${data_dir}/${data_name}.val.c2s
run_name=$2
model_dir=models/python150k-${run_name}
save_prefix=${model_dir}/model
cuda=${3:-0}
seed=${4:-239}

mkdir -p "${model_dir}"
set -e
CUDA_VISIBLE_DEVICES=$cuda python -u code2seq.py \
  --data="${data}" \
  --test="${test}" \
  --save_prefix="${save_prefix}" \
  --seed="${seed}"
