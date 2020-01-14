#!/usr/bin/env bash

MAX_CONTEXTS=200
MAX_DATA_CONTEXTS=1000
SUBTOKEN_VOCAB_SIZE=186277
TARGET_VOCAB_SIZE=26347

data_dir=${1:-data}
mkdir -p "${data_dir}"
train_data_file=$data_dir/train_output_file.txt
valid_data_file=$data_dir/valid_output_file.txt
test_data_file=$data_dir/test_output_file.txt

echo "Creating histograms from the training data..."
target_histogram_file=$data_dir/histo.tgt.c2s
source_subtoken_histogram=$data_dir/histo.ori.c2s
node_histogram_file=$data_dir/histo.node.c2s
cut <"${train_data_file}" -d' ' -f1 | tr '|' '\n' | awk '{n[$0]++} END {for (i in n) print i,n[i]}' >"${target_histogram_file}"
cut <"${train_data_file}" -d' ' -f2- | tr ' ' '\n' | cut -d',' -f1,3 | tr ',|' '\n' | awk '{n[$0]++} END {for (i in n) print i,n[i]}' >"${source_subtoken_histogram}"
cut <"${train_data_file}" -d' ' -f2- | tr ' ' '\n' | cut -d',' -f2 | tr '|' '\n' | awk '{n[$0]++} END {for (i in n) print i,n[i]}' >"${node_histogram_file}"

echo "Preprocessing..."
python ../preprocess.py \
  --train_data "${train_data_file}" \
  --val_data "${valid_data_file}" \
  --test_data "${test_data_file}" \
  --max_contexts ${MAX_CONTEXTS} \
  --max_data_contexts ${MAX_DATA_CONTEXTS} \
  --subtoken_vocab_size ${SUBTOKEN_VOCAB_SIZE} \
  --target_vocab_size ${TARGET_VOCAB_SIZE} \
  --target_histogram "${target_histogram_file}" \
  --subtoken_histogram "${source_subtoken_histogram}" \
  --node_histogram "${node_histogram_file}" \
  --output_name "${data_dir}"/"$(basename "${data_dir}")"
rm \
  "${target_histogram_file}" \
  "${source_subtoken_histogram}" \
  "${node_histogram_file}"
