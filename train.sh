###########################################################
# Change the following values to train a new model.
# type: the name of the new model, only affects the saved file name.
# dataset: the name of the dataset, as was preprocessed using preprocess.sh
# test_data: by default, points to the validation set, since this is the set that
#   will be evaluated after each training iteration. If you wish to test
#   on the final (held-out) test set, change 'val' to 'test'.
type=java-large
dataset_name=java-large
data_dir=data/java-large-full/java-large
data=${data_dir}/${dataset_name}
test_data=${data_dir}/${dataset_name}.test.c2s
model_dir=models/${type}

mkdir -p ${model_dir}
set -e
python3 -u code2seq.py --data ${data} --test ${test_data} --save_prefix ${model_dir}/saved_model
