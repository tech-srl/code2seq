# Python150k dataset

## Steps to reproduce

1. Download parsed python dataset from [here](https://www.sri.inf.ethz.ch/py150
), unarchive and place under `PYTHON150K_DIR`:

```bash
# Replace with desired path.
>>> PYTHON150K_DIR=/path/to/data/dir
>>> mkdir -p $PYTHON150K_DIR
>>> cd $PYTHON150K_DIR
>>> wget http://files.srl.inf.ethz.ch/data/py150.tar.gz
...
>>> tar -xzvf py150.tar.gz
...
```

2. Extract samples to `DATA_DIR`:

```bash
# Replace with desired path.
>>> DATA_DIR=$(pwd)/data/default
>>> SEED=239
>>> python extract.py \
    --data_dir=$PYTHON150K_DIR \
    --output_dir=$DATA_DIR \
    --seed=$SEED
...
```

3. Preprocess for training:

```bash
>>> ./preprocess.sh $DATA_DIR
...
```

4. Train:

```bash
>>> cd ..
>>> DESC=default
>>> CUDA=0
>>> ./train_python150k.sh $DATA_DIR $DESC $CUDA $SEED
...
```

## Test results (seed=239)

### Best scores

**setup#2**: `batch_size=64`  
**setup#3**: `embedding_size=256,use_momentum=False`  
**setup#4**: `batch_size=32,embedding_size=256,embeddings_dropout_keep_prob=0.5,use_momentum=False`

| params | Precision | Recall | F1 | ROUGE-2 | ROUGE-L | 
|---|---|---|---|---|---|
| default | 0.37 | 0.27 | 0.31 | 0.06 | 0.38 |
| setup#2 | 0.40 | 0.31 | 0.34 | 0.08 | 0.41 |
| setup#3 | 0.36 | 0.31 | 0.33 | 0.09 | 0.38 |
| setup#4 | 0.33 | 0.25 | 0.28 | 0.05 | 0.34 |

### Ablation studies

| params | Precision | Recall | F1 | ROUGE-2 | ROUGE-L | 
|---|---|---|---|---|---|
| default | 0.37 | 0.27 | 0.31 | 0.06 | 0.38 |
| no ast nodes (5th epoch) | 0.27 | 0.16 | 0.20 | 0.02 | 0.28 |
| no token split (4th epoch) | 0.60 | 0.09 | 0.15 | 0.00 | 0.60 |