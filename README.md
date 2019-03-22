# code2seq
This is an official implementation of the model described in:

[Uri Alon](http://urialon.cswp.cs.technion.ac.il), [Shaked Brody](http://www.cs.technion.ac.il/people/shakedbr/), [Omer Levy](https://levyomer.wordpress.com) and [Eran Yahav](http://www.cs.technion.ac.il/~yahave/), "code2seq: Generating Sequences from Structured Representations of Code" [[PDF]](https://openreview.net/pdf?id=H1gKYo09tX)

to appear in *ICLR'2019*

An **online demo** is available at [https://code2seq.org](https://code2seq.org).

This is a TensorFlow implementation of the network, with Java and C# extractors for preprocessing the input code. 
It can be easily extended to other languages, 
since the TensorFlow network is agnostic to the input programming language (see [Extending to other languages](#extending-to-other-languages).
Contributions are welcome.

<center style="padding: 40px"><img width="70%" src="https://github.com/tech-srl/code2seq/raw/master/images/network.png" /></center>

Table of Contents
=================
  * [Requirements](#requirements)
  * [Quickstart](#quickstart)
  * [Configuration](#configuration)
  * [Releasing a trained mode](#releasing-a-trained-model)
  * [Extending to other languages](#extending-to-other-languages)
  * [Datasets](#datasets)
  * [Citation](#citation)

## Requirements
  * [python3](https://www.linuxbabe.com/ubuntu/install-python-3-6-ubuntu-16-04-16-10-17-04) 
  * TensorFlow 1.12 or newer ([install](https://www.tensorflow.org/install/install_linux)). To check TensorFlow version:
> python3 -c 'import tensorflow as tf; print(tf.\_\_version\_\_)'
  * For [creating a new Java dataset](#creating-and-preprocessing-a-new-java-dataset) or [manually examining a trained model](#step-4-manual-examination-of-a-trained-model) (any operation that requires parsing of a new code example): [JDK](https://openjdk.java.net/install/)
  * For creating a C# dataset: [dotnet-core](https://dotnet.microsoft.com/download) version 2.2 or newer.

## Quickstart
### Step 0: Cloning this repository
```
git clone https://github.com/tech-srl/code2seq
cd code2seq
```

### Step 1: Creating a new dataset from Java sources
To obtain a preprocessed dataset to train a network on, you can either download our
preprocessed dataset, or create a new dataset from Java source files.

#### Download our preprocessed dataset Java-large dataset (~16M examples, compressed: 11G, extracted 125GB)
```
mkdir data
cd data
wget https://s3.amazonaws.com/code2seq/datasets/java-large-preprocessed.tar.gz
tar -xvzf java-large-preprocessed.tar.gz
```
This will create a `data/java-large/` sub-directory, containing the files that hold training, test and validation sets,
and a dict file for various dataset properties.

#### Creating and preprocessing a new Java dataset
To create and preprocess a new dataset (for example, to compare code2seq to another model on another dataset):
  * Edit the file [preprocess.sh](preprocess.sh) using the instructions there, pointing it to the correct training, validation and test directories.
  * Run the preprocess.sh file:
> bash preprocess.sh

### Step 2: Training a model
You can either download an already trained model, or train a new model using a preprocessed dataset.

#### Downloading a trained model (137 MB)
We already trained a model for 52 epochs on the data that was preprocessed in the previous step. This model is the same model that was 
 used in the paper and the same model that serves the demo at [code2seq.org](code2seq.org).
```
wget https://s3.amazonaws.com/code2seq/model/java-large/java-large-model.tar.gz
tar -xvzf java-large-model.tar.gz
```

##### Note:
This trained model is in a "released" state, which means that we stripped it from its training parameters and can thus be used for inference, but cannot be further trained. 

#### Training a model from scratch
To train a model from scratch:
  * Edit the file [train.sh](train.sh) to point it to the right preprocessed data. By default, 
  it points to our "java-large" dataset that was preprocessed in the previous step.
  * Before training, you can edit the configuration hyper-parameters in the file [config.py](config.py),
  as explained in [Configuration](#configuration).
  * Run the [train.sh](train.sh) script:
```
bash train.sh
```

### Step 3: Evaluating a trained model
After `config.PATIENCE` iterations of no improvement on the validation set, training stops by itself.

Suppose that iteration #52 is our chosen model, run:
```
python3 code2seq.py --load models/java-large-model/model_iter52.release --test data/java-large/java-large.test.c2s
```
While evaluating, a file named "log.txt" is written to the same dir as the saved models, with each test example name and the model's prediction.

### Step 4: Manual examination of a trained model
To manually examine a trained model, run:
```
python3 code2seq.py --load models/java-large-model/model_iter52.release --predict
```
After the model loads, follow the instructions and edit the file `Input.java` and enter a Java 
method or code snippet, and examine the model's predictions and attention scores.

#### Note: 
Due to TensorFlow's limitations, if using beam search (`config.BEAM_WIDTH > 0`), then `BEAM_WIDTH` hypotheses will be printed, but
without attention weights. If not using beam search (`config.BEAM_WIDTH == 0`), then a single hypothesis will be printed *with 
the attention weights* in every decoding timestep. 

## Configuration
Changing hyper-parameters is possible by editing the file [config.py](config.py).

Here are some of the parameters and their description:
#### config.NUM_EPOCHS = 3000
The max number of epochs to train the model. 
#### config.SAVE_EVERY_EPOCHS = 1
The frequency, in epochs, of saving a model and evaluating on the validation set during training.
#### config.PATIENCE = 10
Controlling early stopping: how many epochs of no improvement should training continue before stopping.  
#### config.BATCH_SIZE = 512
Batch size during training.
#### config.TEST_BATCH_SIZE = 256
Batch size during evaluation. Affects only the evaluation speed and memory consumption, does not affect the results.
#### config.SHUFFLE_BUFFER_SIZE = 10000
The buffer size that the reader uses for shuffling the training data. 
Controls the randomness of the data. 
Increasing this value might hurt training throughput. 
#### config.CSV_BUFFER_SIZE = 100 * 1024 * 1024  
The buffer size (in bytes) of the CSV dataset reader.
#### config.MAX_CONTEXTS = 200
The number of contexts to sample in each example during training 
(resampling a different subset of this size every training iteration).
#### config.SUBTOKENS_VOCAB_MAX_SIZE = 190000
The max size of the subtoken vocabulary.
#### config.TARGET_VOCAB_MAX_SIZE = 27000
The max size of the target words vocabulary.
#### config.EMBEDDINGS_SIZE = 128
Embedding size for subtokens, AST nodes and target symbols.
#### config.RNN_SIZE = 128 * 2 
The total size of the two LSTMs that are used to embed the paths if `config.BIRNN` is `True`, or the size of the single LSTM if `config.BIRNN` is `False`.
#### config.DECODER_SIZE = 320
Size of each LSTM layer in the decoder.
#### config.NUM_DECODER_LAYERS = 1
Number of decoder LSTM layers. Can be increased to support long target sequences.
#### config.MAX_PATH_LENGTH = 8 + 1
The max number of nodes in a path 
#### config.MAX_NAME_PARTS = 5
The max number of subtokens in an input token. If the token is longer, only the first subtokens will be read.
#### config.MAX_TARGET_PARTS = 6
The max number of symbols in the target sequence. 
Set to 6 by default for method names, but can be increased for learning datasets with longer sequences.
### config.BIRNN = True
If True, use a bidirectional LSTM to encode each path. If False, use a unidirectional LSTM only. 
#### config.RANDOM_CONTEXTS = True
When True, sample `MAX_CONTEXT` from every example every training iteration. 
When False, take the first `MAX_CONTEXTS` only.
#### config.BEAM_WIDTH = 0
Beam width in beam search. Inactive when 0. 
#### config.USE_MOMENTUM = True
If `True`, use Momentum optimizer with nesterov. If `False`, use Adam 
(Adam converges in fewer epochs; Momentum leads to slightly better results). 

## Releasing a trained model
If you wish to keep a trained model for inference only (without the ability to continue training it) you can
release the model using:
```
python3 code2seq.py --load models/java-large-model/model_iter52 --release
```
This will save a copy of the trained model with the '.release' suffix.
A "released" model usually takes ~3x less disk space.

## Extending to other languages  
To extend code2seq to other languages other than Java and C#, a new extractor (similar to the [JavaExtractor](JavaExtractor))
should be implemented, and be called by [preprocess.sh](preprocess.sh).
Basically, an extractor should be able to output for each directory containing source files:
  * A single text file, where each row is an example.
  * Each example is a space-delimited list of fields, where:
  1. The first field is the target label, internally delimited by the "|" character (for example: `compare|ignore|case`
  2. Each of the following field are contexts, where each context has three components separated by commas (","). None of these components can include spaces nor commas.
  
  We refer to these three components as a token, a path, and another token, but in general other types of ternary contexts can be considered.  
  
  Each "token" component is a token in the code, split to subtokens using the "|" character.
  
  Each path is a path between two tokens, split to path nodes (or other kinds of building blocks) using the "|" character.
  Example for a context:
  
`my|key,StringExression|MethodCall|Name,get|value`

Here `my|key` and `get|value` are tokens, and `StringExression|MethodCall|Name` is the syntactic path that connects them. 

## Datasets
### Java
To download the Java-small, Java-med and Java-large datasets used in the Code Summarization task as raw `*.java` files, use:

  * [Java-small](https://s3.amazonaws.com/code2seq/datasets/java-small.tar.gz)
  * [Java-med](https://s3.amazonaws.com/code2seq/datasets/java-med.tar.gz)
  * [Java-large](https://s3.amazonaws.com/code2seq/datasets/java-large.tar.gz)
  
To download the preprocessed datasets, use:
  * [Java-small-preprocessed](https://s3.amazonaws.com/code2seq/datasets/java-small-preprocessed.tar.gz)
  * [Java-med-preprocessed](https://s3.amazonaws.com/code2seq/datasets/java-med-preprocessed.tar.gz)
  * [Java-large-preprocessed](https://s3.amazonaws.com/code2seq/datasets/java-large-preprocessed.tar.gz)

### C#
The C# dataset used in the Code Captioning task can be downloaded from the [CodeNN](https://github.com/sriniiyer/codenn/) repository.

## Citation 

[code2seq: Generating Sequences from Structured Representations of Code](https://arxiv.org/pdf/1808.01400)

```
@inproceedings{
    alon2018codeseq,
    title={code2seq: Generating Sequences from Structured Representations of Code},
    author={Uri Alon and Shaked Brody and Omer Levy and Eran Yahav},
    booktitle={International Conference on Learning Representations},
    year={2019},
    url={https://openreview.net/forum?id=H1gKYo09tX},
}
```
