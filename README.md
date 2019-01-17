# code2seq
This is an official implementation of the model described in:

[Uri Alon](http://urialon.cswp.cs.technion.ac.il), [Shaked Brody](http://www.cs.technion.ac.il/people/shakedbr/), [Omer Levy](https://levyomer.wordpress.com) and [Eran Yahav](http://www.cs.technion.ac.il/~yahave/), "code2seq: Generating Sequences from Structured Representations of Code" [[PDF]](https://arxiv.org/pdf/1808.01400)

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
  * [Features](#features)
  * [Extending to other languages](#extending-to-other-languages)
  * [Citation](#citation)

## Requirements
  * [Python3](https://www.linuxbabe.com/ubuntu/install-python-3-6-ubuntu-16-04-16-10-17-04). 
  * TensorFlow version 1.12 or newer ([install](https://www.tensorflow.org/install/install_linux)). To check TensorFlow version:
> python3 -c 'import tensorflow as tf; print(tf.\_\_version\_\_)'
  * For [creating a new dataset](#creating-and-preprocessing-a-new-java-dataset) or [manually examining a trained model](#step-4-manual-examination-of-a-trained-model) (any operation that requires parsing of a new code example) - [Java JDK](https://openjdk.java.net/install/)
  * For creating a C# dataset: dotnet-core version 2.2 or newer.

## Quickstart
### Step 0: Cloning this repository
```
git clone https://github.com/tech-srl/code2seq
cd code2vec
```

### Step 1: Creating a new dataset from Java sources
To have a preprocessed dataset to train a network on, you can either download our
preprocessed dataset, or create a new dataset of your own.

#### Download our preprocessed dataset Java-large dataset (~15M examples, compressed: 6.3G, extracted 32G)
```
wget https://s3.amazonaws.com/code2vec/data/java14m_data.tar.gz
tar -xvzf java14m_data.tar.gz
```
This will create a data/java14m/ sub-directory, containing the files that hold that training, test and validation sets,
and a vocabulary file for various dataset properties.

#### Creating and preprocessing a new Java dataset
In order to create and preprocess a new dataset (for example, to compare code2vec to another model on another dataset):
  * Edit the file [preprocess.sh](preprocess.sh) using the instructions there, pointing it to the correct training, validation and test directories.
  * Run the preprocess.sh file:
> source preprocess.sh

### Step 2: Training a model
You can either download an already-trained model, or train a new model using a preprocessed dataset.

#### Downloading a trained model (1.4G)
We already trained a model for 8 epochs on the data that was preprocessed in the previous step.
The number of epochs was chosen using [early stopping](https://en.wikipedia.org/wiki/Early_stopping), as the version that maximized the F1 score on the validation set.
```
wget https://s3.amazonaws.com/code2vec/model/java14m_model.tar.gz
tar -xvzf java14m_model.tar.gz
```

##### Note:
This trained model is in a "released" state, which means that we stripped it from its training parameters and can thus be used for inference, but cannot be further trained. If you use this trained model in the next steps, use 'saved_model_iter8.release' instead of 'saved_model_iter8' in every command line example that loads the model such as: '--load models/java14_model/saved_model_iter8'. To read how to release a model, see [Releasing the model](#releasing-the-model).

#### Training a model from scratch
To train a model from scratch:
  * Edit the file [train.sh](train.sh) to point it to the right preprocessed data. By default, 
  it points to our "java14m" dataset that was preprocessed in the previous step.
  * Before training, you can edit the configuration hyper-parameters in the file [common.py](common.py),
  as explained in [Configuration](#configuration).
  * Run the [train.sh](train.sh) script:
```
source train.sh
```

##### Notes:
  1. By default, the network is evaluated on the validation set after every training epoch.
  2. The newest 10 versions are kept (older are deleted automatically). This can be changed, but will be more space consuming.
  3. By default, the network is training for 20 epochs.
These settings can be changed by simply editing the file [common.py](common.py).
Training on a Tesla v100 GPU takes about 50 minutes per epoch. 
Training on Tesla K80 takes about 4 hours per epoch.

### Step 3: Evaluating a trained model
Once the score on the validation set stops improving over time, you can stop the training process (by killing it)
and pick the iteration that performed the best on the validation set.
Suppose that iteration #8 is our chosen model, run:
```
python3 code2vec.py --load models/java14_model/saved_model_iter8 --test data/java14m/java14m.test.c2v
```
While evaluating, a file named "log.txt" is written with each test example name and the model's prediction.

### Step 4: Manual examination of a trained model
To manually examine a trained model, run:
```
python3 code2vec.py --load models/java14_model/saved_model_iter8 --predict
```
After the model loads, follow the instructions and edit the file Input.java and enter a Java 
method or code snippet, and examine the model's predictions and attention scores.

## Configuration
Changing hyper-parameters is possible by editing the file [config.py](config
.py).

Here are some of the parameters and their description:
#### config.NUM_EPOCHS = 20
The max number of epochs to train the model. 
#### config.SAVE_EVERY_EPOCHS = 1
After how many training iterations a model should be saved and evaluated.
#### config.PATIENCE = 10
Controlling early stopping: how many epochs of no improvement should training continue before stopping  
#### config.BATCH_SIZE = 512
Batch size in training.
#### config.TEST_BATCH_SIZE = 256
Batch size in evaluating. Affects only the evaluation speed and memory consumption, does not affect the results.
#### config.SHUFFLE_BUFFER_SIZE = 10000
The buffer size that the reader uses for shuffling the training data. 
Controls the randomness of the data. 
Increasing this value hurts training throughput 
#### config.CSV_BUFFER_SIZE = 100 * 1024 * 1024  
The buffer size (in bytes) of the CSV dataset reader.
#### config.MAX_CONTEXTS = 200
The number of contexts to sample in each example during training 
(resampling a different subset every training iteration).
#### config.SUBTOKENS_VOCAB_MAX_SIZE = 190000
The max size of the subtoken vocabulary.
#### config.TARGET_VOCAB_MAX_SIZE = 27000
The max size of the target words vocabulary.
#### config.EMBEDDINGS_SIZE = 128
Embedding size for subtokens, AST nodes and target symbols.
#### config.RNN_SIZE = 128 * 2 
The total size of the two LSTMs that are used to embed the paths, or the single LSTM if `config.BIRNN` is `False`.
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
If True, use Momentum optimizer with nesterov. If False, use Adam 
(Adam converges in fewer epochs, Momentum leads to slightly better results). 
## Features
Code2vec supports the following features: 

### Releasing the model
If you wish to keep a trained model for inference only (without the ability to continue training it) you can
release the model using:
```
python3 code2vec.py --load models/java14_model/saved_model_iter8 --release
```
This will save a copy of the trained model with the '.release' suffix.
A "released" model usually takes 3x less disk space.

## Extending to other languages  
To extend code2vec to other languages other than Java and C#, a new extractor (similar to the [JavaExtractor](JavaExtractor))
should be implemented, and be called by [preprocess.sh](preprocess.sh).
Basically, an extractor should be able to output for each directory containing source files:
  * A single text file, where each row is an example.
  * Each example is a space-delimited list of fields, where:
  1. The first field is the target label, internally delimited by the "|" character.
  2. Each of the following field are contexts, where each context has three components separated by commas (","). None of these components can include spaces nor commas.
  We refer to these three components as a token, a path, and another token, but in general other types of ternary contexts can be considered.  
  Each "token" component is a token in the code, split to subtokens using the "|" character.
  Each path is a path between two tokens, split to path nodes (or other kinds of building blocks) using the "|" character.
  Example for a context:
`my|key,StringExression|MethodCall|Name,get|value`
Here `my|key` and `get|value` are tokens, and `StringExression|MethodCall|Name` is the syntactic path that connects them. 

## Citation

[code2vec: Learning Distributed Representations of Code](https://arxiv.org/pdf/1803.09473)

```
@article{alon2018code2vec,
  title={code2vec: Learning Distributed Representations of Code},
  author={Alon, Uri and Zilberstein, Meital and Levy, Omer and Yahav, Eran},
  journal={arXiv preprint arXiv:1803.09473},
  year={2018}
}
```
