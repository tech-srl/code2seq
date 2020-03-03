import _pickle as pickle
import time

import numpy as np
import tensorflow as tf

import reader
from model import Model
from common import Common


class ModelRunner:
    num_batches_to_log = 100

    def __init__(self, config, is_training):
        self.config = config

        if config.LOAD_PATH:
            self.load_model()
        else:
            with open('{}.dict.c2s'.format(config.TRAIN_PATH), 'rb') as file:
                subtoken_to_count = pickle.load(file)
                node_to_count = pickle.load(file)
                target_to_count = pickle.load(file)
                max_contexts = pickle.load(file)
                self.num_training_examples = pickle.load(file)
                print('Dictionaries loaded.')

            if self.config.DATA_NUM_CONTEXTS <= 0:
                self.config.DATA_NUM_CONTEXTS = max_contexts
            self.subtoken_to_index, self.index_to_subtoken, self.subtoken_vocab_size = \
                Common.load_vocab_from_dict(subtoken_to_count, add_values=[Common.PAD, Common.UNK],
                                            max_size=config.SUBTOKENS_VOCAB_MAX_SIZE)
            print('Loaded subtoken vocab. size: %d' % self.subtoken_vocab_size)

            self.target_to_index, self.index_to_target, self.target_vocab_size = \
                Common.load_vocab_from_dict(target_to_count, add_values=[Common.PAD, Common.UNK, Common.SOS],
                                            max_size=config.TARGET_VOCAB_MAX_SIZE)
            print('Loaded target word vocab. size: %d' % self.target_vocab_size)

            self.node_to_index, self.index_to_node, self.nodes_vocab_size = \
                Common.load_vocab_from_dict(node_to_count, add_values=[Common.PAD, Common.UNK], max_size=None)
            print('Loaded nodes vocab. size: %d' % self.nodes_vocab_size)

        self.dataset_reader = reader.Reader(subtoken_to_index=self.subtoken_to_index,
                                            node_to_index=self.node_to_index,
                                            target_to_index=self.target_to_index,
                                            config=self.config,
                                            is_evaluating=(not is_training))

        self.model = Model(self.config, self.subtoken_vocab_size, self.target_vocab_size, self.nodes_vocab_size,
                           is_training)

    def train(self):
        print('Starting training')

        self.print_hyperparams()
        print('Number of trainable params:',
              np.sum([np.prod(v.get_shape().as_list()) for v in self.model.trainable_variables]))

        if self.config.LOAD_PATH:
            print('Loading model...')
            self.load_model()

        print('Start training loop...')
        dataset_iterator = self.dataset_reader.get_output()

        if self.config.USE_MOMENTUM:
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=0.01,
                decay_steps=self.num_training_examples,
                decay_rate=0.95
            )
            optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.95, nesterov=True)

        else:
            optimizer = tf.keras.optimizers.Adam()

        sum_loss = 0
        batch_num = 0
        multi_batch_start_time = time.time()
        start_time = time.time()
        epochs_trained = 0
        for iteration in range(self.config.NUM_EPOCHS):
            self.dataset_reader.reset()
            for input_tensors in dataset_iterator:
                target_lengths = input_tensors[reader.TARGET_LENGTH_KEY]
                target_index = input_tensors[reader.TARGET_INDEX_KEY]
                batch_size = tf.shape(target_index)[0]
                with tf.GradientTape() as tape:
                    outputs, _ = self.model(input_tensors)
                    logits = outputs.rnn_output  # (batch, max_output_length, dim * 2 + rnn_size)
                    crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_index, logits=logits)
                    target_words_nonzero = tf.sequence_mask(target_lengths + 1,
                                                            maxlen=self.config.MAX_TARGET_PARTS + 1, dtype=tf.float32)
                    loss = tf.reduce_sum(crossent * target_words_nonzero) / tf.to_float(batch_size)

                gradients = tape.gradient(loss, self.model.trainable_variables)
                if self.config.USE_MOMENTUM:
                    clipped_gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=5)
                optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

                sum_loss += loss
                batch_num += 1

                if batch_num % self.num_batches_to_log == 0:
                    self.trace(sum_loss, batch_num, multi_batch_start_time)
                    sum_loss = 0
                    multi_batch_start_time = time.time()

            # the end of an epoch
            epochs_trained += 1
            print('Finished {0} epochs'.format(epochs_trained))
            if epochs_trained % self.config.SAVE_EVERY_EPOCHS == 0:
                if self.config.SAVE_PATH:
                    self.save_model(self.config.SAVE_PATH)
            # TODO: add evaluation

        # the end of training
        if self.config.SAVE_PATH:
            self.save_model(self.config.SAVE_PATH + '.final')
            print('Model saved in file: %s' % self.config.SAVE_PATH)

        elapsed = int(time.time() - start_time)
        print("Training time: %sh%sm%ss\n" % ((elapsed // 60 // 60), (elapsed // 60) % 60, elapsed % 60))

    def trace(self, sum_loss, batch_num, multi_batch_start_time):
        multi_batch_elapsed = time.time() - multi_batch_start_time
        avg_loss = sum_loss / self.num_batches_to_log
        print('Average loss at batch %d: %f, \tthroughput: %d samples/sec' % (batch_num, avg_loss,
                                                                              self.config.BATCH_SIZE * self.num_batches_to_log / (
                                                                                  multi_batch_elapsed if multi_batch_elapsed > 0 else 1)))

    def print_hyperparams(self):
        print('Training batch size:\t\t\t', self.config.BATCH_SIZE)
        print('Dataset path:\t\t\t\t', self.config.TRAIN_PATH)
        print('Training file path:\t\t\t', self.config.TRAIN_PATH + '.train.c2s')
        print('Validation path:\t\t\t', self.config.TEST_PATH)
        print('Taking max contexts from each example:\t', self.config.MAX_CONTEXTS)
        print('Random path sampling:\t\t\t', self.config.RANDOM_CONTEXTS)
        print('Embedding size:\t\t\t\t', self.config.EMBEDDINGS_SIZE)
        if self.config.BIRNN:
            print('Using BiLSTMs, each of size:\t\t', self.config.RNN_SIZE // 2)
        else:
            print('Uni-directional LSTM of size:\t\t', self.config.RNN_SIZE)
        print('Decoder size:\t\t\t\t', self.config.DECODER_SIZE)
        print('Decoder layers:\t\t\t\t', self.config.NUM_DECODER_LAYERS)
        print('Max path lengths:\t\t\t', self.config.MAX_PATH_LENGTH)
        print('Max subtokens in a token:\t\t', self.config.MAX_NAME_PARTS)
        print('Max target length:\t\t\t', self.config.MAX_TARGET_PARTS)
        print('Embeddings dropout keep_prob:\t\t', self.config.EMBEDDINGS_DROPOUT_KEEP_PROB)
        print('LSTM dropout keep_prob:\t\t\t', self.config.RNN_DROPOUT_KEEP_PROB)
        print('============================================')

    def save_model(self, path):
        pass

    def load_model(self):
        pass
