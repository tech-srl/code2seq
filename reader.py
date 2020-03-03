import os
import _pickle as pickle
import tensorflow as tf
import numpy as np
from common import Common
from config import Config
from argparse import ArgumentParser

TARGET_INDEX_KEY = 'TARGET_INDEX_KEY'
TARGET_STRING_KEY = 'TARGET_STRING_KEY'
TARGET_LENGTH_KEY = 'TARGET_LENGTH_KEY'
PATH_SOURCE_INDICES_KEY = 'PATH_SOURCE_INDICES_KEY'
NODE_INDICES_KEY = 'NODES_INDICES_KEY'
PATH_TARGET_INDICES_KEY = 'PATH_TARGET_INDICES_KEY'
VALID_CONTEXT_MASK_KEY = 'VALID_CONTEXT_MASK_KEY'
PATH_SOURCE_LENGTHS_KEY = 'PATH_SOURCE_LENGTHS_KEY'
PATH_LENGTHS_KEY = 'PATH_LENGTHS_KEY'
PATH_TARGET_LENGTHS_KEY = 'PATH_TARGET_LENGTHS_KEY'
PATH_SOURCE_STRINGS_KEY = 'PATH_SOURCE_STRINGS_KEY'
PATH_STRINGS_KEY = 'PATH_STRINGS_KEY'
PATH_TARGET_STRINGS_KEY = 'PATH_TARGET_STRINGS_KEY'


class Reader:
    class_subtoken_table = None
    class_target_table = None
    class_node_table = None

    def __init__(self, subtoken_to_index, target_to_index, node_to_index, config, is_evaluating=False, is_debug=False):
        self.config = config
        self.file_path = config.TEST_PATH if is_evaluating else (config.TRAIN_PATH + '.train.c2s')
        if self.file_path is not None and not os.path.exists(self.file_path):
            print(
                '%s cannot find file: %s' % ('Evaluation reader' if is_evaluating else 'Train reader', self.file_path))
        self.batch_size = config.TEST_BATCH_SIZE if is_evaluating else config.BATCH_SIZE
        self.is_evaluating = is_evaluating

        self.context_pad = '{},{},{}'.format(Common.PAD, Common.PAD, Common.PAD)
        self.record_defaults = [[self.context_pad]] * (self.config.DATA_NUM_CONTEXTS + 1)

        self.subtoken_table = Reader.get_subtoken_table(subtoken_to_index)
        self.target_table = Reader.get_target_table(target_to_index)
        self.node_table = Reader.get_node_table(node_to_index)
        if self.file_path is not None and not is_debug:
            self.output_tensors = self.compute_output()

    @classmethod
    def get_subtoken_table(cls, subtoken_to_index):
        if cls.class_subtoken_table is None:
            cls.class_subtoken_table = cls.initialize_hash_map(subtoken_to_index, subtoken_to_index[Common.UNK])
        return cls.class_subtoken_table

    @classmethod
    def get_target_table(cls, target_to_index):
        if cls.class_target_table is None:
            cls.class_target_table = cls.initialize_hash_map(target_to_index, target_to_index[Common.UNK])
        return cls.class_target_table

    @classmethod
    def get_node_table(cls, node_to_index):
        if cls.class_node_table is None:
            cls.class_node_table = cls.initialize_hash_map(node_to_index, node_to_index[Common.UNK])
        return cls.class_node_table

    @classmethod
    def initialize_hash_map(cls, word_to_index, default_value):
        return tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(list(word_to_index.keys()), list(word_to_index.values()),
                                                key_dtype=tf.string,
                                                value_dtype=tf.int32), default_value)

    def process_from_placeholder(self, row):
        parts = tf.io.decode_csv(row, record_defaults=self.record_defaults, field_delim=' ', use_quote_delim=False)
        return self.process_dataset(*parts)

    def process_dataset(self, *row_parts):
        row_parts = list(row_parts)
        word = row_parts[0]  # (, )

        if not self.is_evaluating and self.config.RANDOM_CONTEXTS:
            all_contexts = tf.stack(row_parts[1:])
            all_contexts_padded = tf.concat([all_contexts, [self.context_pad]], axis=-1)
            index_of_blank_context = tf.where(tf.equal(all_contexts_padded, self.context_pad))
            num_contexts_per_example = tf.reduce_min(index_of_blank_context)

            # if there are less than self.max_contexts valid contexts, still sample self.max_contexts
            safe_limit = tf.cast(tf.maximum(num_contexts_per_example, self.config.MAX_CONTEXTS), tf.int32)
            rand_indices = tf.random.shuffle(tf.range(safe_limit))[:self.config.MAX_CONTEXTS]
            contexts = tf.gather(all_contexts, rand_indices)  # (max_contexts,)
        else:
            contexts = row_parts[1:(self.config.MAX_CONTEXTS + 1)]  # (max_contexts,)

        # contexts: (max_contexts, )
        split_contexts = tf.strings.split(contexts, sep=',')
        sparse_split_contexts = split_contexts.to_sparse()

        dense_split_contexts = tf.reshape(
            tf.sparse.to_dense(sp_input=sparse_split_contexts, default_value=Common.PAD),
            shape=[self.config.MAX_CONTEXTS, 3])  # (batch, max_contexts, 3)

        split_target_labels = tf.strings.split(tf.expand_dims(word, -1), sep='|')
        sparse_target_labels = split_target_labels.to_sparse()
        sparse_target_labels = tf.sparse.reset_shape(sparse_target_labels,
                                                     [1, tf.maximum(tf.cast(self.config.MAX_TARGET_PARTS, tf.int64),
                                                                    sparse_target_labels.dense_shape[1] + 1)])
        dense_target_label = tf.reshape(tf.sparse.to_dense(sp_input=sparse_target_labels,
                                                           default_value=Common.PAD),
                                        shape=[-1])
        index_of_blank = tf.where(tf.equal(dense_target_label, Common.PAD))
        target_length = tf.reduce_min(index_of_blank)
        dense_target_label = dense_target_label[:self.config.MAX_TARGET_PARTS]
        clipped_target_lengths = tf.clip_by_value(target_length, clip_value_min=0,
                                                  clip_value_max=self.config.MAX_TARGET_PARTS)
        target_word_labels = tf.concat([
            self.target_table.lookup(dense_target_label), [0]], axis=-1)  # (max_target_parts + 1) of int

        path_source_strings = tf.slice(dense_split_contexts, [0, 0], [self.config.MAX_CONTEXTS, 1])  # (max_contexts, 1)
        flat_source_strings = tf.reshape(path_source_strings, [-1])  # (max_contexts)
        split_source = tf.strings.split(flat_source_strings, sep='|')  # (max_contexts, max_name_parts)

        sparse_split_source = split_source.to_sparse()
        sparse_split_source = tf.sparse.reset_shape(sparse_split_source,
                                                    [self.config.MAX_CONTEXTS,
                                                     tf.maximum(
                                                         tf.cast(self.config.MAX_NAME_PARTS, tf.int64),
                                                         sparse_split_source.dense_shape[1])])

        dense_split_source = tf.sparse.to_dense(sp_input=sparse_split_source,
                                                default_value=Common.PAD)  # (max_contexts, max_name_parts)
        dense_split_source = tf.slice(dense_split_source, [0, 0], [-1, self.config.MAX_NAME_PARTS])
        path_source_indices = self.subtoken_table.lookup(dense_split_source)  # (max_contexts, max_name_parts)
        path_source_lengths = tf.reduce_sum(tf.cast(tf.not_equal(dense_split_source, Common.PAD), tf.int32),
                                            -1)  # (max_contexts)

        path_strings = tf.slice(dense_split_contexts, [0, 1], [self.config.MAX_CONTEXTS, 1])
        flat_path_strings = tf.reshape(path_strings, [-1])
        split_path = tf.strings.split(flat_path_strings, sep='|')
        sparse_split_path = split_path.to_sparse()
        sparse_split_path = tf.sparse.reset_shape(sparse_split_path,
                                                  [self.config.MAX_CONTEXTS, self.config.MAX_PATH_LENGTH])
        dense_split_path = tf.sparse.to_dense(sp_input=sparse_split_path,
                                              default_value=Common.PAD)  # (batch, max_contexts, max_path_length)

        node_indices = self.node_table.lookup(dense_split_path)  # (max_contexts, max_path_length)
        path_lengths = tf.reduce_sum(tf.cast(tf.not_equal(dense_split_path, Common.PAD), tf.int32),
                                     -1)  # (max_contexts)

        path_target_strings = tf.slice(dense_split_contexts, [0, 2], [self.config.MAX_CONTEXTS, 1])  # (max_contexts, 1)
        flat_target_strings = tf.reshape(path_target_strings, [-1])  # (max_contexts)
        split_target = tf.strings.split(flat_target_strings, sep='|')  # (max_contexts, max_name_parts)
        sparse_split_target = split_target.to_sparse()
        sparse_split_target = tf.sparse.reset_shape(sparse_split_target, [self.config.MAX_CONTEXTS,
                                                                          tf.maximum(
                                                                              tf.cast(self.config.MAX_NAME_PARTS,
                                                                                      tf.int64),
                                                                              sparse_split_target.dense_shape[1])])
        dense_split_target = tf.sparse.to_dense(sp_input=sparse_split_target,
                                                default_value=Common.PAD)  # (max_contexts, max_name_parts)
        dense_split_target = tf.slice(dense_split_target, [0, 0], [-1, self.config.MAX_NAME_PARTS])
        path_target_indices = self.subtoken_table.lookup(dense_split_target)  # (max_contexts, max_name_parts)
        path_target_lengths = tf.reduce_sum(tf.cast(tf.not_equal(dense_split_target, Common.PAD), tf.int32),
                                            -1)  # (max_contexts)

        valid_contexts_mask = tf.cast(tf.not_equal(
            tf.reduce_max(path_source_indices, -1) + tf.reduce_max(node_indices, -1) + tf.reduce_max(
                path_target_indices, -1), 0), tf.float32)

        return {TARGET_STRING_KEY: word, TARGET_INDEX_KEY: target_word_labels,
                TARGET_LENGTH_KEY: clipped_target_lengths,
                PATH_SOURCE_INDICES_KEY: path_source_indices, NODE_INDICES_KEY: node_indices,
                PATH_TARGET_INDICES_KEY: path_target_indices, VALID_CONTEXT_MASK_KEY: valid_contexts_mask,
                PATH_SOURCE_LENGTHS_KEY: path_source_lengths, PATH_LENGTHS_KEY: path_lengths,
                PATH_TARGET_LENGTHS_KEY: path_target_lengths, PATH_SOURCE_STRINGS_KEY: path_source_strings,
                PATH_STRINGS_KEY: path_strings, PATH_TARGET_STRINGS_KEY: path_target_strings
                }

    def reset(self):
        self.reset_op()

    def get_output(self):
        return self.output_tensors

    def compute_output(self):
        dataset = tf.data.experimental.CsvDataset(self.file_path, record_defaults=self.record_defaults, field_delim=' ',
                                                  use_quote_delim=False, buffer_size=self.config.CSV_BUFFER_SIZE)

        if not self.is_evaluating:
            if self.config.SAVE_EVERY_EPOCHS > 1:
                dataset = dataset.repeat(self.config.SAVE_EVERY_EPOCHS)
            dataset = dataset.shuffle(self.config.SHUFFLE_BUFFER_SIZE, reshuffle_each_iteration=True)

        dataset = dataset.map(map_func=self.process_dataset,
                              num_parallel_calls=self.config.READER_NUM_PARALLEL_BATCHES).batch(
            batch_size=self.batch_size)

        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        self.iterator = iter(dataset)
        self.reset_op = dataset.repeat
        return self.iterator


if __name__ == '__main__':

    tf.config.experimental_run_functions_eagerly(True)

    print("tf executing eagerly: " + str(tf.executing_eagerly()))

    parser = ArgumentParser()
    parser.add_argument("-d", "--data", dest="data_path",
                        help="path to preprocessed dataset", required=False)
    parser.add_argument("-te", "--test", dest="test_path",
                        help="path to test file", metavar="FILE", required=False)
    parser.add_argument("-s", "--save_prefix", dest="save_path_prefix",
                        help="path to save file", metavar="FILE", required=False)
    parser.add_argument("-l", "--load", dest="load_path",
                        help="path to saved file", metavar="FILE", required=False)
    parser.add_argument('--release', action='store_true',
                        help='if specified and loading a trained model, release the loaded model for a smaller model '
                             'size.')
    args = parser.parse_args()
    config = Config.get_default_config(args)
    config.DATA_NUM_CONTEXTS

    with open('{}.dict.c2s'.format(config.TRAIN_PATH), 'rb') as file:
        subtoken_to_count = pickle.load(file)
        node_to_count = pickle.load(file)
        target_to_count = pickle.load(file)
        max_contexts = pickle.load(file)
        num_training_examples = pickle.load(file)
        print('Dictionaries loaded.')

        if config.DATA_NUM_CONTEXTS <= 0:
            config.DATA_NUM_CONTEXTS = max_contexts
        subtoken_to_index, index_to_subtoken, subtoken_vocab_size = \
            Common.load_vocab_from_dict(subtoken_to_count, add_values=[Common.PAD, Common.UNK],
                                        max_size=config.SUBTOKENS_VOCAB_MAX_SIZE)
        print('Loaded subtoken vocab. size: %d' % subtoken_vocab_size)

        target_to_index, index_to_target, target_vocab_size = \
            Common.load_vocab_from_dict(target_to_count, add_values=[Common.PAD, Common.UNK, Common.SOS],
                                        max_size=config.TARGET_VOCAB_MAX_SIZE)
        print('Loaded target word vocab. size: %d' % target_vocab_size)

        node_to_index, index_to_node, nodes_vocab_size = \
            Common.load_vocab_from_dict(node_to_count, add_values=[Common.PAD, Common.UNK], max_size=None)
        print('Loaded nodes vocab. size: %d' % nodes_vocab_size)

        is_debug = False
        reader = Reader(subtoken_to_index, target_to_index, node_to_index, config, False, is_debug)

        if not is_debug:
            dataset_iterator = reader.get_output()
        else:
            file_path = '{}.train.c2s'.format(config.TRAIN_PATH)
            context_pad = '{},{},{}'.format(Common.PAD, Common.PAD, Common.PAD)
            record_defaults = [[context_pad]] * (config.DATA_NUM_CONTEXTS + 1)
            dataset = tf.data.experimental.CsvDataset(file_path, record_defaults=record_defaults, field_delim=' ',
                                                      use_quote_delim=False, buffer_size=config.CSV_BUFFER_SIZE)

            dataset = dataset.map(map_func=reader.process_dataset).batch(batch_size=16)
            dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
            dataset_iterator = iter(dataset)

        # row = next(dataset_iterator)
        # output = reader.process_dataset(*row)

        try:
            for output in dataset_iterator:
                target_indices = output[TARGET_INDEX_KEY].numpy()
                target_strings = output[TARGET_STRING_KEY].numpy()
                target_lengths = output[TARGET_LENGTH_KEY].numpy()
                path_source_indices = output[PATH_SOURCE_INDICES_KEY].numpy()
                node_indices = output[NODE_INDICES_KEY].numpy()
                path_target_indices = output[PATH_TARGET_INDICES_KEY].numpy()
                valid_context_mask = output[VALID_CONTEXT_MASK_KEY].numpy()
                path_source_lengths = output[PATH_SOURCE_LENGTHS_KEY].numpy()
                path_lengths = output[PATH_LENGTHS_KEY].numpy()
                path_target_lengths = output[PATH_TARGET_LENGTHS_KEY].numpy()
                path_source_strings = output[PATH_SOURCE_STRINGS_KEY].numpy()
                path_strings = output[PATH_STRINGS_KEY].numpy()
                path_target_strings = output[PATH_TARGET_STRINGS_KEY].numpy()

                print('Target strings: ', Common.binary_to_string_list(target_strings))
                print('Context strings: ', Common.binary_to_string_3d(
                    np.concatenate([path_source_strings, path_strings, path_target_strings], -1)))
                print('Target indices: ', target_indices)
                print('Target lengths: ', target_lengths)
                print('Path source strings: ', Common.binary_to_string_3d(path_source_strings))
                print('Path source indices: ', path_source_indices)
                print('Path source lengths: ', path_source_lengths)
                print('Path strings: ', Common.binary_to_string_3d(path_strings))
                print('Node indices: ', node_indices)
                print('Path lengths: ', path_lengths)
                print('Path target strings: ', Common.binary_to_string_3d(path_target_strings))
                print('Path target indices: ', path_target_indices)
                print('Path target lengths: ', path_target_lengths)
                print('Valid context mask: ', valid_context_mask)

        except tf.errors.OutOfRangeError:
            print('Done training, epoch reached')
