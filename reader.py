import os

import tensorflow as tf

from common import Common

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

    def __init__(self, subtoken_to_index, target_to_index, node_to_index, config, is_evaluating=False):
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
        if self.file_path is not None:
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
        return tf.contrib.lookup.HashTable(
            tf.contrib.lookup.KeyValueTensorInitializer(list(word_to_index.keys()), list(word_to_index.values()),
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
            rand_indices = tf.random_shuffle(tf.range(safe_limit))[:self.config.MAX_CONTEXTS]
            contexts = tf.gather(all_contexts, rand_indices)  # (max_contexts,)
        else:
            contexts = row_parts[1:(self.config.MAX_CONTEXTS + 1)]  # (max_contexts,)

        # contexts: (max_contexts, )
        split_contexts = tf.string_split(contexts, delimiter=',', skip_empty=False)
        sparse_split_contexts = tf.sparse.SparseTensor(indices=split_contexts.indices,
                                                       values=split_contexts.values,
                                                       dense_shape=[self.config.MAX_CONTEXTS, 3])
        dense_split_contexts = tf.reshape(
            tf.sparse.to_dense(sp_input=sparse_split_contexts, default_value=Common.PAD),
            shape=[self.config.MAX_CONTEXTS, 3])  # (batch, max_contexts, 3)

        split_target_labels = tf.string_split(tf.expand_dims(word, -1), delimiter='|')
        target_dense_shape = [1, tf.maximum(tf.to_int64(self.config.MAX_TARGET_PARTS),
                                            split_target_labels.dense_shape[1] + 1)]
        sparse_target_labels = tf.sparse.SparseTensor(indices=split_target_labels.indices,
                                                      values=split_target_labels.values,
                                                      dense_shape=target_dense_shape)
        dense_target_label = tf.reshape(tf.sparse.to_dense(sp_input=sparse_target_labels,
                                                           default_value=Common.PAD), [-1])
        index_of_blank = tf.where(tf.equal(dense_target_label, Common.PAD))
        target_length = tf.reduce_min(index_of_blank)
        dense_target_label = dense_target_label[:self.config.MAX_TARGET_PARTS]
        clipped_target_lengths = tf.clip_by_value(target_length, clip_value_min=0,
                                                  clip_value_max=self.config.MAX_TARGET_PARTS)
        target_word_labels = tf.concat([
            self.target_table.lookup(dense_target_label), [0]], axis=-1)  # (max_target_parts + 1) of int

        path_source_strings = tf.slice(dense_split_contexts, [0, 0], [self.config.MAX_CONTEXTS, 1])  # (max_contexts, 1)
        flat_source_strings = tf.reshape(path_source_strings, [-1])  # (max_contexts)
        split_source = tf.string_split(flat_source_strings, delimiter='|',
                                       skip_empty=False)  # (max_contexts, max_name_parts)

        sparse_split_source = tf.sparse.SparseTensor(indices=split_source.indices, values=split_source.values,
                                                     dense_shape=[self.config.MAX_CONTEXTS,
                                                                  tf.maximum(tf.to_int64(self.config.MAX_NAME_PARTS),
                                                                             split_source.dense_shape[1])])
        dense_split_source = tf.sparse.to_dense(sp_input=sparse_split_source,
                                                default_value=Common.PAD)  # (max_contexts, max_name_parts)
        dense_split_source = tf.slice(dense_split_source, [0, 0], [-1, self.config.MAX_NAME_PARTS])
        path_source_indices = self.subtoken_table.lookup(dense_split_source)  # (max_contexts, max_name_parts)
        path_source_lengths = tf.reduce_sum(tf.cast(tf.not_equal(dense_split_source, Common.PAD), tf.int32),
                                            -1)  # (max_contexts)

        path_strings = tf.slice(dense_split_contexts, [0, 1], [self.config.MAX_CONTEXTS, 1])
        flat_path_strings = tf.reshape(path_strings, [-1])
        split_path = tf.string_split(flat_path_strings, delimiter='|', skip_empty=False)
        sparse_split_path = tf.sparse.SparseTensor(indices=split_path.indices, values=split_path.values,
                                                   dense_shape=[self.config.MAX_CONTEXTS, self.config.MAX_PATH_LENGTH])
        dense_split_path = tf.sparse.to_dense(sp_input=sparse_split_path,
                                              default_value=Common.PAD)  # (batch, max_contexts, max_path_length)

        node_indices = self.node_table.lookup(dense_split_path)  # (max_contexts, max_path_length)
        path_lengths = tf.reduce_sum(tf.cast(tf.not_equal(dense_split_path, Common.PAD), tf.int32),
                                     -1)  # (max_contexts)

        path_target_strings = tf.slice(dense_split_contexts, [0, 2], [self.config.MAX_CONTEXTS, 1])  # (max_contexts, 1)
        flat_target_strings = tf.reshape(path_target_strings, [-1])  # (max_contexts)
        split_target = tf.string_split(flat_target_strings, delimiter='|',
                                       skip_empty=False)  # (max_contexts, max_name_parts)
        sparse_split_target = tf.sparse.SparseTensor(indices=split_target.indices, values=split_target.values,
                                                     dense_shape=[self.config.MAX_CONTEXTS,
                                                                  tf.maximum(tf.to_int64(self.config.MAX_NAME_PARTS),
                                                                             split_target.dense_shape[1])])
        dense_split_target = tf.sparse.to_dense(sp_input=sparse_split_target,
                                                default_value=Common.PAD)  # (max_contexts, max_name_parts)
        dense_split_target = tf.slice(dense_split_target, [0, 0], [-1, self.config.MAX_NAME_PARTS])
        path_target_indices = self.subtoken_table.lookup(dense_split_target)  # (max_contexts, max_name_parts)
        path_target_lengths = tf.reduce_sum(tf.cast(tf.not_equal(dense_split_target, Common.PAD), tf.int32),
                                            -1)  # (max_contexts)

        valid_contexts_mask = tf.to_float(tf.not_equal(
            tf.reduce_max(path_source_indices, -1) + tf.reduce_max(node_indices, -1) + tf.reduce_max(
                path_target_indices, -1), 0))

        return {TARGET_STRING_KEY: word, TARGET_INDEX_KEY: target_word_labels,
                TARGET_LENGTH_KEY: clipped_target_lengths,
                PATH_SOURCE_INDICES_KEY: path_source_indices, NODE_INDICES_KEY: node_indices,
                PATH_TARGET_INDICES_KEY: path_target_indices, VALID_CONTEXT_MASK_KEY: valid_contexts_mask,
                PATH_SOURCE_LENGTHS_KEY: path_source_lengths, PATH_LENGTHS_KEY: path_lengths,
                PATH_TARGET_LENGTHS_KEY: path_target_lengths, PATH_SOURCE_STRINGS_KEY: path_source_strings,
                PATH_STRINGS_KEY: path_strings, PATH_TARGET_STRINGS_KEY: path_target_strings
                }

    def reset(self, sess):
        sess.run(self.reset_op)

    def get_output(self):
        return self.output_tensors

    def compute_output(self):
        dataset = tf.data.experimental.CsvDataset(self.file_path, record_defaults=self.record_defaults, field_delim=' ',
                                                  use_quote_delim=False, buffer_size=self.config.CSV_BUFFER_SIZE)

        if not self.is_evaluating:
            if self.config.SAVE_EVERY_EPOCHS > 1:
                dataset = dataset.repeat(self.config.SAVE_EVERY_EPOCHS)
            dataset = dataset.shuffle(self.config.SHUFFLE_BUFFER_SIZE, reshuffle_each_iteration=True)
        dataset = dataset.apply(tf.data.experimental.map_and_batch(
            map_func=self.process_dataset, batch_size=self.batch_size,
            num_parallel_batches=self.config.READER_NUM_PARALLEL_BATCHES))
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
        self.iterator = dataset.make_initializable_iterator()
        self.reset_op = self.iterator.initializer
        return self.iterator.get_next()


if __name__ == '__main__':
    target_word_to_index = {Common.PAD: 0, Common.UNK: 1, Common.SOS: 2,
                            'a': 3, 'b': 4, 'c': 5, 'd': 6, 't': 7}
    subtoken_to_index = {Common.PAD: 0, Common.UNK: 1, 'a': 2, 'b': 3, 'c': 4, 'd': 5}
    node_to_index = {Common.PAD: 0, Common.UNK: 1, '1': 2, '2': 3, '3': 4, '4': 5}
    import numpy as np


    class Config:
        def __init__(self):
            self.SAVE_EVERY_EPOCHS = 1
            self.TRAIN_PATH = self.TEST_PATH = 'test_input/test_input'
            self.BATCH_SIZE = 2
            self.TEST_BATCH_SIZE = self.BATCH_SIZE
            self.READER_NUM_PARALLEL_BATCHES = 1
            self.READING_BATCH_SIZE = 2
            self.SHUFFLE_BUFFER_SIZE = 100
            self.MAX_CONTEXTS = 4
            self.DATA_NUM_CONTEXTS = 4
            self.MAX_PATH_LENGTH = 3
            self.MAX_NAME_PARTS = 2
            self.MAX_TARGET_PARTS = 4
            self.RANDOM_CONTEXTS = True
            self.CSV_BUFFER_SIZE = None


    config = Config()
    reader = Reader(subtoken_to_index, target_word_to_index, node_to_index, config, False)

    output = reader.get_output()
    target_index_op = output[TARGET_INDEX_KEY]
    target_string_op = output[TARGET_STRING_KEY]
    target_length_op = output[TARGET_LENGTH_KEY]
    path_source_indices_op = output[PATH_SOURCE_INDICES_KEY]
    node_indices_op = output[NODE_INDICES_KEY]
    path_target_indices_op = output[PATH_TARGET_INDICES_KEY]
    valid_context_mask_op = output[VALID_CONTEXT_MASK_KEY]
    path_source_lengths_op = output[PATH_SOURCE_LENGTHS_KEY]
    path_lengths_op = output[PATH_LENGTHS_KEY]
    path_target_lengths_op = output[PATH_TARGET_LENGTHS_KEY]
    path_source_strings_op = output[PATH_SOURCE_STRINGS_KEY]
    path_strings_op = output[PATH_STRINGS_KEY]
    path_target_strings_op = output[PATH_TARGET_STRINGS_KEY]

    sess = tf.InteractiveSession()
    tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()).run()
    reader.reset(sess)

    try:
        while True:
            target_indices, target_strings, target_lengths, path_source_indices, \
            node_indices, path_target_indices, valid_context_mask, path_source_lengths, \
            path_lengths, path_target_lengths, path_source_strings, path_strings, \
            path_target_strings = sess.run(
                [target_index_op, target_string_op, target_length_op, path_source_indices_op,
                 node_indices_op, path_target_indices_op, valid_context_mask_op, path_source_lengths_op,
                 path_lengths_op, path_target_lengths_op, path_source_strings_op, path_strings_op,
                 path_target_strings_op])

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
