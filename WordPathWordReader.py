import tensorflow as tf
import numpy as np
from common import common

no_such_word = 'NOSUCH'
no_such_composite = no_such_word + ',' + no_such_word + ',' + no_such_word

class WordPathWordReader:
    class_word_table = None
    class_target_word_table = None
    class_node_table = None

    def __init__(self, word_to_index, target_word_to_index, node_to_index, config, is_evaluating=False):
        self.config = config
        self.file_path = config.TEST_PATH if is_evaluating else config.TRAIN_PATH
        self.batch_size = config.TEST_BATCH_SIZE if is_evaluating else config.BATCH_SIZE
        self.num_epochs = config.NUM_EPOCHS
        self.num_batching_threads = config.NUM_BATCHING_THREADS
        self.batch_queue_size = config.BATCH_QUEUE_SIZE
        self.data_num_contexts = config.DATA_NUM_CONTEXTS
        self.max_contexts = config.MAX_CONTEXTS
        self.random_contexts = config.RANDOM_CONTEXTS
        #if is_evaluating and config.RANDOM_CONTEXTS:
        #    self.max_contexts = config.DATA_NUM_CONTEXTS
        self.is_evaluating = is_evaluating
        
        self.max_path_length = config.MAX_PATH_LENGTH
        self.max_name_parts = config.MAX_NAME_PARTS
        self.max_target_parts = config.MAX_TARGET_PARTS
        
        self.record_defaults = [[no_such_composite]] * (self.data_num_contexts + 1)
        # We assume that the input data contains max_contexts contexts for each target word, including spaces for blank contexts
        # We assume that all the contexts appear in the vocabulary, words are checked dynamically

        # The default word index is 0, in which we filter the whole example
        self.word_table = WordPathWordReader.get_word_table(word_to_index)
        self.target_word_table = WordPathWordReader.get_target_word_table(target_word_to_index)
        # The default path index is 0, and we don't filter the whole example, and don't want to leave bad indices
        self.node_table = WordPathWordReader.get_node_table(node_to_index)
        if self.file_path is not None:
            self.filtered_output = self._create_filtered_output()

    @classmethod
    def get_word_table(cls, word_to_index):
        if cls.class_word_table is None:
            cls.class_word_table = cls.initalize_hash_map(word_to_index, 0)
        return cls.class_word_table

    @classmethod
    def get_target_word_table(cls, target_word_to_index):
        if cls.class_target_word_table is None:
            cls.class_target_word_table = cls.initalize_hash_map(target_word_to_index, 0)
        return cls.class_target_word_table

    @classmethod
    def get_node_table(cls, path_to_index):
        if cls.class_node_table is None:
            cls.class_node_table = cls.initalize_hash_map(path_to_index, 0)
        return cls.class_node_table

    @classmethod
    def initalize_hash_map(cls, word_to_index, default_value):
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
        
        if not self.is_evaluating and self.random_contexts:
            all_contexts = tf.stack(row_parts[1:])
            all_contexts_padded = tf.concat([all_contexts, [no_such_composite]], axis=-1)
            index_of_blank_context = tf.where(tf.equal(all_contexts_padded, no_such_composite))
            #num_contexts_per_example = tf.segment_min(data=index_of_blank_context[:, 1], segment_ids=index_of_blank_context[:, 0])
            num_contexts_per_example = tf.reduce_min(index_of_blank_context)
                        
            # if there are less than self.max_contexts valid contexts, still sample self.max_contexts
            safe_limit = tf.cast(tf.maximum(num_contexts_per_example, self.max_contexts), tf.int32)
            rand_indices = tf.random_shuffle(tf.range(safe_limit))[:self.max_contexts]
            contexts = tf.gather(all_contexts, rand_indices)  # (max_contexts,)
        else:
            contexts = row_parts[1:(self.max_contexts+1)]          # (max_contexts,)
            
        # contexts: (max_contexts, )
        split_contexts = tf.string_split(contexts, delimiter=',', skip_empty=False)
        sparse_split_contexts = tf.sparse.SparseTensor(indices=split_contexts.indices, 
                                                     values=split_contexts.values, 
                                                     dense_shape=[self.max_contexts, 3]) 
        dense_split_contexts = tf.reshape(
            tf.sparse.to_dense(sp_input=sparse_split_contexts, default_value=no_such_word),
            shape=[self.max_contexts, 3])  # (batch, max_contexts, 3)
        if self.is_evaluating:
            target_word_labels = word                                 # (batch, ) of string
            clipped_target_lengths = tf.ones(shape=tf.shape(word))
        else:
            split_target_labels = tf.string_split(tf.expand_dims(word, -1), delimiter='|')
            target_dense_shape = [1, tf.maximum(tf.to_int64(self.max_target_parts), split_target_labels.dense_shape[1]+1)]
            sparse_target_labels = tf.sparse.SparseTensor(indices=split_target_labels.indices,
                                                         values=split_target_labels.values,
                                                         dense_shape=target_dense_shape)
            dense_target_label = tf.reshape(tf.sparse.to_dense(sp_input=sparse_target_labels, 
                                                          default_value=common.blank_target_padding), [-1])
            index_of_blank = tf.where(tf.equal(dense_target_label, common.blank_target_padding))
            target_length = tf.reduce_min(index_of_blank)
            dense_target_label = dense_target_label[:self.max_target_parts] #tf.slice(dense_target_label, [0,0], [-1, self.max_target_parts])
            clipped_target_lengths = tf.clip_by_value(target_length, clip_value_min=0, clip_value_max=self.max_target_parts)
            target_word_labels = tf.concat([
                self.target_word_table.lookup(dense_target_label), [0]], axis=-1)  # (max_target_parts + 1) of int
        
        path_source_strings = tf.slice(dense_split_contexts, [0, 0], [self.max_contexts, 1]) # (max_contexts, 1)
        flat_source_strings = tf.reshape(path_source_strings, [-1])                                 # (max_contexts)
        split_source = tf.string_split(flat_source_strings, delimiter='|', skip_empty=False)                          # (max_contexts, max_name_parts)
        
        sparse_split_source = tf.sparse.SparseTensor(indices=split_source.indices, values=split_source.values,
                                                     dense_shape=[self.max_contexts, 
                                                                  tf.maximum(tf.to_int64(self.max_name_parts), split_source.dense_shape[1])])
        dense_split_source = tf.sparse.to_dense(sp_input=sparse_split_source, default_value=no_such_word) # (max_contexts, max_name_parts)
        dense_split_source = tf.slice(dense_split_source, [0,0], [-1, self.max_name_parts])
        #batched_source = tf.reshape(dense_split_source, shape=[-1, self.max_contexts, self.max_name_parts])     # (batch, max_contexts, max_name_parts)
        path_source_indices = self.word_table.lookup(dense_split_source)  # (max_contexts, max_name_parts)
        path_source_lengths = tf.reduce_sum(tf.cast(tf.not_equal(dense_split_source, no_such_word), tf.int32), -1)  # (max_contexts)
        
        path_strings = tf.slice(dense_split_contexts, [0, 1], [self.max_contexts, 1])
        flat_path_strings = tf.reshape(path_strings, [-1])
        split_path = tf.string_split(flat_path_strings, delimiter='|', skip_empty=False)
        sparse_split_path = tf.sparse.SparseTensor(indices=split_path.indices, values=split_path.values, 
                                                   dense_shape=[self.max_contexts, self.max_path_length])
        dense_split_path = tf.sparse.to_dense(sp_input=sparse_split_path, default_value=no_such_word)  # (batch, max_contexts, max_path_length)
        
        path_indices = self.node_table.lookup(dense_split_path)  # (max_contexts, max_path_length)
        path_lengths = tf.reduce_sum(tf.cast(tf.not_equal(dense_split_path, no_such_word), tf.int32), -1) # (max_contexts)
        
        path_target_strings = tf.slice(dense_split_contexts, [0, 2], [self.max_contexts, 1]) # (max_contexts, 1)
        flat_target_strings = tf.reshape(path_target_strings, [-1])                                 # (max_contexts)
        split_target = tf.string_split(flat_target_strings, delimiter='|', skip_empty=False)        # (max_contexts, max_name_parts)
        sparse_split_target = tf.sparse.SparseTensor(indices=split_target.indices, values=split_target.values, 
                                                     dense_shape=[self.max_contexts, 
                                                                  tf.maximum(tf.to_int64(self.max_name_parts), split_target.dense_shape[1])])
        dense_split_target = tf.sparse.to_dense(sp_input=sparse_split_target, default_value=no_such_word)  # (max_contexts, max_name_parts)
        dense_split_target = tf.slice(dense_split_target, [0,0], [-1, self.max_name_parts])
        path_target_indices = self.word_table.lookup(dense_split_target)  # (max_contexts, max_name_parts)
        path_target_lengths = tf.reduce_sum(tf.cast(tf.not_equal(dense_split_target, no_such_word), tf.int32), -1)  # (max_contexts)

        valid_contexts_mask = tf.to_float(tf.not_equal(tf.reduce_max(path_source_indices, -1) + tf.reduce_max(path_indices, -1) + tf.reduce_max(path_target_indices, -1), 0))
        
        return target_word_labels, tf.expand_dims(clipped_target_lengths, -1), path_source_indices, path_indices, path_target_indices, \
               valid_contexts_mask, path_source_lengths, path_lengths, path_target_lengths, \
               path_source_strings, path_strings, path_target_strings
    

    def reset(self, sess):
        sess.run(self.reset_op)
    
    def filter_dataset(self, *example):
        target_word_labels, target_length, path_source_indices, path_indices, path_target_indices, \
            valid_contexts_mask, path_source_length, path_length, path_target_length, \
            source_strings, path_strings, target_strings = example

        if self.is_evaluating:  
            return tf.constant(True)
        else: # training
            word_is_valid = tf.greater(tf.reduce_max(target_word_labels), 0)
            return word_is_valid

    def get_output(self):
        return self.filtered_output

    def _create_filtered_output(self):
        dataset = tf.data.experimental.CsvDataset(self.file_path, record_defaults=self.record_defaults, field_delim=' ', 
                                              use_quote_delim=False)
        #dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(self.batch_queue_size, self.config.NUM_EPOCHS))
        if not self.is_evaluating:
            if self.config.SAVE_EVERY_EPOCHS > 1:
                dataset = dataset.repeat(self.config.SAVE_EVERY_EPOCHS)
            dataset = dataset.shuffle(self.batch_queue_size, reshuffle_each_iteration=True)
        dataset = dataset.map(self.process_dataset, num_parallel_calls=self.config.NUM_BATCHING_THREADS)
        dataset = dataset.filter(self.filter_dataset)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(self.config.PREFETCH_NUM_BATCHES)
        self.iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
        self.reset_op = self.iterator.make_initializer(dataset)
        return self.iterator.get_next()


if __name__ == '__main__':
    target_word_to_index = {'a': 1, 'b':2, 'c':3, 'd':4, 't':5, '</S>':6}
    word_to_index = {'a': 1, 'b':2, 'c':3, 'd':4}
    path_to_index = {'1': 1, '2':2, '3':3, '4':4}
    import numpy as np
    class Config:
        def __init__(self):
            self.NUM_EPOCHS = 1
            self.SAVE_EVERY_EPOCHS = 1
            self.TRAIN_PATH = self.TEST_PATH = 'test_input/test_wpw_input.txt'
            self.BATCH_SIZE = 6
            self.TEST_BATCH_SIZE = self.BATCH_SIZE
            self.PREFETCH_NUM_BATCHES = 1
            self.NUM_BATCHING_THREADS = 1
            self.READING_BATCH_SIZE = 2
            self.BATCH_QUEUE_SIZE = 100
            self.MAX_CONTEXTS = 3
            self.DATA_NUM_CONTEXTS = 4
            self.MAX_PATH_LENGTH = 3
            self.MAX_NAME_PARTS = 2
            self.MAX_TARGET_PARTS = 4
            self.RANDOM_CONTEXTS = True
    
    
    config = Config()
    reader = WordPathWordReader(word_to_index, target_word_to_index, path_to_index, config, False)
    
    w, wlen, s, p, t, m, s_len, p_len, t_len = reader.get_output()[:-3]
    sess = tf.InteractiveSession()
    tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()).run()
    reader.reset(sess)
    
    try:
        while True:
            word, word_len, source, path, target, mask, source_len, path_len, target_len = sess.run([w, wlen, s, p, t, m, s_len, p_len, t_len])
            #print(word, np.concatenate([np.expand_dims(source, -1), np.expand_dims(path,-1), np.expand_dims(target, -1)], axis=2) + np.log(np.expand_dims(mask, 2)))
            print(word, word_len, np.concatenate([source, path, target], axis=2) + np.log(np.expand_dims(mask, 2)))
            print('Source len: ', source_len)
            print('Path len: ', path_len)
            print('Target len: ', target_len)
            #print(np.concatenate([np.expand_dims(source_len, -1), np.expand_dims(path_len, -1), np.expand_dims(target_len, -1)], axis=2))
    except tf.errors.OutOfRangeError:
        print('Done training, epoch reached')
    
