import tensorflow as tf
import re
import WordPathWordReader
import numpy as np
import time
import sys
import _pickle as pickle

from common import common


class Model:
    topk = 10
    num_batches_to_log = 100

    def __init__(self, config):
        self.config = config
        self.sess = tf.Session()

        self.eval_data_lines = None
        self.eval_queue = None
        self.predict_queue = None

        self.eval_placeholder = None
        self.predict_placeholder = None
        self.eval_top_words_op, self.eval_top_values_op, self.eval_original_names_op, self.eval_topk_values = None, None, None, None
        self.predict_top_words_op, self.predict_top_values_op, self.predict_original_names_op = None, None, None
        
        self.word_to_index = None

        if config.LOAD_PATH:
            self.load_model(sess=None)
        else:
            self.word_to_index, self.index_to_word, self.word_vocab_size = \
                common.load_vocab_from_histogram(config.WORDS_HISTOGRAM_PATH, config.WORDS_MIN_COUNT, start_from=1)
            print('Loaded word vocab. size: %d' % self.word_vocab_size)

            self.target_word_to_index, self.index_to_target_word, self.target_word_vocab_size = \
                common.load_vocab_from_histogram(config.TARGET_WORDS_HISTOGRAM_PATH, config.TARGET_WORDS_MIN_COUNT,
                                                 start_from=1, add_values=[common.PRED_START])
            print('Loaded target word vocab. size: %d' % self.target_word_vocab_size)

            self.node_to_index, self.index_to_node, self.nodes_vocab_size = \
                common.load_vocab_from_histogram(config.NODES_HISTOGRAM_PATH, 0, start_from=1)
            print('Loaded nodes vocab. size: %d' % self.nodes_vocab_size)

        self.index_to_target_word_table = tf.contrib.lookup.HashTable(
            tf.contrib.lookup.KeyValueTensorInitializer(list(self.index_to_target_word.keys()),
                                                        list(self.index_to_target_word.values()),
                                                        key_dtype=tf.int64, value_dtype=tf.string),
            default_value=tf.constant(common.blank_target_padding, dtype=tf.string))
        self.target_word_to_index_table = tf.contrib.lookup.HashTable(
            tf.contrib.lookup.KeyValueTensorInitializer(list(self.target_word_to_index.keys()),
                                                        list(self.target_word_to_index.values()),
                                                        key_dtype=tf.string, value_dtype=tf.int32),
            default_value=0)
        
    def close_session(self):
        self.sess.close()

    def train(self):
        print('Starting training')
        start_time = time.time()

        # summaries = tf.summary.merge_all()
        # summary_writer = tf.summary.FileWriter('summary', sess.graph)
        batch_num = 0
        sum_loss = 0
        #num_batches_to_evaluate = int(self.config.NUM_EXAMPLES / self.config.BATCH_SIZE * self.config.SAVE_EVERY_EPOCHS)

        self.queue_thread = WordPathWordReader.WordPathWordReader(word_to_index=self.word_to_index,
                                                                  node_to_index=self.node_to_index,
                                                                  target_word_to_index=self.target_word_to_index,
                                                                  config=self.config)
        optimizer, train_loss = self.build_training_graph(self.queue_thread.get_output())

        self.initialize_session_variables(self.sess)
        print('Initalized variables')
        if self.config.LOAD_PATH:
            self.load_model(self.sess)

        time.sleep(1)
        print('Started reader...')

        multi_batch_start_time = time.time()
        for iteration in range(1, self.config.NUM_EPOCHS + 1):
            self.queue_thread.reset(self.sess)
            try:
                while True:
                    batch_num += 1
                    _, batch_loss = self.sess.run([optimizer, train_loss])
                    sum_loss += batch_loss
                    if batch_num % self.num_batches_to_log == 0:
                        self.trace(sum_loss, batch_num, multi_batch_start_time)
                        #print('Number of waiting examples in queue: %d' % self.sess.run(
                        #    "shuffle_batch/random_shuffle_queue_Size:0"))
                        sum_loss = 0
                        multi_batch_start_time = time.time()


            except tf.errors.OutOfRangeError:
                epoch_num = iteration * self.config.SAVE_EVERY_EPOCHS
                print('Finished %d epochs' % self.config.SAVE_EVERY_EPOCHS)
                save_target = self.config.SAVE_PATH + '_iter' + str(epoch_num)
                self.save_model(self.sess, save_target)
                print('Saved after %d epochs in: %s' % (epoch_num, save_target))
                results, precision, recall, f1 = self.evaluate()
                print('Accuracy after %d epochs: %f' % (epoch_num, results))
                print('After ' + str(epoch_num) + ' epochs: Precision: ' + str(precision) + ', recall: ' + str(
                    recall) + ', F1: ' + str(f1))

        if self.config.SAVE_PATH:
            self.save_model(self.sess, self.config.SAVE_PATH)
            print('Model saved in file: %s' % self.config.SAVE_PATH)

        elapsed = int(time.time() - start_time)
        print("Training time: %sh%sm%ss\n" % ((elapsed // 60 // 60), (elapsed // 60) % 60, elapsed % 60))

    def trace(self, sum_loss, batch_num, multi_batch_start_time):
        multi_batch_elapsed = time.time() - multi_batch_start_time
        avg_loss = sum_loss / (self.num_batches_to_log * self.config.BATCH_SIZE)
        print('Average loss at batch %d: %f, \tthroughput: %d samples/sec' % (batch_num, avg_loss,
                                                                                             self.config.BATCH_SIZE * self.num_batches_to_log / (
                                                                                multi_batch_elapsed if multi_batch_elapsed > 0 else 1)))

    def evaluate(self):
        eval_start_time = time.time()
        if self.eval_queue is None:
            self.eval_queue = WordPathWordReader.WordPathWordReader(word_to_index=self.word_to_index,
                                                                    node_to_index=self.node_to_index,
                                                                    target_word_to_index=self.target_word_to_index,
                                                                    config=self.config, is_evaluating=True)
            self.eval_top_words_op, self.eval_topk_values, self.eval_original_names_op, _, _, _, _ = \
                self.build_test_graph(self.eval_queue.get_output())
            self.saver = tf.train.Saver(max_to_keep=10)

        if self.config.LOAD_PATH and not self.config.TRAIN_PATH:
            self.initialize_session_variables(self.sess)
            self.load_model(self.sess)
        ref_file_name = 'ref.txt'
        predicted_file_name = 'pred.txt'
        with open('log.txt', 'w') as output_file, open(ref_file_name, 'w') as ref_file, open(predicted_file_name, 'w') as pred_file:
            num_correct_predictions = 0
            total_predictions = 0
            total_prediction_batches = 0
            true_positive, false_positive, false_negative = 0, 0, 0
            self.eval_queue.reset(self.sess)
            start_time = time.time()
            
            try:
                while True:
                    top_words, original_names, top_values = self.sess.run(
                        [self.eval_top_words_op, self.eval_original_names_op, self.eval_topk_values],
                        )
                    original_names = common.binary_to_string_list(original_names)
                    if self.config.BEAM_WIDTH > 0:
                        top_words = common.binary_to_string_3d(top_words)
                        top_words = [[pred[0] for pred in batch] for batch in top_words]
                    else:
                        top_words = common.binary_to_string_matrix(top_words)
                    # Flatten original names from [[]] to []
                    #original_names = [w for l in original_names for w in l]
                    ref_file.write('\n'.join([name.replace(common.internal_delimiter, ' ') for name in original_names]) + '\n')
                    pred_file.write('\n'.join([' '.join(common.filter_impossible_names(subtokens)) for subtokens in top_words]) + '\n')
    
                    num_correct_predictions = self.update_correct_predictions(num_correct_predictions, output_file,
                                                                              zip(original_names, top_words))
                    true_positive, false_positive, false_negative = self.update_per_subtoken_statistics(
                        zip(original_names, top_words),
                        true_positive, false_positive, false_negative)
    
                    total_predictions += len(original_names)
                    total_prediction_batches += 1
                    if total_prediction_batches % self.num_batches_to_log == 0:
                        elapsed = time.time() - start_time
                        #start_time = time.time()
                        self.trace_evaluation(output_file, num_correct_predictions, total_predictions, elapsed)
            except tf.errors.OutOfRangeError:
                pass
            
            print('Done testing, epoch reached')
            output_file.write(str(num_correct_predictions / total_predictions) + '\n')
            common.compute_bleu(ref_file_name, predicted_file_name)

        elapsed = int(time.time() - eval_start_time)
        precision, recall, f1 = self.calculate_results(true_positive, false_positive, false_negative)
        print("Evaluation time: %sh%sm%ss" % ((elapsed / 60 / 60), (elapsed / 60) % 60, elapsed % 60))
        del self.eval_data_lines
        self.eval_data_lines = None
        return num_correct_predictions / total_predictions, precision, recall, f1

    def update_correct_predictions(self, num_correct_predictions, output_file, results):
        for original_name, predicted_suggestions in results: # top_words: (num_targets, topk)
            predicted = predicted_suggestions #[0]
            original_name_parts = original_name.split(common.internal_delimiter)
            output_file.write('Original: ' + common.internal_delimiter.join(original_name_parts) + 
                              ' , predicted 1st: ' + common.internal_delimiter.join([target for target in predicted]) + '\n')
            
            filtered_original = common.filter_impossible_names(original_name_parts)
            filtered_predicted_parts = common.filter_impossible_names(predicted)
            if filtered_original == filtered_predicted_parts or common.unique(filtered_original) == common.unique(filtered_predicted_parts) or ''.join(filtered_original) == ''.join(filtered_predicted_parts):
                num_correct_predictions += 1
        return num_correct_predictions
    
    def update_per_subtoken_statistics(self, results, true_positive, false_positive, false_negative):
        for original_name, predicted_suggestions in results: # top_words: (num_target_parts, topk)
            predicted = predicted_suggestions #[0]
            filtered_predicted_names = common.filter_impossible_names(predicted)
            filtered_original_subtokens = common.filter_impossible_names(original_name.split(common.internal_delimiter))
            #if len(filtered_predicted_names) > 0 and len(filtered_predicted_names[0]) > 0:
                
            '''for target_position in filtered_predicted_names:
                for target_position_suggestion in target_position:
                    if target_position_suggestion == common.blank_target_padding:
                        break
                    if not target_position_suggestion in predicted_subtokens:
                        predicted_subtokens.append(target_position_suggestion)
                        break'''

            if ''.join(filtered_original_subtokens) == ''.join(filtered_predicted_names):
                true_positive += len(filtered_original_subtokens)
                continue
            
            for subtok in filtered_predicted_names:
                if subtok in filtered_original_subtokens:
                    true_positive += 1
                else:
                    false_positive += 1
            for subtok in filtered_original_subtokens:
                if not subtok in filtered_predicted_names:
                    false_negative += 1
        return true_positive, false_positive, false_negative

    @staticmethod
    def calculate_results(true_positive, false_positive, false_negative):
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0
        return precision, recall, f1

    @staticmethod
    def trace_evaluation(output_file, correct_predictions, total_predictions, elapsed):
        accuracy_message = str(correct_predictions / total_predictions)
        throughput_message = "Prediction throughput: %d" % int(total_predictions / (elapsed if elapsed > 0 else 1))
        output_file.write(accuracy_message + '\n')
        output_file.write(throughput_message)
        print(accuracy_message)
        print(throughput_message)




    def build_training_graph(self, input_tensors):
        words_input, word_input_lengths, source_input, path_input, target_input, valid_mask, \
            source_lengths, path_lengths, target_lengths, _, _, _ = input_tensors  
        # (batch, max_target_parts), (batch, max_contexts, max_name_parts), (batch, max_contexts, max_path_length), (batch, max_contexts, max_name_parts), (batch, max_contexts), 
        # (batch, max_contexts), (batch, max_contexts) 

        with tf.variable_scope('model'):
            words_vocab = tf.get_variable('WORDS_VOCAB', shape=(self.word_vocab_size + 1, self.config.EMBEDDINGS_SIZE),
                                          dtype=tf.float32,
                                          initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                                                                     mode='FAN_OUT',
                                                                                                     uniform=True))
            target_words_vocab = tf.get_variable('TARGET_WORDS_VOCAB',
                                                 shape=(self.target_word_vocab_size + 1, self.config.EMBEDDINGS_SIZE),
                                                 dtype=tf.float32,
                                                 initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                                                                            mode='FAN_OUT',
                                                                                                            uniform=True))
            nodes_vocab = tf.get_variable('NODES_VOCAB', shape=(self.nodes_vocab_size + 1, self.config.EMBEDDINGS_SIZE),
                                          dtype=tf.float32,
                                          initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                                                                     mode='FAN_OUT',
                                                                                                     uniform=True))

            #biases = tf.zeros(shape=(self.target_word_vocab_size + 1), dtype=tf.float32)

            batched_inputs = self.compute_contexts(words_vocab, nodes_vocab, source_input, path_input, target_input,
                                                        valid_mask, source_lengths, path_lengths, target_lengths)
                # (batch, max_contexts, dim)

            batch_size = tf.shape(words_input)[0]
            outputs, final_states = self.decode_outputs(target_words_vocab, words_input, batch_size, batched_inputs, valid_mask, is_evaluating=False)
            self.saver = tf.train.Saver(max_to_keep=10)
            logits = outputs.rnn_output # (batch, max_output_length, dim * 2 + rnn_size)

            #shifted_lables = tf.concat([words_input, tf.fill([batch_size, 1], 0)], axis=-1)
            #sampled_softmax = tf.nn.sampled_softmax_loss(target_words_vocab, biases, tf.reshape(words_input, [-1, 1]), tf.reshape(logits, [-1, 10]), num_sampled=3, num_classes=self.target_word_vocab_size+1)
            # words_input: (batch, max_target_parts)
            # logits: (batch, max_target_parts, target_vocab_size)
            
            # permutated_words_input: (batch, max_target_parts!, max_target_parts)
            # tiled_logits: (batch, max_target_parts!, max_target_parts, target_vocab_size)
            # 
            crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=words_input, logits=logits)
            #target_words_nonzero = tf.concat([
            #    tf.ones((batch_size, 1), dtype=tf.float32),
            #    tf.slice(tf.to_float(tf.greater(words_input, 0)), (0, 0), (-1, self.config.MAX_TARGET_PARTS))
            #], axis=-1)
            target_words_nonzero = tf.sequence_mask(tf.squeeze(word_input_lengths)+1, maxlen=self.config.MAX_TARGET_PARTS+1, dtype=tf.float32)
            loss = tf.reduce_sum(crossent * target_words_nonzero) / tf.to_float(batch_size)
            
            #flat_weighted_average_contexts = tf.reshape(weighted_average_contexts, shape=(-1, self.config.EMBEDDINGS_SIZE * 2 + self.config.RNN_SIZE))
            
            #logits = tf.matmul(flat_weighted_average_contexts, target_words_vocab)
            
            # labels_one_hot = tf.one_hot(tf.squeeze(words_input, axis=[1]), self.target_word_vocab_size + 1)
            #cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(words_input, [-1]), logits=logits)
            #no_blank = tf.reshape(tf.to_float(tf.greater(words_input, 0)), [-1])
            #no_blank_factor = 2
            #loss = tf.reduce_sum(cross_entropy + (cross_entropy * no_blank * (no_blank_factor - 1)))
            #loss = tf.reduce_sum(cross_entropy)
            #predictions = tf.cast(tf.argmax(logits, axis=-1), tf.int32)
            #accuracy = tf.reduce_mean(tf.to_float(tf.equal(predictions, tf.reshape(words_input, [-1]))))
            # loss = tf.reduce_sum(tf.nn.sampled_softmax_loss(weights=target_words_vocab, inputs=weighted_average_contexts,
            #    labels=words_input, num_sampled=self.config.SOFTMAX_SAMPLE,
            #    num_classes=self.target_word_vocab_size+1, biases=biases))
            
            # Calculate and clip gradients
            step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(0.01, step * self.config.BATCH_SIZE, self.config.NUM_EXAMPLES, 0.95, staircase=True)
            optimizer = tf.train.MomentumOptimizer(learning_rate, 0.95, use_nesterov=True)
            train_op = optimizer.minimize(loss, global_step=step)
        return train_op, loss

    def decode_outputs(self, target_words_vocab, words_input, batch_size, batched_embed, valid_mask, is_evaluating):
        num_contexts_per_example = tf.count_nonzero(valid_mask, axis=-1)
        
        start_fill = tf.fill([batch_size], self.target_word_to_index_table.lookup(tf.constant(common.PRED_START))) # (batch, )
        decoder_cell = tf.nn.rnn_cell.LSTMCell(self.config.DECODER_SIZE)
        contexts_sum = tf.reduce_sum(batched_embed * tf.expand_dims(valid_mask, -1), axis=1) # (batch_size, dim * 2 + rnn_size)
        contexts_average = tf.divide(contexts_sum, tf.to_float(tf.expand_dims(num_contexts_per_example, -1)))
        fake_encoder_state = tf.nn.rnn_cell.LSTMStateTuple(contexts_average, contexts_average)
        projection_layer = tf.layers.Dense(self.target_word_vocab_size + 1, use_bias=False)
        if is_evaluating and self.config.BEAM_WIDTH > 0:
            batched_embed = tf.contrib.seq2seq.tile_batch(batched_embed, multiplier=self.config.BEAM_WIDTH)
            num_contexts_per_example = tf.contrib.seq2seq.tile_batch(num_contexts_per_example, multiplier=self.config.BEAM_WIDTH) 
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(
            num_units=self.config.DECODER_SIZE,
            memory=batched_embed,
            #memory_sequence_length=num_contexts_per_example,
        )
        should_save_alignment_history = is_evaluating and self.config.BEAM_WIDTH == 0 # TF doesn't support beam search with alignment history
        decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism, attention_layer_size=self.config.DECODER_SIZE, alignment_history=should_save_alignment_history)
        if is_evaluating:
            if self.config.BEAM_WIDTH > 0:
                decoder_initial_state = decoder_cell.zero_state(dtype=tf.float32, batch_size=batch_size*self.config.BEAM_WIDTH)
                decoder_initial_state = decoder_initial_state.clone(cell_state=tf.contrib.seq2seq.tile_batch(fake_encoder_state, multiplier=self.config.BEAM_WIDTH))
                decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                    cell=decoder_cell,
                    embedding=target_words_vocab,
                    start_tokens=start_fill,
                    end_token=0,
                    initial_state=decoder_initial_state,
                    beam_width=self.config.BEAM_WIDTH,
                    output_layer=projection_layer,
                    length_penalty_weight=0.0)
            else:
                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(target_words_vocab, start_fill, 0)
                initial_state = decoder_cell.zero_state(batch_size, tf.float32).clone(cell_state=fake_encoder_state)
                decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell, helper=helper, initial_state=initial_state, output_layer=projection_layer)
                
        else:
            decoder_cell = tf.nn.rnn_cell.DropoutWrapper(decoder_cell, output_keep_prob=self.config.RNN_DROPOUT_KEEP_PROB)
            target_words_embedding = tf.nn.embedding_lookup(target_words_vocab, tf.concat([tf.expand_dims(start_fill, -1), words_input], axis=-1))  # (batch, max_target_parts, dim * 2 + rnn_size)
            helper = tf.contrib.seq2seq.TrainingHelper(inputs=target_words_embedding,
                sequence_length=tf.ones([batch_size], dtype=tf.int32) * (self.config.MAX_TARGET_PARTS+1))  # tf.count_nonzero(words_input, axis=-1, dtype=tf.int32))
         
            initial_state = decoder_cell.zero_state(batch_size, tf.float32).clone(cell_state=fake_encoder_state)
        
            decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell, helper=helper, initial_state=initial_state,
                                              output_layer=projection_layer)
        outputs, final_states, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=self.config.MAX_TARGET_PARTS+1)
        return outputs, final_states

    def calculate_path_abstraction(self, path_embed, path_lengths, valid_contexts_mask, is_evaluating=False):
        #return self.path_max_pool(is_evaluating, path_embed, path_lengths, valid_contexts_mask)
        return self.path_rnn_last_state(is_evaluating, path_embed, path_lengths, valid_contexts_mask)
    
    def path_rnn_last_state(self, is_evaluating, path_embed, path_lengths, valid_contexts_mask):
        # path_embed:           (batch, max_contexts, max_path_length+1, dim)
        # path_length:          (batch, max_contexts)
        # valid_contexts_mask:  (batch, max_contexts)
        max_contexts = tf.shape(path_embed)[1]
        flat_paths = tf.reshape(path_embed, shape=[-1, self.config.MAX_PATH_LENGTH,
                                                   self.config.EMBEDDINGS_SIZE])  # (batch * max_contexts, max_path_length+1, dim)
        flat_valid_contexts_mask = tf.reshape(valid_contexts_mask, [-1])  # (batch * max_contexts)
        lengths = tf.multiply(tf.reshape(path_lengths, [-1]),
                              tf.cast(flat_valid_contexts_mask, tf.int32))  # (batch * max_contexts)
        if self.config.BIRNN:
            rnn_cell_fw = tf.nn.rnn_cell.LSTMCell(self.config.RNN_SIZE / 2)
            rnn_cell_bw = tf.nn.rnn_cell.LSTMCell(self.config.RNN_SIZE / 2)
            if not is_evaluating:
                rnn_cell_fw = tf.nn.rnn_cell.DropoutWrapper(rnn_cell_fw, output_keep_prob=self.config.RNN_DROPOUT_KEEP_PROB)
                rnn_cell_bw = tf.nn.rnn_cell.DropoutWrapper(rnn_cell_bw, output_keep_prob=self.config.RNN_DROPOUT_KEEP_PROB)
            _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=rnn_cell_fw,
                cell_bw=rnn_cell_bw,
                inputs=flat_paths,
                dtype=tf.float32,
                sequence_length=lengths)
            final_rnn_state = tf.concat([state_fw.h, state_bw.h], axis=-1) # (batch * max_contexts, rnn_size)  
        else:
            rnn_cell = tf.nn.rnn_cell.LSTMCell(self.config.RNN_SIZE)
            if not is_evaluating:
                rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, output_keep_prob=self.config.RNN_DROPOUT_KEEP_PROB)
            _, state = tf.nn.dynamic_rnn(
                cell=rnn_cell,
                inputs=flat_paths,
                dtype=tf.float32,
                sequence_length=lengths
            )  # (batch * max_contexts, max_path_length + 1, rnn_size / 2) 
            final_rnn_state = state.h
        
        
        return tf.reshape(final_rnn_state, shape=[-1, max_contexts, self.config.RNN_SIZE])  # (batch, max_contexts, rnn_size)
    
    def path_max_pool(self, is_evaluating, path_embed, path_lengths, valid_contexts_mask):
        # path_embed:           (batch, max_contexts, max_path_length+1, dim)
        # path_length:          (batch, max_contexts)
        # valid_contexts_mask:  (batch, max_contexts)
        flat_paths = tf.reshape(path_embed, shape=[-1, self.config.MAX_PATH_LENGTH,
                                                   self.config.EMBEDDINGS_SIZE])  # (batch * max_contexts, max_path_length+1, dim)
        flat_valid_contexts_mask = tf.reshape(valid_contexts_mask, [-1])  # (batch * max_contexts)
        lengths = tf.multiply(tf.reshape(path_lengths, [-1]),
                              tf.cast(flat_valid_contexts_mask, tf.int32))  # (batch * max_contexts)
        if self.config.BIRNN:
            rnn_cell_fw = tf.nn.rnn_cell.LSTMCell(self.config.RNN_SIZE / 2)
            rnn_cell_bw = tf.nn.rnn_cell.LSTMCell(self.config.RNN_SIZE / 2)
            if not is_evaluating:
                rnn_cell_fw = tf.nn.rnn_cell.DropoutWrapper(rnn_cell_fw,
                                                            output_keep_prob=self.config.RNN_DROPOUT_KEEP_PROB)
                rnn_cell_bw = tf.nn.rnn_cell.DropoutWrapper(rnn_cell_bw,
                                                            output_keep_prob=self.config.RNN_DROPOUT_KEEP_PROB)
            output, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=rnn_cell_fw,
                cell_bw=rnn_cell_bw,
                inputs=flat_paths,
                dtype=tf.float32,
                sequence_length=lengths)
            output = tf.concat(output, 2)  # (batch * max_contexts, max_path_length+1, rnn_size / 2)
            # final_rnn_state = tf.concat([state_fw.h, state_bw.h], axis=-1) # (batch * max_contexts, rnn_size) 
        else:
            rnn_cell = tf.nn.rnn_cell.LSTMCell(self.config.RNN_SIZE)
            if not is_evaluating:
                rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, output_keep_prob=self.config.RNN_DROPOUT_KEEP_PROB)
            output, _ = tf.nn.dynamic_rnn(
                cell=rnn_cell,
                inputs=flat_paths,
                dtype=tf.float32,
                sequence_length=lengths
            )  # (batch * max_contexts, max_path_length + 1, rnn_size / 2) 
            # final_rnn_state = state.h
            # output # (batch * max_contexts, max_path_length+1, rnn_size / 2)
        mask = tf.expand_dims(tf.to_float(tf.sequence_mask(lengths, maxlen=self.config.MAX_PATH_LENGTH)),
                              -1)  # (batch * max_contexts, max_path_length+1, 1)
        output = tf.multiply(mask, output)  # (batch * max_contexts, max_path_length+1, rnn_size / 2)
        max_output = tf.reduce_max(output, axis=1)  # (batch * max_contexts, rnn_size)
        final_rnn_state = max_output  # (batch * max_contexts, rnn_size)
        return tf.reshape(final_rnn_state,
                          shape=[-1, self.config.MAX_CONTEXTS, self.config.RNN_SIZE])  # (batch, max_contexts, rnn_size)
        # paths_mask = tf.expand_dims(tf.sequence_mask(path_lengths, maxlen=self.config.MAX_PATH_LENGTH, dtype=tf.float32), -1)        # (batch, max_contexts, max_path_length+1, 1)
        # return tf.reduce_sum(path_embed * paths_mask, axis=2)                      # (batch, max_contexts, dim)

    def compute_contexts(self, words_vocab, nodes_vocab, source_input, path_input,
                         target_input, valid_mask, source_lengths, path_lengths, target_lengths, is_evaluating=False):
        max_contexts = tf.shape(source_input)[1]
        
        source_word_embed = tf.nn.embedding_lookup(params=words_vocab, ids=source_input)  # (batch, max_contexts, max_name_parts, dim)
        path_embed = tf.nn.embedding_lookup(params=nodes_vocab, ids=path_input)  # (batch, max_contexts, max_path_length+1, dim)
        target_word_embed = tf.nn.embedding_lookup(params=words_vocab, ids=target_input)  # (batch, max_contexts, max_name_parts, dim)
        
        #clipped_source_lengths = tf.maximum(1, source_lengths)
        #clipped_target_lengths = tf.maximum(1, target_lengths)
        
        source_word_mask = tf.expand_dims(tf.sequence_mask(source_lengths, maxlen=self.config.MAX_NAME_PARTS, dtype=tf.float32), -1) # (batch, max_contexts, max_name_parts, 1)
        target_word_mask = tf.expand_dims(tf.sequence_mask(target_lengths, maxlen=self.config.MAX_NAME_PARTS, dtype=tf.float32), -1) # (batch, max_contexts, max_name_parts, 1)
        
        source_words_sum = tf.reduce_sum(source_word_embed * source_word_mask, axis=2)  # (batch, max_contexts, dim)        
        path_nodes_aggregation = self.calculate_path_abstraction(path_embed, path_lengths, valid_mask, is_evaluating)     # (batch, max_contexts, rnn_size)
        target_words_sum = tf.reduce_sum(target_word_embed * target_word_mask, axis=2)  # (batch, max_contexts, dim)
        
        
        context_embed = tf.concat([source_words_sum, path_nodes_aggregation, target_words_sum], axis=-1)  # (batch, max_contexts, dim * 2 + rnn_size)
        if not is_evaluating:
            context_embed = tf.nn.dropout(context_embed, self.config.EMBEDDINGS_DROPOUT_KEEP_PROB)

        flat_embed = tf.reshape(context_embed, [-1, self.config.EMBEDDINGS_SIZE * 2 + self.config.RNN_SIZE])  # (batch * max_contexts, dim * 2 + rnn_size)
        transform_param = tf.get_variable('TRANSFORM',
                                          shape=(self.config.EMBEDDINGS_SIZE * 2 + self.config.RNN_SIZE, self.config.DECODER_SIZE),
                                          dtype=tf.float32)

        flat_embed = tf.tanh(tf.matmul(flat_embed, transform_param))  # (batch * max_contexts, dim)

        #attention_weights = tf.nn.softmax(batched_contexts_weights, dim=1)  # (batch, max_contexts, 1)
        batched_embed = tf.reshape(flat_embed, shape=[-1, max_contexts, self.config.DECODER_SIZE]) # (batch, max_contexts, dim)
             
        #weighted_average_contexts = tf.reduce_sum(tf.multiply(batched_embed, attention_weights), axis=1)  # (batch, dim * 2 + rnn_size)
             
        return batched_embed

    def build_test_graph(self, input_tensors):
        with tf.variable_scope('model', reuse=self.get_should_reuse_variables()):
            words_vocab = tf.get_variable('WORDS_VOCAB', shape=(self.word_vocab_size + 1, self.config.EMBEDDINGS_SIZE),
                                          dtype=tf.float32, trainable=False)
            target_words_vocab = tf.get_variable('TARGET_WORDS_VOCAB',
                                                 shape=(self.target_word_vocab_size + 1, self.config.EMBEDDINGS_SIZE),
                                                 dtype=tf.float32, trainable=False)
            nodes_vocab = tf.get_variable('NODES_VOCAB',
                                          shape=(self.nodes_vocab_size + 1, self.config.EMBEDDINGS_SIZE),
                                          dtype=tf.float32, trainable=False)

            index_to_target_word_table = self.index_to_target_word_table

            words_input, word_input_lengths, source_input, path_input, target_input, valid_mask, source_lengths, path_lengths, target_lengths, source_string, path_string, path_target_string = input_tensors  # (batch, max_target_words), (batch, max_contexts)

            batched_inputs = self.compute_contexts(words_vocab, nodes_vocab, source_input, path_input, target_input, valid_mask, source_lengths, path_lengths, target_lengths, True)
            # (batch, dim * 2 + rnn_size), .. , (batch, max_contexts, dim * 2 + rnn_size)
        
            outputs, final_states = self.decode_outputs(target_words_vocab, words_input, tf.shape(words_input)[0], batched_inputs, valid_mask, is_evaluating=True)
        
        if self.config.BEAM_WIDTH > 0:
            #translations = tf.squeeze(tf.slice(outputs.predicted_ids, [0,0,0], [-1, 1, -1]), 1)     # (batch, max_target_parts)
            translations = outputs.predicted_ids
            topk_values = outputs.beam_search_decoder_output.scores
            attention_weights = [tf.no_op()]
        else:
            translations = outputs.sample_id
            topk_values = tf.constant(1, shape=(1,1), dtype=tf.float32)
            attention_weights = tf.squeeze(final_states.alignment_history.stack(), 1)
        #flat_weighted_average_contexts = tf.reshape(weighted_average_contexts, shape=(-1, self.config.EMBEDDINGS_SIZE * 2 + self.config.RNN_SIZE))
            # (batch * max_target_paths, dim * 2 + rnn_size)
        #cos = tf.matmul(flat_weighted_average_contexts, target_words_vocab) # (batch * max_target_paths, target_word_vocab+1)
        topk = tf.minimum(self.topk, self.target_word_vocab_size)
        #topk_candidates = tf.nn.top_k(cos, k=topk)
        top_indices = None # tf.reshape(tf.to_int64(topk_candidates.indices), shape=(-1, self.config.MAX_TARGET_PARTS, topk))      # (batch, max_target_parts, topk) of int64
         # tf.reshape(topk_candidates.values, shape=(-1, self.config.MAX_TARGET_PARTS, topk))                    # (batch, max_target_parts, topk) of float32
        predicted_strings = index_to_target_word_table.lookup(tf.cast(translations, dtype=tf.int64))  # (batch, max_target_parts) of string
        #original_words = index_to_target_word_table.lookup(tf.to_int64(words_input))
        original_words = words_input

        return predicted_strings, topk_values, original_words, attention_weights, source_string, path_string, path_target_string

    def predict(self, predict_data_lines):
        if self.predict_queue is None:
            self.predict_queue = WordPathWordReader.WordPathWordReader(word_to_index=self.word_to_index,
                                                                       node_to_index=self.node_to_index,
                                                                       target_word_to_index=self.target_word_to_index,
                                                                       config=self.config, is_evaluating=True)
            self.predict_placeholder = tf.placeholder(tf.string)
            reader_output = self.predict_queue.process_from_placeholder(self.predict_placeholder)
            reader_output = [tf.expand_dims(tensor, 0) for tensor in reader_output]
            self.predict_top_words_op, self.predict_top_values_op, self.predict_original_names_op, \
                self.attention_weights_op, self.predict_source_string, self.predict_path_string, self.predict_path_target_string = \
                self.build_test_graph(reader_output)

            self.initialize_session_variables(self.sess)
            self.saver = tf.train.Saver()
            self.load_model(self.sess)

        results = []
        for line in predict_data_lines:
            predicted_strings, top_scores, original_name, attention_weights, source_strings, path_strings, target_strings = self.sess.run(
                [self.predict_top_words_op, self.predict_top_values_op, self.predict_original_names_op, self.attention_weights_op,
                 self.predict_source_string, self.predict_path_string, self.predict_path_target_string],
                feed_dict={self.predict_placeholder: line})
            
            top_scores = np.squeeze(top_scores)
            source_strings = np.squeeze(source_strings)
            path_strings = np.squeeze(path_strings)
            target_strings = np.squeeze(target_strings)
            predicted_strings = np.squeeze(predicted_strings, axis=0)
            
            if self.config.BEAM_WIDTH > 0:
                predicted_strings = common.binary_to_string_matrix(predicted_strings) # (batch, target_length, top-k)
                predicted_strings = [list(map(list, zip(*batch))) for batch in predicted_strings] # (batch, top-k, target_length)
                top_scores = [np.exp(np.sum(s, 0)) for s in top_scores]
            #else:
                #predicted_strings = [[sugg] for sugg in common.binary_to_string_list(predicted_strings)] # (batch, top-1, target_length)
            original_name = common.binary_to_string(original_name[0])
            predicted_strings = common.binary_to_string_list(predicted_strings)
            
            attention_per_path = None
            if self.config.BEAM_WIDTH == 0:
                attention_per_path = self.get_attention_per_path(source_strings, path_strings, target_strings, attention_weights)
            
            #original_names = [w for l in original_names for w in l]
            results.append((original_name, predicted_strings, top_scores, attention_per_path))
        return results

    def get_attention_per_path(self, source_strings, path_strings, target_strings, attention_weights):
        # attention_weights # (time, contexts)
        results = []
        for time_step in attention_weights:
            attention_per_context = {}
            for source, path, target, weight in zip(source_strings, path_strings, target_strings, time_step):
                string_triplet = (common.binary_to_string(source), common.binary_to_string(path), common.binary_to_string(target))
                attention_per_context[string_triplet] = weight
            results.append(attention_per_context)
        return results

    @staticmethod
    def score_per_word_in_batch(words, weighted_average_contexts_per_word):
        """
        calculates (word dot avg_context) for each word and its corresponding average context 
        
        :param words:                                   # (batch, num_words, dim)
        :param weighted_average_contexts_per_word:      # (batch, num_words, dim)
        :return: score for every word in every batch    # (batch, num_words)
        """
        word_scores = tf.reduce_sum(tf.multiply(
            words, weighted_average_contexts_per_word),
            axis=2)  # (batch, num_words)

        # word_scores = tf.einsum('ijk,ijk->ij', words, weighted_average_contexts_per_word)
        return word_scores

    def init_graph_from_values(self, session,
                               final_words, words_vocab_variable,
                               final_words_attention, words_attention_vocab_variable,
                               final_contexts, contexts_vocab_variable,
                               final_attention_param, attention_variable):
        words_placeholder = tf.placeholder(tf.float32, shape=(self.word_vocab_size, self.config.EMBEDDINGS_SIZE))
        words_vocab_init = words_vocab_variable.assign(words_placeholder)
        words_attention_placeholder = tf.placeholder(tf.float32,
                                                     shape=(self.word_vocab_size, self.config.EMBEDDINGS_SIZE))
        words_attention_vocab_init = words_attention_vocab_variable.assign(words_attention_placeholder)
        contexts_placeholder = tf.placeholder(tf.float32,
                                              shape=(self.nodes_vocab_size + 1, self.config.EMBEDDINGS_SIZE))
        contexts_vocab_init = contexts_vocab_variable.assign(contexts_placeholder)
        attention_placeholder = tf.placeholder(tf.float32,
                                               shape=(self.config.EMBEDDINGS_SIZE, self.config.EMBEDDINGS_SIZE))
        attention_init = attention_variable.assign(attention_placeholder)

        session.run(words_vocab_init, feed_dict={words_placeholder: final_words})
        session.run(words_attention_vocab_init, feed_dict={words_attention_placeholder: final_words_attention})
        session.run(contexts_vocab_init, feed_dict={contexts_placeholder: final_contexts})
        session.run(attention_init, feed_dict={attention_placeholder: final_attention_param})
    
    @staticmethod
    def get_dictionaries_path(model_file_path):
        dictionaries_save_file_name = "dictionaries.bin"
        return '/'.join(model_file_path.split('/')[:-1] + [dictionaries_save_file_name])
    
    def save_model(self, sess, path):
        self.saver.save(sess, path)
        
        with open(self.get_dictionaries_path(path), 'wb') as file:
            # pickle.dump(self.final_words, file)
            # pickle.dump(self.final_attention, file)
            pickle.dump(self.word_to_index, file)
            pickle.dump(self.index_to_word, file)
            pickle.dump(self.word_vocab_size, file)

            pickle.dump(self.target_word_to_index, file)
            pickle.dump(self.index_to_target_word, file)
            pickle.dump(self.target_word_vocab_size, file)

            # pickle.dump(self.final_paths, file)
            pickle.dump(self.node_to_index, file)
            pickle.dump(self.index_to_node, file)
            pickle.dump(self.nodes_vocab_size, file)

    def load_model(self, sess):
        if not sess is None:
            self.saver.restore(sess, self.config.LOAD_PATH)
            print('Done loading model')
        with open(self.get_dictionaries_path(self.config.LOAD_PATH), 'rb') as file:
            if self.word_to_index is not None:
                return
            print('Loading dictionaries from: ' + self.config.LOAD_PATH)
            self.word_to_index = pickle.load(file)
            self.index_to_word = pickle.load(file)
            self.word_vocab_size = pickle.load(file)

            self.target_word_to_index = pickle.load(file)
            self.index_to_target_word = pickle.load(file)
            self.target_word_vocab_size = pickle.load(file)

            # self.final_paths = pickle.load(file)
            self.node_to_index = pickle.load(file)
            self.index_to_node = pickle.load(file)
            self.nodes_vocab_size = pickle.load(file)
            print('Done loading dictionaries')


    @staticmethod
    def initialize_session_variables(sess):
        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()))

    def get_should_reuse_variables(self):
        if self.config.TRAIN_PATH:
            return True
        else:
            return None
