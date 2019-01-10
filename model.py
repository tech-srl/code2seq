import tensorflow as tf
import reader
import numpy as np
import time
import os
import _pickle as pickle

from common import Common


class Model:
    topk = 10
    num_batches_to_log = 100

    def __init__(self, config):
        self.config = config
        self.sess = tf.Session()

        self.eval_queue = None
        self.predict_queue = None

        self.eval_placeholder = None
        self.predict_placeholder = None
        self.eval_predicted_indices_op, self.eval_top_values_op, self.eval_true_target_strings_op, self.eval_topk_values = None, None, None, None
        self.predict_top_indices_op, self.predict_top_scores_op, self.predict_target_strings_op = None, None, None
        self.subtoken_to_index = None

        if config.LOAD_PATH:
            self.load_model(sess=None)
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
                Common.load_vocab_from_dict(subtoken_to_count, add_values=[Common.PAD, Common.UNK])
            print('Loaded subtoken vocab. size: %d' % self.subtoken_vocab_size)

            self.target_to_index, self.index_to_target, self.target_vocab_size = \
                Common.load_vocab_from_dict(target_to_count, add_values=[Common.PAD, Common.UNK, Common.SOS])
            print('Loaded target word vocab. size: %d' % self.target_vocab_size)

            self.node_to_index, self.index_to_node, self.nodes_vocab_size = \
                Common.load_vocab_from_dict(node_to_count, add_values=[Common.PAD, Common.UNK])
            print('Loaded nodes vocab. size: %d' % self.nodes_vocab_size)

    def close_session(self):
        self.sess.close()

    def train(self):
        print('Starting training')
        start_time = time.time()

        batch_num = 0
        sum_loss = 0
        best_f1 = 0
        best_epoch = 0
        best_f1_precision = 0
        best_f1_recall = 0
        epochs_no_improve = 0

        self.queue_thread = reader.Reader(subtoken_to_index=self.subtoken_to_index,
                                          node_to_index=self.node_to_index,
                                          target_to_index=self.target_to_index,
                                          config=self.config)
        optimizer, train_loss = self.build_training_graph(self.queue_thread.get_output())
        self.print_hyperparams()
        print('Number of trainable params:', 
              np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
        self.initialize_session_variables(self.sess)
        print('Initalized variables')
        if self.config.LOAD_PATH:
            self.load_model(self.sess)

        time.sleep(1)
        print('Started reader...')

        multi_batch_start_time = time.time()
        for iteration in range(1, (self.config.NUM_EPOCHS // self.config.SAVE_EVERY_EPOCHS) + 1):
            self.queue_thread.reset(self.sess)
            try:
                while True:
                    batch_num += 1
                    _, batch_loss = self.sess.run([optimizer, train_loss])
                    sum_loss += batch_loss
                    if batch_num % self.num_batches_to_log == 0:
                        self.trace(sum_loss, batch_num, multi_batch_start_time)
                        sum_loss = 0
                        multi_batch_start_time = time.time()


            except tf.errors.OutOfRangeError:
                epoch_num = iteration * self.config.SAVE_EVERY_EPOCHS
                print('Finished %d epochs' % self.config.SAVE_EVERY_EPOCHS)
                results, precision, recall, f1 = self.evaluate()
                print('Accuracy after %d epochs: %f' % (epoch_num, results))
                print('After ' + str(epoch_num) + ' epochs: Precision: ' + str(precision) + ', recall: ' + str(
                    recall) + ', F1: ' + str(f1))
                if f1 > best_f1:
                    best_f1 = f1
                    best_f1_precision = precision
                    best_f1_recall = recall
                    best_epoch = epoch_num
                    epochs_no_improve = 0
                    save_target = self.config.SAVE_PATH + '_iter' + str(epoch_num)
                    self.save_model(self.sess, save_target)
                    print('Saved after %d epochs in: %s' % (epoch_num, save_target))
                else:
                    epochs_no_improve += self.config.SAVE_EVERY_EPOCHS
                    if epochs_no_improve >= self.config.PATIENCE:
                        print('Not improved for %d epochs, stopping training' % self.config.PATIENCE)
                        print('Best scores - epoch %d: ' % best_epoch)
                        print('Precision: %f, recall: %f, F1: %f' % (best_f1_precision, best_f1_recall, best_f1))
                        return

        if self.config.SAVE_PATH:
            self.save_model(self.sess, self.config.SAVE_PATH + '.final')
            print('Model saved in file: %s' % self.config.SAVE_PATH)

        elapsed = int(time.time() - start_time)
        print("Training time: %sh%sm%ss\n" % ((elapsed // 60 // 60), (elapsed // 60) % 60, elapsed % 60))

    def trace(self, sum_loss, batch_num, multi_batch_start_time):
        multi_batch_elapsed = time.time() - multi_batch_start_time
        avg_loss = sum_loss / (self.num_batches_to_log * self.config.BATCH_SIZE)
        print('Average loss at batch %d: %f, \tthroughput: %d samples/sec' % (batch_num, avg_loss,
                                                                              self.config.BATCH_SIZE * self.num_batches_to_log / (
                                                                                  multi_batch_elapsed if multi_batch_elapsed > 0 else 1)))

    def evaluate(self, release=False):
        eval_start_time = time.time()
        if self.eval_queue is None:
            self.eval_queue = reader.Reader(subtoken_to_index=self.subtoken_to_index,
                                            node_to_index=self.node_to_index,
                                            target_to_index=self.target_to_index,
                                            config=self.config, is_evaluating=True)
            reader_output = self.eval_queue.get_output()
            self.eval_predicted_indices_op, self.eval_topk_values, _, _ = \
                self.build_test_graph(reader_output)
            self.eval_true_target_strings_op = reader_output[reader.TARGET_STRING_KEY]
            self.saver = tf.train.Saver(max_to_keep=10)

        if self.config.LOAD_PATH and not self.config.TRAIN_PATH:
            self.initialize_session_variables(self.sess)
            self.load_model(self.sess)
            if release:
                release_name = self.config.LOAD_PATH + '.release'
                print('Releasing model, output model: %s' % release_name )
                self.saver.save(self.sess, release_name)
                return None
        model_dirname = os.path.dirname(self.config.SAVE_PATH if self.config.SAVE_PATH else self.config.LOAD_PATH)
        ref_file_name = model_dirname + '/ref.txt'
        predicted_file_name = model_dirname + '/pred.txt'
        if not os.path.exists(model_dirname):
            os.makedirs(model_dirname)
        
        with open(model_dirname + '/log.txt', 'w') as output_file, open(ref_file_name, 'w') as ref_file, open(predicted_file_name,
                                                                                             'w') as pred_file:
            num_correct_predictions = 0
            total_predictions = 0
            total_prediction_batches = 0
            true_positive, false_positive, false_negative = 0, 0, 0
            self.eval_queue.reset(self.sess)
            start_time = time.time()

            try:
                while True:
                    predicted_indices, true_target_strings, top_values = self.sess.run(
                        [self.eval_predicted_indices_op, self.eval_true_target_strings_op, self.eval_topk_values],
                    )
                    true_target_strings = Common.binary_to_string_list(true_target_strings)
                    if self.config.BEAM_WIDTH > 0:
                        predicted_strings = [[self.index_to_target[i] for i in predicted_indices_batch]
                                             for predicted_indices_batch in predicted_indices]
                    else:
                        predicted_strings = [[self.index_to_target[i] for i in predicted_indices_batch]
                                             for predicted_indices_batch in predicted_indices]
                    ref_file.write(
                        '\n'.join(
                            [name.replace(Common.internal_delimiter, ' ') for name in true_target_strings]) + '\n')
                    pred_file.write('\n'.join(
                        [' '.join(Common.filter_impossible_names(words)) for words in predicted_strings]) + '\n')

                    num_correct_predictions = self.update_correct_predictions(num_correct_predictions, output_file,
                                                                              zip(true_target_strings,
                                                                                  predicted_strings))
                    true_positive, false_positive, false_negative = self.update_per_subtoken_statistics(
                        zip(true_target_strings, predicted_strings),
                        true_positive, false_positive, false_negative)

                    total_predictions += len(true_target_strings)
                    total_prediction_batches += 1
                    if total_prediction_batches % self.num_batches_to_log == 0:
                        elapsed = time.time() - start_time
                        self.trace_evaluation(output_file, num_correct_predictions, total_predictions, elapsed)
            except tf.errors.OutOfRangeError:
                pass

            print('Done testing, epoch reached')
            output_file.write(str(num_correct_predictions / total_predictions) + '\n')
            # Common.compute_bleu(ref_file_name, predicted_file_name)

        elapsed = int(time.time() - eval_start_time)
        precision, recall, f1 = self.calculate_results(true_positive, false_positive, false_negative)
        print("Evaluation time: %sh%sm%ss" % ((elapsed // 60 // 60), (elapsed // 60) % 60, elapsed % 60))
        return num_correct_predictions / total_predictions, precision, recall, f1

    def update_correct_predictions(self, num_correct_predictions, output_file, results):
        for original_name, predicted_suggestions in results:  # top_words: (num_targets, topk)
            predicted = predicted_suggestions  # [0]
            original_name_parts = original_name.split(Common.internal_delimiter)
            filtered_original = Common.filter_impossible_names(original_name_parts)
            filtered_predicted_parts = Common.filter_impossible_names(predicted)
            output_file.write('Original: ' + Common.internal_delimiter.join(original_name_parts) +
                              ' , predicted 1st: ' + Common.internal_delimiter.join(
                [target for target in predicted]) + '\n')
            if filtered_original == filtered_predicted_parts or Common.unique(filtered_original) == Common.unique(
                    filtered_predicted_parts) or ''.join(filtered_original) == ''.join(filtered_predicted_parts):
                num_correct_predictions += 1
        return num_correct_predictions

    def update_per_subtoken_statistics(self, results, true_positive, false_positive, false_negative):
        for original_name, predicted_suggestions in results:  # top_words: (num_target_parts, topk)
            predicted = predicted_suggestions  # [0]
            filtered_predicted_names = Common.filter_impossible_names(predicted)
            filtered_original_subtokens = Common.filter_impossible_names(original_name.split(Common.internal_delimiter))
            # if len(filtered_predicted_names) > 0 and len(filtered_predicted_names[0]) > 0:

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

    def print_hyperparams(self):
        print('Training batch size:\t', self.config.BATCH_SIZE)
        print('Training path:\t\t\t', self.config.TRAIN_PATH)
        print('Taking max contexts from each example:\t', self.config.MAX_CONTEXTS)
        print('Random path sampling:\t', self.config.RANDOM_CONTEXTS)
        print('Embedding size:\t\t\t', self.config.EMBEDDINGS_SIZE)
        if self.config.BIRNN:
            print('Using BiLSTMs, each of size:\t', self.config.RNN_SIZE // 2)
        else:
            print('Uni-directional LSTM of size:\t', self.config.RNN_SIZE)
        print('Decoder size:\t\t\t', self.config.DECODER_SIZE)
        print('Decoder layers:\t\t\t', self.config.NUM_DECODER_LAYERS)
        print('Max path lengths:\t\t', self.config.MAX_PATH_LENGTH)
        print('Max subtokens in a token:\t', self.config.MAX_NAME_PARTS)
        print('Max target length:\t\t', self.config.MAX_TARGET_PARTS)
        print('Embeddings dropout keep_prob:\t', self.config.EMBEDDINGS_DROPOUT_KEEP_PROB)
        print('LSTM dropout keep_prob:\t', self.config.RNN_DROPOUT_KEEP_PROB)
        print('============================')

    @staticmethod
    def calculate_results(true_positive, false_positive, false_negative):
        if true_positive + false_positive > 0:
            precision = true_positive / (true_positive + false_positive)
        else:
            precision = 0
        if true_positive + false_negative > 0:
            recall = true_positive / (true_positive + false_negative)
        else:
            recall = 0
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
        target_index = input_tensors[reader.TARGET_INDEX_KEY]
        target_lengths = input_tensors[reader.TARGET_LENGTH_KEY]
        path_source_indices = input_tensors[reader.PATH_SOURCE_INDICES_KEY]
        node_indices = input_tensors[reader.NODE_INDICES_KEY]
        path_target_indices = input_tensors[reader.PATH_TARGET_INDICES_KEY]
        valid_context_mask = input_tensors[reader.VALID_CONTEXT_MASK_KEY]
        path_source_lengths = input_tensors[reader.PATH_SOURCE_LENGTHS_KEY]
        path_lengths = input_tensors[reader.PATH_LENGTHS_KEY]
        path_target_lengths = input_tensors[reader.PATH_TARGET_LENGTHS_KEY]

        with tf.variable_scope('model'):
            subtoken_vocab = tf.get_variable('SUBTOKENS_VOCAB',
                                             shape=(self.subtoken_vocab_size, self.config.EMBEDDINGS_SIZE),
                                             dtype=tf.float32,
                                             initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                                                                        mode='FAN_OUT',
                                                                                                        uniform=True))
            target_words_vocab = tf.get_variable('TARGET_WORDS_VOCAB',
                                                 shape=(self.target_vocab_size, self.config.EMBEDDINGS_SIZE),
                                                 dtype=tf.float32,
                                                 initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                                                                            mode='FAN_OUT',
                                                                                                            uniform=True))
            nodes_vocab = tf.get_variable('NODES_VOCAB', shape=(self.nodes_vocab_size, self.config.EMBEDDINGS_SIZE),
                                          dtype=tf.float32,
                                          initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                                                                     mode='FAN_OUT',
                                                                                                     uniform=True))
            # (batch, max_contexts, decoder_size)
            batched_contexts = self.compute_contexts(subtoken_vocab=subtoken_vocab, nodes_vocab=nodes_vocab,
                                                     source_input=path_source_indices, nodes_input=node_indices,
                                                     target_input=path_target_indices,
                                                     valid_mask=valid_context_mask,
                                                     path_source_lengths=path_source_lengths,
                                                     path_lengths=path_lengths, path_target_lengths=path_target_lengths)

            batch_size = tf.shape(target_index)[0]
            outputs, final_states = self.decode_outputs(target_words_vocab=target_words_vocab,
                                                        target_input=target_index, batch_size=batch_size,
                                                        batched_contexts=batched_contexts,
                                                        valid_mask=valid_context_mask)
            self.saver = tf.train.Saver(max_to_keep=10)
            logits = outputs.rnn_output  # (batch, max_output_length, dim * 2 + rnn_size)

            crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_index, logits=logits)
            target_words_nonzero = tf.sequence_mask(target_lengths + 1,
                                                    maxlen=self.config.MAX_TARGET_PARTS + 1, dtype=tf.float32)
            loss = tf.reduce_sum(crossent * target_words_nonzero) / tf.to_float(batch_size)

            if self.config.USE_NESTEROV:
                step = tf.Variable(0, trainable=False)
                learning_rate = tf.train.exponential_decay(0.01, step * self.config.BATCH_SIZE,
                                                           self.num_training_examples,
                                                           0.95, staircase=True)
                optimizer = tf.train.MomentumOptimizer(learning_rate, 0.95, use_nesterov=True)
                train_op = optimizer.minimize(loss, global_step=step)
            else:
                optimizer = tf.train.AdamOptimizer()
                train_op = optimizer.minimize(loss)

        return train_op, loss

    def decode_outputs(self, target_words_vocab, target_input, batch_size, batched_contexts, valid_mask,
                       is_evaluating=False):
        num_contexts_per_example = tf.count_nonzero(valid_mask, axis=-1)

        start_fill = tf.fill([batch_size],
                             self.target_to_index[Common.SOS])  # (batch, )
        decoder_cell = tf.nn.rnn_cell.MultiRNNCell([
            tf.nn.rnn_cell.LSTMCell(self.config.DECODER_SIZE) for _ in range(self.config.NUM_DECODER_LAYERS)
        ])
        contexts_sum = tf.reduce_sum(batched_contexts * tf.expand_dims(valid_mask, -1),
                                     axis=1)  # (batch_size, dim * 2 + rnn_size)
        contexts_average = tf.divide(contexts_sum, tf.to_float(tf.expand_dims(num_contexts_per_example, -1)))
        fake_encoder_state = tuple(tf.nn.rnn_cell.LSTMStateTuple(contexts_average, contexts_average) for _ in range(self.config.NUM_DECODER_LAYERS))
        projection_layer = tf.layers.Dense(self.target_vocab_size, use_bias=False)
        if is_evaluating and self.config.BEAM_WIDTH > 0:
            batched_contexts = tf.contrib.seq2seq.tile_batch(batched_contexts, multiplier=self.config.BEAM_WIDTH)
            num_contexts_per_example = tf.contrib.seq2seq.tile_batch(num_contexts_per_example,
                                                                     multiplier=self.config.BEAM_WIDTH)
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(
            num_units=self.config.DECODER_SIZE,
            memory=batched_contexts,
            memory_sequence_length=num_contexts_per_example,
        )
        # TF doesn't support beam search with alignment history
        should_save_alignment_history = is_evaluating and self.config.BEAM_WIDTH == 0
        decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,
                                                           attention_layer_size=self.config.DECODER_SIZE,
                                                           alignment_history=should_save_alignment_history)
        if is_evaluating:
            if self.config.BEAM_WIDTH > 0:
                decoder_initial_state = decoder_cell.zero_state(dtype=tf.float32,
                                                                batch_size=batch_size * self.config.BEAM_WIDTH)
                decoder_initial_state = decoder_initial_state.clone(
                    cell_state=tf.contrib.seq2seq.tile_batch(fake_encoder_state, multiplier=self.config.BEAM_WIDTH))
                decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                    cell=decoder_cell,
                    embedding=target_words_vocab,
                    start_tokens=start_fill,
                    end_token=self.target_to_index[Common.PAD],
                    initial_state=decoder_initial_state,
                    beam_width=self.config.BEAM_WIDTH,
                    output_layer=projection_layer,
                    length_penalty_weight=0.0)
            else:
                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(target_words_vocab, start_fill, 0)
                initial_state = decoder_cell.zero_state(batch_size, tf.float32).clone(cell_state=fake_encoder_state)
                decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell, helper=helper, initial_state=initial_state,
                                                          output_layer=projection_layer)

        else:
            decoder_cell = tf.nn.rnn_cell.DropoutWrapper(decoder_cell,
                                                         output_keep_prob=self.config.RNN_DROPOUT_KEEP_PROB)
            target_words_embedding = tf.nn.embedding_lookup(target_words_vocab,
                                                            tf.concat([tf.expand_dims(start_fill, -1), target_input],
                                                                      axis=-1))  # (batch, max_target_parts, dim * 2 + rnn_size)
            helper = tf.contrib.seq2seq.TrainingHelper(inputs=target_words_embedding,
                                                       sequence_length=tf.ones([batch_size], dtype=tf.int32) * (
                                                           self.config.MAX_TARGET_PARTS + 1))

            initial_state = decoder_cell.zero_state(batch_size, tf.float32).clone(cell_state=fake_encoder_state)

            decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell, helper=helper, initial_state=initial_state,
                                                      output_layer=projection_layer)
        outputs, final_states, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                                                          maximum_iterations=self.config.MAX_TARGET_PARTS + 1)
        return outputs, final_states

    def calculate_path_abstraction(self, path_embed, path_lengths, valid_contexts_mask, is_evaluating=False):
        # return self.path_max_pool(is_evaluating, path_embed, path_lengths, valid_contexts_mask)
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
                rnn_cell_fw = tf.nn.rnn_cell.DropoutWrapper(rnn_cell_fw,
                                                            output_keep_prob=self.config.RNN_DROPOUT_KEEP_PROB)
                rnn_cell_bw = tf.nn.rnn_cell.DropoutWrapper(rnn_cell_bw,
                                                            output_keep_prob=self.config.RNN_DROPOUT_KEEP_PROB)
            _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=rnn_cell_fw,
                cell_bw=rnn_cell_bw,
                inputs=flat_paths,
                dtype=tf.float32,
                sequence_length=lengths)
            final_rnn_state = tf.concat([state_fw.h, state_bw.h], axis=-1)  # (batch * max_contexts, rnn_size)  
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

        return tf.reshape(final_rnn_state,
                          shape=[-1, max_contexts, self.config.RNN_SIZE])  # (batch, max_contexts, rnn_size)

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

    def compute_contexts(self, subtoken_vocab, nodes_vocab, source_input, nodes_input,
                         target_input, valid_mask, path_source_lengths, path_lengths, path_target_lengths,
                         is_evaluating=False):
        max_contexts = tf.shape(source_input)[1]

        source_word_embed = tf.nn.embedding_lookup(params=subtoken_vocab,
                                                   ids=source_input)  # (batch, max_contexts, max_name_parts, dim)
        path_embed = tf.nn.embedding_lookup(params=nodes_vocab,
                                            ids=nodes_input)  # (batch, max_contexts, max_path_length+1, dim)
        target_word_embed = tf.nn.embedding_lookup(params=subtoken_vocab,
                                                   ids=target_input)  # (batch, max_contexts, max_name_parts, dim)

        # clipped_source_lengths = tf.maximum(1, source_lengths)
        # clipped_target_lengths = tf.maximum(1, target_lengths)

        source_word_mask = tf.expand_dims(
            tf.sequence_mask(path_source_lengths, maxlen=self.config.MAX_NAME_PARTS, dtype=tf.float32),
            -1)  # (batch, max_contexts, max_name_parts, 1)
        target_word_mask = tf.expand_dims(
            tf.sequence_mask(path_target_lengths, maxlen=self.config.MAX_NAME_PARTS, dtype=tf.float32),
            -1)  # (batch, max_contexts, max_name_parts, 1)

        source_words_sum = tf.reduce_sum(source_word_embed * source_word_mask,
                                         axis=2)  # (batch, max_contexts, dim)        
        path_nodes_aggregation = self.calculate_path_abstraction(path_embed, path_lengths, valid_mask,
                                                                 is_evaluating)  # (batch, max_contexts, rnn_size)
        target_words_sum = tf.reduce_sum(target_word_embed * target_word_mask, axis=2)  # (batch, max_contexts, dim)

        context_embed = tf.concat([source_words_sum, path_nodes_aggregation, target_words_sum],
                                  axis=-1)  # (batch, max_contexts, dim * 2 + rnn_size)
        if not is_evaluating:
            context_embed = tf.nn.dropout(context_embed, self.config.EMBEDDINGS_DROPOUT_KEEP_PROB)

        flat_embed = tf.reshape(context_embed, [-1,
                                                self.config.EMBEDDINGS_SIZE * 2 + self.config.RNN_SIZE])  # (batch * max_contexts, dim * 2 + rnn_size)
        transform_param = tf.get_variable('TRANSFORM',
                                          shape=(self.config.EMBEDDINGS_SIZE * 2 + self.config.RNN_SIZE,
                                                 self.config.DECODER_SIZE),
                                          dtype=tf.float32)

        flat_embed = tf.tanh(tf.matmul(flat_embed, transform_param))  # (batch * max_contexts, dim)

        batched_embed = tf.reshape(flat_embed,
                                   shape=[-1, max_contexts,
                                          self.config.DECODER_SIZE])  # (batch, max_contexts, decoder_size)

        return batched_embed

    def build_test_graph(self, input_tensors):
        target_index = input_tensors[reader.TARGET_INDEX_KEY]
        path_source_indices = input_tensors[reader.PATH_SOURCE_INDICES_KEY]
        node_indices = input_tensors[reader.NODE_INDICES_KEY]
        path_target_indices = input_tensors[reader.PATH_TARGET_INDICES_KEY]
        valid_mask = input_tensors[reader.VALID_CONTEXT_MASK_KEY]
        path_source_lengths = input_tensors[reader.PATH_SOURCE_LENGTHS_KEY]
        path_lengths = input_tensors[reader.PATH_LENGTHS_KEY]
        path_target_lengths = input_tensors[reader.PATH_TARGET_LENGTHS_KEY]

        with tf.variable_scope('model', reuse=self.get_should_reuse_variables()):
            subtoken_vocab = tf.get_variable('SUBTOKENS_VOCAB',
                                             shape=(self.subtoken_vocab_size, self.config.EMBEDDINGS_SIZE),
                                             dtype=tf.float32, trainable=False)
            target_words_vocab = tf.get_variable('TARGET_WORDS_VOCAB',
                                                 shape=(self.target_vocab_size, self.config.EMBEDDINGS_SIZE),
                                                 dtype=tf.float32, trainable=False)
            nodes_vocab = tf.get_variable('NODES_VOCAB',
                                          shape=(self.nodes_vocab_size, self.config.EMBEDDINGS_SIZE),
                                          dtype=tf.float32, trainable=False)

            batched_contexts = self.compute_contexts(subtoken_vocab=subtoken_vocab, nodes_vocab=nodes_vocab,
                                                     source_input=path_source_indices, nodes_input=node_indices,
                                                     target_input=path_target_indices,
                                                     valid_mask=valid_mask,
                                                     path_source_lengths=path_source_lengths,
                                                     path_lengths=path_lengths, path_target_lengths=path_target_lengths,
                                                     is_evaluating=True)

            outputs, final_states = self.decode_outputs(target_words_vocab=target_words_vocab,
                                                        target_input=target_index, batch_size=tf.shape(target_index)[0],
                                                        batched_contexts=batched_contexts, valid_mask=valid_mask,
                                                        is_evaluating=True)

        if self.config.BEAM_WIDTH > 0:
            predicted_indices = outputs.predicted_ids
            topk_values = outputs.beam_search_decoder_output.scores
            attention_weights = [tf.no_op()]
        else:
            predicted_indices = outputs.sample_id
            topk_values = tf.constant(1, shape=(1, 1), dtype=tf.float32)
            attention_weights = tf.squeeze(final_states.alignment_history.stack(), 1)

        return predicted_indices, topk_values, target_index, attention_weights

    def predict(self, predict_data_lines):
        if self.predict_queue is None:
            self.predict_queue = reader.Reader(subtoken_to_index=self.subtoken_to_index,
                                               node_to_index=self.node_to_index,
                                               target_to_index=self.target_to_index,
                                               config=self.config, is_evaluating=True)
            self.predict_placeholder = tf.placeholder(tf.string)
            reader_output = self.predict_queue.process_from_placeholder(self.predict_placeholder)
            reader_output = {key:tf.expand_dims(tensor, 0) for key,tensor in reader_output.items()}
            self.predict_top_indices_op, self.predict_top_scores_op, _, self.attention_weights_op = \
                self.build_test_graph(reader_output)
            self.predict_source_string = reader_output[reader.PATH_SOURCE_STRINGS_KEY]
            self.predict_path_string = reader_output[reader.PATH_STRINGS_KEY]
            self.predict_path_target_string = reader_output[reader.PATH_TARGET_STRINGS_KEY]
            self.predict_target_strings_op = reader_output[reader.TARGET_STRING_KEY]

            self.initialize_session_variables(self.sess)
            self.saver = tf.train.Saver()
            self.load_model(self.sess)

        results = []
        for line in predict_data_lines:
            predicted_indices, top_scores, true_target_strings, attention_weights, path_source_string, path_strings, path_target_string = self.sess.run(
                [self.predict_top_indices_op, self.predict_top_scores_op, self.predict_target_strings_op,
                 self.attention_weights_op,
                 self.predict_source_string, self.predict_path_string, self.predict_path_target_string],
                feed_dict={self.predict_placeholder: line})

            top_scores = np.squeeze(top_scores, axis=0)
            path_source_string = path_source_string.reshape((-1))
            path_strings = path_strings.reshape((-1))
            path_target_string = path_target_string.reshape((-1))
            predicted_indices = np.squeeze(predicted_indices, axis=0)
            true_target_strings = Common.binary_to_string(true_target_strings[0])

            if self.config.BEAM_WIDTH > 0:
                predicted_strings = [[self.index_to_target[sugg] for sugg in timestep] 
                                      for timestep in predicted_indices] # (target_length, top-k)  
                predicted_strings = [list(map(list, zip(*batch))) for batch in
                                     predicted_strings]  # (top-k, target_length)
                top_scores = [np.exp(np.sum(s, 0)) for s in top_scores]
            else:
                predicted_strings = [self.index_to_target[idx] 
                                      for idx in predicted_indices] # (batch, target_length)  

            attention_per_path = None
            if self.config.BEAM_WIDTH == 0:
                attention_per_path = self.get_attention_per_path(path_source_string, path_strings, path_target_string,
                                                                 attention_weights)

            results.append((true_target_strings, predicted_strings, top_scores, attention_per_path))
        return results

    def get_attention_per_path(self, source_strings, path_strings, target_strings, attention_weights):
        # attention_weights # (time, contexts)
        results = []
        for time_step in attention_weights:
            attention_per_context = {}
            for source, path, target, weight in zip(source_strings, path_strings, target_strings, time_step):
                string_triplet = (
                    Common.binary_to_string(source), Common.binary_to_string(path), Common.binary_to_string(target))
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
        words_placeholder = tf.placeholder(tf.float32, shape=(self.subtoken_vocab_size, self.config.EMBEDDINGS_SIZE))
        words_vocab_init = words_vocab_variable.assign(words_placeholder)
        words_attention_placeholder = tf.placeholder(tf.float32,
                                                     shape=(self.subtoken_vocab_size, self.config.EMBEDDINGS_SIZE))
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

    def save_model(self, sess, path):
        dirname = os.path.dirname(path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        self.saver.save(sess, path)

        dictionaries_path = path + '.dict'
        with open(dictionaries_path, 'wb') as file:
            pickle.dump(self.subtoken_to_index, file)
            pickle.dump(self.index_to_subtoken, file)
            pickle.dump(self.subtoken_vocab_size, file)

            pickle.dump(self.target_to_index, file)
            pickle.dump(self.index_to_target, file)
            pickle.dump(self.target_vocab_size, file)

            pickle.dump(self.node_to_index, file)
            pickle.dump(self.index_to_node, file)
            pickle.dump(self.nodes_vocab_size, file)

            pickle.dump(self.num_training_examples, file)
            pickle.dump(self.config, file)

    def load_model(self, sess):
        if not sess is None:
            self.saver.restore(sess, self.config.LOAD_PATH)
            print('Done loading model')
        with open(self.config.LOAD_PATH + '.dict', 'rb') as file:
            if self.subtoken_to_index is not None:
                return
            print('Loading dictionaries from: ' + self.config.LOAD_PATH)
            self.subtoken_to_index = pickle.load(file)
            self.index_to_subtoken = pickle.load(file)
            self.subtoken_vocab_size = pickle.load(file)

            self.target_to_index = pickle.load(file)
            self.index_to_target = pickle.load(file)
            self.target_vocab_size = pickle.load(file)

            # self.final_paths = pickle.load(file)
            self.node_to_index = pickle.load(file)
            self.index_to_node = pickle.load(file)
            self.nodes_vocab_size = pickle.load(file)

            self.num_training_examples = pickle.load(file)
            saved_config = pickle.load(file)
            self.config.take_model_hyperparams_from(saved_config)
            print('Done loading dictionaries')

    @staticmethod
    def initialize_session_variables(sess):
        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()))

    def get_should_reuse_variables(self):
        if self.config.TRAIN_PATH:
            return True
        else:
            return None
