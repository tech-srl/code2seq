import os
import sys
import time

import _pickle as pickle
import tensorflow as tf
import tqdm
from rouge import FilesRouge

import reader
from model import Model
from results import *


class ModelRunner:
    num_batches_to_log = 100

    def __init__(self, config):
        self.config = config

        if config.LOAD_PATH:
            self.model = None
            self.load_model(self.config.LOAD_PATH)
        else:
            with open('{}.dict.c2s'.format(config.TRAIN_PATH), 'rb') as file:
                subtoken_to_count = pickle.load(file)
                node_to_count = pickle.load(file)
                target_to_count = pickle.load(file)
                max_contexts = pickle.load(file)
                self.num_training_examples = pickle.load(file)
                print('Num training samples: {0}'.format(self.num_training_examples))
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

            self.model = Model(self.config, self.subtoken_vocab_size, self.target_vocab_size, self.nodes_vocab_size,
                               self.target_to_index)

        if self.config.TRAIN_PATH:
            self.train_dataset_reader = reader.Reader(subtoken_to_index=self.subtoken_to_index,
                                                      node_to_index=self.node_to_index,
                                                      target_to_index=self.target_to_index,
                                                      config=self.config,
                                                      is_evaluating=False)
        else:
            self.train_dataset_reader = None

        self.test_dataset_reader = reader.Reader(subtoken_to_index=self.subtoken_to_index,
                                                 node_to_index=self.node_to_index,
                                                 target_to_index=self.target_to_index,
                                                 config=self.config,
                                                 is_evaluating=True)

    def train(self):
        print('Starting training')

        self.print_hyperparams()
        print('Number of trainable params:',
              np.sum([np.prod(v.get_shape().as_list()) for v in self.model.trainable_variables]))

        print('Start training loop...')
        dataset = self.train_dataset_reader.get_dataset()

        if self.config.USE_MOMENTUM:
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=0.01,
                decay_steps=self.num_training_examples,
                decay_rate=0.95
            )
            optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.95, nesterov=True)

        else:
            optimizer = tf.keras.optimizers.Adam()

        checkpoint = None
        checkpoint_manager = None
        if self.config.MODEL_PATH:
            print('Loading model...')
            checkpoint = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, model=self.model)
            checkpoint_manager = tf.train.CheckpointManager(checkpoint, self.config.MODEL_PATH, max_to_keep=3)

            if checkpoint_manager.latest_checkpoint:
                checkpoint.restore(checkpoint_manager.latest_checkpoint)
                print("Restored from {}".format(checkpoint_manager.latest_checkpoint))
            else:
                print("Initializing model from scratch.")

        sum_loss = 0
        batch_num = 0
        epochs_trained = 0
        best_f1 = 0
        best_epoch = 0
        best_f1_precision = 0
        best_f1_recall = 0
        epochs_no_improve = 0
        multi_batch_start_time = time.time()
        start_time = time.time()

        for iteration in range(self.config.NUM_EPOCHS):
            pbar = tqdm.tqdm(total=self.num_training_examples)
            for input_tensors in dataset:
                target_lengths = input_tensors[reader.TARGET_LENGTH_KEY]
                target_index = input_tensors[reader.TARGET_INDEX_KEY]
                batch_size = tf.shape(target_index)[0]
                with tf.GradientTape() as tape:
                    batched_contexts = self.model.run_encoder(input_tensors, is_training=True)
                    outputs, _ = self.model.run_decoder(batched_contexts, input_tensors, is_training=True)

                    logits = outputs.rnn_output  # (batch, max_output_length, dim * 2 + rnn_size)
                    crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_index, logits=logits)
                    target_words_nonzero = tf.sequence_mask(target_lengths + 1,
                                                            maxlen=self.config.MAX_TARGET_PARTS + 1, dtype=tf.float32)
                    loss = tf.reduce_sum(crossent * target_words_nonzero) / tf.cast(batch_size, dtype=tf.float32)

                gradients = tape.gradient(loss, self.model.trainable_variables)
                if self.config.USE_MOMENTUM:
                    clipped_gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=5)
                optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

                sum_loss += loss
                batch_num += 1

                if batch_num % self.num_batches_to_log == 0:
                    self.trace(pbar, sum_loss, batch_num, multi_batch_start_time)
                    sum_loss = 0
                    multi_batch_start_time = time.time()

                pbar.update(self.config.BATCH_SIZE)
                sys.stdout.flush()

            # the end of an epoch
            epochs_trained += 1
            print('Finished {0} epochs'.format(epochs_trained))
            if epochs_trained % self.config.SAVE_EVERY_EPOCHS == 0:
                if self.config.MODEL_PATH:
                    print("Checkpoint saved")
                    checkpoint.step.assign_add(1)
                    checkpoint_manager.save()

            # validate model to calculate metrics or stop training
            results, precision, recall, f1, rouge = self.evaluate()
            if self.config.BEAM_WIDTH == 0:
                print('Accuracy after %d epochs: %.5f' % (epochs_trained, results))
            else:
                print('Accuracy after {} epochs: {}'.format(epochs_trained, results))
            print('After %d epochs: Precision: %.5f, recall: %.5f, F1: %.5f' % (
                epochs_trained, precision, recall, f1))
            print('Rouge: ', rouge)
            if f1 > best_f1:
                best_f1 = f1
                best_f1_precision = precision
                best_f1_recall = recall
                best_epoch = epochs_trained
                epochs_no_improve = 0
            else:
                epochs_no_improve += self.config.SAVE_EVERY_EPOCHS
                if epochs_no_improve >= self.config.PATIENCE:
                    print('Not improved for %d epochs, stopping training' % self.config.PATIENCE)
                    print('Best scores - epoch %d: ' % best_epoch)
                    print('Precision: %.5f, recall: %.5f, F1: %.5f' % (best_f1_precision, best_f1_recall, best_f1))
                    break

        # the end of training
        if self.config.SAVE_PATH:
            self.save_model(self.config.SAVE_PATH)
            print('Model saved into : {0}'.format(self.config.SAVE_PATH))

        elapsed = int(time.time() - start_time)
        print("Training time: %sh%sm%ss\n" % ((elapsed // 60 // 60), (elapsed // 60) % 60, elapsed % 60))

    def evaluate(self):
        if not self.model:
            print('Model is not initialized')
            exit(-1)

        print("Testing...")
        eval_start_time = time.time()

        if self.config.LOAD_PATH and not self.config.TRAIN_PATH:
            model_dirname = os.path.dirname(self.config.LOAD_PATH)
        elif self.config.MODEL_PATH:
            model_dirname = os.path.dirname(self.config.MODEL_PATH)
        else:
            model_dirname = None
            print('Model directory is mossing')
            exit(-1)

        ref_file_name = os.path.join(model_dirname, 'ref.txt')
        predicted_file_name = os.path.join(model_dirname, 'pred.txt')
        if not os.path.exists(model_dirname):
            os.makedirs(model_dirname)

        log_file_name = os.path.join(model_dirname, 'log.txt')
        with open(log_file_name, 'w') as output_file, open(ref_file_name, 'w') as ref_file, open(
                predicted_file_name,
                'w') as pred_file:
            num_correct_predictions = 0 if self.config.BEAM_WIDTH == 0 \
                else np.zeros([self.config.BEAM_WIDTH], dtype=np.int32)
            total_predictions = 0
            total_prediction_batches = 0
            true_positive, false_positive, false_negative = 0, 0, 0
            dataset = self.test_dataset_reader.get_dataset()
            start_time = time.time()

            for input_tensors in dataset:
                true_target_strings = input_tensors[reader.TARGET_STRING_KEY]

                batched_contexts = self.model.run_encoder(input_tensors, is_training=False)
                outputs, final_states = self.model.run_decoder(batched_contexts, input_tensors, is_training=False)

                if self.config.BEAM_WIDTH > 0:
                    predicted_indices = outputs.predicted_ids
                else:
                    predicted_indices = outputs.sample_id

                true_target_strings = Common.binary_to_string_list(true_target_strings.numpy())
                ref_file.write(
                    '\n'.join(
                        [name.replace(Common.internal_delimiter, ' ') for name in true_target_strings]) + '\n')
                if self.config.BEAM_WIDTH > 0:
                    # predicted indices: (batch, time, beam_width)
                    predicted_strings = [[[self.index_to_target[i] for i in timestep] for timestep in example] for
                                         example in predicted_indices]
                    predicted_strings = [list(map(list, zip(*example))) for example in
                                         predicted_strings]  # (batch, top-k, target_length)
                    pred_file.write('\n'.join(
                        [' '.join(Common.filter_impossible_names(words)) for words in predicted_strings[0]]) + '\n')
                else:
                    predicted_strings = [[self.index_to_target[i] for i in example]
                                         for example in predicted_indices.numpy()]
                    pred_file.write('\n'.join(
                        [' '.join(Common.filter_impossible_names(words)) for words in predicted_strings]) + '\n')

                num_correct_predictions = update_correct_predictions(self.config.BEAM_WIDTH, num_correct_predictions,
                                                                     output_file,
                                                                     zip(true_target_strings,
                                                                         predicted_strings))
                true_positive, false_positive, false_negative = update_per_subtoken_statistics(self.config.BEAM_WIDTH,
                                                                                               zip(true_target_strings,
                                                                                                   predicted_strings),
                                                                                               true_positive,
                                                                                               false_positive,
                                                                                               false_negative)

                total_predictions += len(true_target_strings)
                total_prediction_batches += 1
                if total_prediction_batches % self.num_batches_to_log == 0:
                    elapsed = time.time() - start_time
                    trace_evaluation(output_file, num_correct_predictions, total_predictions, elapsed)

            print('Done testing, epoch reached', flush=True)
            output_file.write(str(num_correct_predictions / total_predictions) + '\n')

        elapsed = int(time.time() - eval_start_time)
        precision, recall, f1 = calculate_results(true_positive, false_positive, false_negative)
        files_rouge = FilesRouge(predicted_file_name, ref_file_name)
        rouge = files_rouge.get_scores(avg=True, ignore_empty=True)
        print("Evaluation time: %sh%sm%ss" % ((elapsed // 60 // 60), (elapsed // 60) % 60, elapsed % 60))
        return num_correct_predictions / total_predictions, precision, recall, f1, rouge

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

    def trace(self, pbar, sum_loss, batch_num, multi_batch_start_time):
        multi_batch_elapsed = time.time() - multi_batch_start_time
        avg_loss = sum_loss / self.num_batches_to_log
        msg = 'Average loss at batch {0}: {1}, \tthroughput: {2} samples/sec'. \
            format(batch_num, avg_loss,
                   self.config.BATCH_SIZE * self.num_batches_to_log / (
                       multi_batch_elapsed if multi_batch_elapsed > 0 else 1))
        pbar.set_description(msg)

    def predict(self, predict_data_lines):
        if not self.model:
            print('Model is not initialized')
            exit(-1)

        predict_reader = reader.Reader(subtoken_to_index=self.subtoken_to_index,
                                       node_to_index=self.node_to_index,
                                       target_to_index=self.target_to_index,
                                       config=self.config,
                                       is_evaluating=True)
        results = []
        for line in predict_data_lines:
            input_tensors = predict_reader.process_from_placeholder(line)

            path_source_string = input_tensors[reader.PATH_SOURCE_STRINGS_KEY]
            path_strings = input_tensors[reader.PATH_STRINGS_KEY]
            path_target_string = input_tensors[reader.PATH_TARGET_STRINGS_KEY]
            true_target_strings = input_tensors[reader.TARGET_STRING_KEY]

            batched_contexts = self.model.run_encoder(input_tensors, is_training=False)
            outputs, final_states = self.model.run_decoder(batched_contexts, input_tensors, is_training=False)

            if self.config.BEAM_WIDTH > 0:
                predicted_indices = outputs.predicted_ids
                top_scores = outputs.beam_search_decoder_output.scores
                attention_weights = [tf.no_op()]
            else:
                predicted_indices = outputs.sample_id
                top_scores = tf.constant(1, shape=(1, 1), dtype=tf.float32)
                attention_weights = tf.squeeze(final_states.alignment_history.stack(), 1)

            top_scores = np.squeeze(top_scores.numpy(), axis=0)
            path_source_string = path_source_string.numpy().reshape((-1))
            path_strings = path_strings.numpy().reshape((-1))
            path_target_string = path_target_string.numpy().reshape((-1))
            predicted_indices = np.squeeze(predicted_indices.numpy(), axis=0)
            true_target_strings = Common.binary_to_string(true_target_strings.numpy()[0])

            if self.config.BEAM_WIDTH > 0:
                predicted_strings = [[self.index_to_target[sugg] for sugg in timestep]
                                     for timestep in predicted_indices]  # (target_length, top-k)
                predicted_strings = list(map(list, zip(*predicted_strings)))  # (top-k, target_length)
                top_scores = [np.exp(np.sum(s)) for s in zip(*top_scores)]
            else:
                predicted_strings = [self.index_to_target[idx]
                                     for idx in predicted_indices]  # (batch, target_length)

            attention_per_path = None
            if self.config.BEAM_WIDTH == 0:
                attention_per_path = self.get_attention_per_path(path_source_string, path_strings, path_target_string,
                                                                 attention_weights.numpy())

            results.append((true_target_strings, predicted_strings, top_scores, attention_per_path))
        return results

    @staticmethod
    def get_attention_per_path(source_strings, path_strings, target_strings, attention_weights):
        # attention_weights:  (time, contexts)
        results = []
        for time_step in attention_weights:
            attention_per_context = {}
            for source, path, target, weight in zip(source_strings, path_strings, target_strings, time_step):
                string_triplet = (
                    Common.binary_to_string(source), Common.binary_to_string(path), Common.binary_to_string(target))
                attention_per_context[string_triplet] = weight
            results.append(attention_per_context)
        return results

    def save_model(self, path):
        path_name = os.path.dirname(path)
        if not os.path.exists(path_name):
            os.makedirs(path_name)
        checkpoint = tf.train.Checkpoint(model=self.model)
        checkpoint.save(os.path.join(path_name, 'model'))

        dictionaries_path = os.path.join(path_name, 'model.dict')
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

    def load_model(self, path):
        path_name = os.path.dirname(path)
        if os.path.exists(path_name):
            with open(os.path.join(path_name, 'model.dict'), 'rb') as file:
                self.subtoken_to_index = pickle.load(file)
                self.index_to_subtoken = pickle.load(file)
                self.subtoken_vocab_size = pickle.load(file)

                self.target_to_index = pickle.load(file)
                self.index_to_target = pickle.load(file)
                self.target_vocab_size = pickle.load(file)

                self.node_to_index = pickle.load(file)
                self.index_to_node = pickle.load(file)
                self.nodes_vocab_size = pickle.load(file)

                self.num_training_examples = pickle.load(file)
                saved_config = pickle.load(file)
                self.config.take_model_hyperparams_from(saved_config)

            self.model = Model(self.config, self.subtoken_vocab_size, self.target_vocab_size, self.nodes_vocab_size,
                               self.target_to_index)
            checkpoint = tf.train.Checkpoint(model=self.model)
            status = checkpoint.restore(tf.train.latest_checkpoint(path))
            status.expect_partial()
