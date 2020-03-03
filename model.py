import tensorflow as tf
import tensorflow_addons as tfa

import reader
from common import Common


class Model(tf.Module):
    def __init__(self, config, subtoken_vocab_size, target_vocab_size, nodes_vocab_size, is_training):
        super().__init__()
        self.config = config
        self.subtoken_vocab_shape = (subtoken_vocab_size, self.config.EMBEDDINGS_SIZE)
        self.target_vocab_shape = (target_vocab_size, self.config.EMBEDDINGS_SIZE)
        self.nodes_vocab_shape = (nodes_vocab_size, self.config.EMBEDDINGS_SIZE)
        self.is_training = is_training

        initializer = tf.initializers.VarianceScaling(scale=1.0,
                                                      mode='fan_out',
                                                      distribution='uniform')

        self.subtoken_vocab = tf.Variable(name='SUBTOKENS_VOCAB',
                                          shape=self.subtoken_vocab_shape,
                                          dtype=tf.float32,
                                          trainable=is_training,
                                          initial_value=initializer(self.subtoken_vocab_shape))

        self.target_words_vocab = tf.Variable(name='TARGET_WORDS_VOCAB',
                                              shape=self.target_vocab_shape,
                                              dtype=tf.float32,
                                              trainable=is_training,
                                              initial_value=initializer(self.target_vocab_shape))

        self.nodes_vocab = tf.Variable(name='NODES_VOCAB',
                                       shape=self.nodes_vocab_shape,
                                       dtype=tf.float32,
                                       trainable=is_training,
                                       initial_value=initializer(self.nodes_vocab_shape))

        if self.config.BIRNN:
            rnn_cell_fw = tf.keras.layers.LSTMCell(self.config.RNN_SIZE // 2, dropout=self.config.RNN_DROPOUT_KEEP_PROB)
            rnn_cell_bw = tf.keras.layers.LSTMCell(self.config.RNN_SIZE // 2, dropout=self.config.RNN_DROPOUT_KEEP_PROB)
            self.rnn = tf.keras.layers.Bidirectional(layer=tf.keras.layers.RNN(rnn_cell_fw, return_state=True),
                                                     backward_layer=tf.keras.layers.RNN(rnn_cell_bw, go_backwards=True,
                                                                                        return_state=True),
                                                     merge_mode="concat",
                                                     dtype=tf.float32)
        else:
            rnn_cell = tf.keras.layers.LSTMCell(self.config.RNN_SIZE, dropout=self.config.RNN_DROPOUT_KEEP_PROB)
            self.rnn = tf.keras.layers.RNN(rnn_cell, dtype=tf.float32, return_state=True)

        self.embed_dense_layer = tf.keras.layers.Dense(units=self.config.DECODER_SIZE,
                                                       activation=tf.nn.tanh, use_bias=False)

        decoder_cells = [tf.keras.layers.LSTMCell(self.config.DECODER_SIZE) for _ in
                         range(self.config.NUM_DECODER_LAYERS)]
        self.decoder_cell = tf.keras.layers.StackedRNNCells(decoder_cells)

        if self.is_training:
            self.decoder_cell = tf.nn.RNNCellDropoutWrapper(self.decoder_cell,
                                                            output_keep_prob=self.config.RNN_DROPOUT_KEEP_PROB)

        self.projection_layer = tf.keras.layers.Dense(units=self.target_vocab_shape[0], use_bias=False)

    @tf.function
    def __call__(self, input_tensors):
        target_index = input_tensors[reader.TARGET_INDEX_KEY]
        path_source_indices = input_tensors[reader.PATH_SOURCE_INDICES_KEY]
        node_indices = input_tensors[reader.NODE_INDICES_KEY]
        path_target_indices = input_tensors[reader.PATH_TARGET_INDICES_KEY]
        valid_context_mask = input_tensors[reader.VALID_CONTEXT_MASK_KEY]
        path_source_lengths = input_tensors[reader.PATH_SOURCE_LENGTHS_KEY]
        path_lengths = input_tensors[reader.PATH_LENGTHS_KEY]
        path_target_lengths = input_tensors[reader.PATH_TARGET_LENGTHS_KEY]

        batched_contexts = self.compute_contexts(subtoken_vocab=self.subtoken_vocab,
                                                 nodes_vocab=self.nodes_vocab,
                                                 source_input=path_source_indices,
                                                 nodes_input=node_indices,
                                                 target_input=path_target_indices,
                                                 valid_mask=valid_context_mask,
                                                 path_source_lengths=path_source_lengths,
                                                 path_lengths=path_lengths,
                                                 path_target_lengths=path_target_lengths)

        batch_size = tf.shape(target_index)[0]
        outputs, final_states = self.decode_outputs(target_words_vocab=self.target_words_vocab,
                                                    target_input=target_index,
                                                    batch_size=batch_size,
                                                    batched_contexts=batched_contexts,
                                                    valid_mask=valid_context_mask)

        return outputs, final_states

    def path_rnn_last_state(self, path_embed, path_lengths, valid_contexts_mask):
        # path_embed:           (batch, max_contexts, max_path_length+1, dim)
        # path_length:          (batch, max_contexts)
        # valid_contexts_mask:  (batch, max_contexts)
        max_contexts = tf.shape(path_embed)[1]

        # (batch * max_contexts, max_path_length+1, dim)
        flat_paths = tf.reshape(path_embed, shape=[-1, self.config.MAX_PATH_LENGTH,
                                                   self.config.EMBEDDINGS_SIZE])

        # flat_valid_contexts_mask = tf.reshape(valid_contexts_mask, [-1])  # (batch * max_contexts)
        flat_valid_contexts_mask = tf.expand_dims(
            tf.sequence_mask(tf.reshape(path_lengths, [-1]), maxlen=self.config.MAX_PATH_LENGTH,
                             dtype=tf.float32), axis=-1)

        # lengths = tf.multiply(tf.reshape(path_lengths, [-1]),
        #                      tf.cast(flat_valid_contexts_mask, tf.int32))  # (batch * max_contexts)

        # https://github.com/tensorflow/tensorflow/issues/26974
        if self.config.BIRNN:
            res = self.rnn(inputs=flat_paths, mask=flat_valid_contexts_mask,
                           training=self.is_training)
            _, state_fw, _, state_bw, _ = res  # state = [mem, carry]
            final_rnn_state = tf.concat([state_fw, state_bw], axis=-1)  # (batch * max_contexts, rnn_size)
        else:
            _, state, _ = self.rnn(inputs=flat_paths, mask=flat_valid_contexts_mask, training=self.is_training)
            final_rnn_state = state

        return tf.reshape(final_rnn_state,
                          shape=[-1, max_contexts, self.config.RNN_SIZE])  # (batch, max_contexts, rnn_size)

    def compute_contexts(self, subtoken_vocab, nodes_vocab, source_input, nodes_input,
                         target_input, valid_mask, path_source_lengths, path_lengths, path_target_lengths):

        source_word_embed = tf.nn.embedding_lookup(params=subtoken_vocab,
                                                   ids=source_input)  # (batch, max_contexts, max_name_parts, dim)
        path_embed = tf.nn.embedding_lookup(params=nodes_vocab,
                                            ids=nodes_input)  # (batch, max_contexts, max_path_length+1, dim)
        target_word_embed = tf.nn.embedding_lookup(params=subtoken_vocab,
                                                   ids=target_input)  # (batch, max_contexts, max_name_parts, dim)

        source_word_mask = tf.expand_dims(
            tf.sequence_mask(path_source_lengths, maxlen=self.config.MAX_NAME_PARTS, dtype=tf.float32),
            -1)  # (batch, max_contexts, max_name_parts, 1)
        target_word_mask = tf.expand_dims(
            tf.sequence_mask(path_target_lengths, maxlen=self.config.MAX_NAME_PARTS, dtype=tf.float32),
            -1)  # (batch, max_contexts, max_name_parts, 1)

        source_words_sum = tf.reduce_sum(source_word_embed * source_word_mask,
                                         axis=2)  # (batch, max_contexts, dim)
        path_nodes_aggregation = self.path_rnn_last_state(path_embed, path_lengths,
                                                          valid_mask)  # (batch, max_contexts, rnn_size)
        target_words_sum = tf.reduce_sum(target_word_embed * target_word_mask, axis=2)  # (batch, max_contexts, dim)

        context_embed = tf.concat([source_words_sum, path_nodes_aggregation, target_words_sum],
                                  axis=-1)  # (batch, max_contexts, dim * 2 + rnn_size)
        if self.is_training:
            context_embed = tf.nn.dropout(context_embed, self.config.EMBEDDINGS_DROPOUT_KEEP_PROB)

        batched_embed = self.embed_dense_layer(inputs=context_embed)

        return batched_embed

    def decode_outputs(self, target_words_vocab, target_input, batch_size, batched_contexts, valid_mask):
        num_contexts_per_example = tf.math.count_nonzero(valid_mask, axis=-1)

        start_fill = tf.fill([batch_size],
                             self.target_to_index[Common.SOS])  # (batch, )

        contexts_sum = tf.reduce_sum(batched_contexts * tf.expand_dims(valid_mask, -1),
                                     axis=1)  # (batch_size, dim * 2 + rnn_size)
        contexts_average = tf.divide(contexts_sum, tf.cast(tf.expand_dims(num_contexts_per_example, -1), tf.float32))

        # tf.compat.v1.nn.rnn_cell.LSTMStateTuple
        fake_encoder_state = tuple((contexts_average, contexts_average) for _ in
                                   range(self.config.NUM_DECODER_LAYERS))

        if not self.is_training and self.config.BEAM_WIDTH > 0:
            batched_contexts = tfa.seq2seq.tile_batch(batched_contexts, multiplier=self.config.BEAM_WIDTH)

        attention_mechanism = tfa.seq2seq.LuongAttention(
            units=self.config.DECODER_SIZE,
            memory=batched_contexts
        )
        # TF doesn't support beam search with alignment history
        should_save_alignment_history = not self.is_training and self.config.BEAM_WIDTH == 0
        decoder_cell = tfa.seq2seq.AttentionWrapper(self.decoder_cell, attention_mechanism,
                                                    attention_layer_size=self.config.DECODER_SIZE,
                                                    alignment_history=should_save_alignment_history)
        if not self.is_training:
            target_words_embedding = target_words_vocab
            if self.config.BEAM_WIDTH > 0:
                decoder_initial_state = decoder_cell.get_initial_state(dtype=tf.float32,
                                                                       batch_size=batch_size * self.config.BEAM_WIDTH)
                decoder_initial_state = decoder_initial_state.clone(
                    cell_state=tfa.seq2seq.tile_batch(fake_encoder_state, multiplier=self.config.BEAM_WIDTH))
                decoder = tfa.seq2seq.BeamSearchDecoder(
                    cell=decoder_cell,
                    beam_width=self.config.BEAM_WIDTH,
                    output_layer=self.projection_layer,
                    length_penalty_weight=0.0)
            else:
                helper = tfa.seq2seq.GreedyEmbeddingHelper(target_words_vocab, start_fill, 0)
                decoder_initial_state = decoder_cell.get_initial_state(batch_size, tf.float32).clone(
                    cell_state=fake_encoder_state)
                decoder = tfa.seq2seq.BasicDecoder(cell=decoder_cell, helper=helper,
                                                   initial_state=decoder_initial_state,
                                                   output_layer=self.projection_layer)

        else:
            # (batch, max_target_parts, dim * 2 + rnn_size)
            target_words_embedding = tf.nn.embedding_lookup(target_words_vocab,
                                                            tf.concat([tf.expand_dims(start_fill, -1), target_input],
                                                                      axis=-1))
            # helper = tfa.seq2seq.TrainingHelper(inputs=target_words_embedding,
            #                                            sequence_length=tf.ones([batch_size], dtype=tf.int32) * (
            #                                                    self.config.MAX_TARGET_PARTS + 1))

            sampler = tfa.seq2seq.sampler.TrainingSampler()

            decoder_initial_state = decoder_cell.get_initial_state(target_words_embedding, batch_size,
                                                                   tf.float32).clone(
                cell_state=fake_encoder_state)

            decoder = tfa.seq2seq.BasicDecoder(cell=decoder_cell, sampler=sampler, output_layer=self.projection_layer)

            #        outputs, final_states, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder,
            #                                                   maximum_iterations=self.config.MAX_TARGET_PARTS + 1)

        outputs, final_states, final_sequence_lengths = decoder(
            target_words_embedding,
            initial_state=decoder_initial_state,
            start_tokens=start_fill,
            end_token=self.target_to_index[Common.PAD],
            sequence_length=tf.ones([batch_size], dtype=tf.int32) * (self.config.MAX_TARGET_PARTS + 1))
        return outputs, final_states

    # def run(self):
    #     if self.is_training:
    #         target_lengths = input_tensors[reader.TARGET_LENGTH_KEY]
    #
    #         step = tf.Variable(0, trainable=False)
    #
    #         logits = outputs.rnn_output  # (batch, max_output_length, dim * 2 + rnn_size)
    #
    #         crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_index, logits=logits)
    #         target_words_nonzero = tf.sequence_mask(target_lengths + 1,
    #                                                 maxlen=self.config.MAX_TARGET_PARTS + 1, dtype=tf.float32)
    #         loss = tf.reduce_sum(crossent * target_words_nonzero) / tf.to_float(batch_size)
    #
    #         if self.config.USE_MOMENTUM:
    #             learning_rate = tf.train.exponential_decay(0.01, step * self.config.BATCH_SIZE,
    #                                                        self.num_training_examples,
    #                                                        0.95, staircase=True)
    #             optimizer = tf.train.MomentumOptimizer(learning_rate, 0.95, use_nesterov=True)
    #             train_op = optimizer.minimize(loss, global_step=step)
    #         else:
    #             params = tf.trainable_variables()
    #             gradients = tf.gradients(loss, params)
    #             clipped_gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=5)
    #             optimizer = tf.train.AdamOptimizer()
    #             train_op = optimizer.apply_gradients(zip(clipped_gradients, params))
    #
    #         self.saver = tf.train.Saver(max_to_keep=10)
    #
    #         return train_op, loss
    #     else:
    #
    #         if self.config.BEAM_WIDTH > 0:
    #             predicted_indices = outputs.predicted_ids
    #             topk_values = outputs.beam_search_decoder_output.scores
    #             attention_weights = [tf.no_op()]
    #         else:
    #             predicted_indices = outputs.sample_id
    #             topk_values = tf.constant(1, shape=(1, 1), dtype=tf.float32)
    #             attention_weights = tf.squeeze(final_states.alignment_history.stack(), 1)
    #
    #         return predicted_indices, topk_values, target_index, attention_weights
