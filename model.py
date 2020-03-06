import tensorflow as tf
import tensorflow_addons as tfa

import reader
from common import Common


class Model(tf.Module):
    def __init__(self, config, subtoken_vocab_size, target_vocab_size, nodes_vocab_size, target_to_index):
        super().__init__()
        self.config = config
        self.subtoken_vocab_shape = (subtoken_vocab_size, self.config.EMBEDDINGS_SIZE)
        self.target_vocab_shape = (target_vocab_size, self.config.EMBEDDINGS_SIZE)
        self.nodes_vocab_shape = (nodes_vocab_size, self.config.EMBEDDINGS_SIZE)
        self.target_to_index = target_to_index

        initializer = tf.initializers.VarianceScaling(scale=1.0,
                                                      mode='fan_out',
                                                      distribution='uniform')

        self.subtoken_vocab = tf.Variable(name='SUBTOKENS_VOCAB',
                                          shape=self.subtoken_vocab_shape,
                                          dtype=tf.float32,
                                          initial_value=initializer(self.subtoken_vocab_shape))

        self.target_words_vocab = tf.Variable(name='TARGET_WORDS_VOCAB',
                                              shape=self.target_vocab_shape,
                                              dtype=tf.float32,
                                              initial_value=initializer(self.target_vocab_shape))

        self.nodes_vocab = tf.Variable(name='NODES_VOCAB',
                                       shape=self.nodes_vocab_shape,
                                       dtype=tf.float32,
                                       initial_value=initializer(self.nodes_vocab_shape))

        if self.config.BIRNN:
            rnn_cell_fw = tf.keras.layers.LSTMCell(self.config.RNN_SIZE // 2,
                                                   dropout=1 - self.config.RNN_DROPOUT_KEEP_PROB)
            rnn_cell_bw = tf.keras.layers.LSTMCell(self.config.RNN_SIZE // 2,
                                                   dropout=1 - self.config.RNN_DROPOUT_KEEP_PROB)
            self.rnn = tf.keras.layers.Bidirectional(layer=tf.keras.layers.RNN(rnn_cell_fw, return_state=True),
                                                     backward_layer=tf.keras.layers.RNN(rnn_cell_bw, go_backwards=True,
                                                                                        return_state=True),
                                                     merge_mode="concat",
                                                     dtype=tf.float32)
        else:
            rnn_cell = tf.keras.layers.LSTMCell(self.config.RNN_SIZE, dropout=1 - self.config.RNN_DROPOUT_KEEP_PROB)
            self.rnn = tf.keras.layers.RNN(rnn_cell, dtype=tf.float32, return_state=True)

        self.embed_dense_layer = tf.keras.layers.Dense(units=self.config.DECODER_SIZE,
                                                       activation=tf.nn.tanh, use_bias=False)

        decoder_cells = [
            tf.keras.layers.LSTMCell(self.config.DECODER_SIZE, dropout=1 - self.config.RNN_DROPOUT_KEEP_PROB) for _ in
            range(self.config.NUM_DECODER_LAYERS)]
        self.decoder_cell = tf.keras.layers.StackedRNNCells(decoder_cells)

        self.projection_layer = tf.keras.layers.Dense(units=self.target_vocab_shape[0], use_bias=False)

        self.attention_mechanism = tfa.seq2seq.LuongAttention(units=self.config.DECODER_SIZE)

        should_save_alignment_history = self.config.BEAM_WIDTH == 0
        self.decoder_cell = tfa.seq2seq.AttentionWrapper(self.decoder_cell, self.attention_mechanism,
                                                         attention_layer_size=self.config.DECODER_SIZE,
                                                         alignment_history=should_save_alignment_history)

        if self.config.BEAM_WIDTH > 0:
            self.eval_decoder = tfa.seq2seq.BeamSearchDecoder(
                cell=self.decoder_cell,
                beam_width=self.config.BEAM_WIDTH,
                output_layer=self.projection_layer,
                length_penalty_weight=0.0)
        else:
            greedy_sampler = tfa.seq2seq.GreedyEmbeddingSampler()
            self.eval_decoder = tfa.seq2seq.BasicDecoder(cell=self.decoder_cell, sampler=greedy_sampler,
                                                         output_layer=self.projection_layer)

        sampler = tfa.seq2seq.sampler.TrainingSampler()
        self.train_decoder = tfa.seq2seq.BasicDecoder(cell=self.decoder_cell, sampler=sampler,
                                                      output_layer=self.projection_layer)

    @tf.function
    def run_encoder(self, input_tensors, is_training):
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
                                                 path_target_lengths=path_target_lengths,
                                                 is_training=is_training)
        return batched_contexts

    def setup_attention_memory(self, batched_contexts):
        self.attention_mechanism.setup_memory(memory=batched_contexts)

    # @tf.function
    def run_decoder(self, batched_contexts, input_tensors, is_training):
        target_index = input_tensors[reader.TARGET_INDEX_KEY]
        valid_context_mask = input_tensors[reader.VALID_CONTEXT_MASK_KEY]
        batch_size = tf.shape(target_index)[0]
        outputs, final_states = self.decode_outputs(target_words_vocab=self.target_words_vocab,
                                                    target_input=target_index,
                                                    batch_size=batch_size,
                                                    batched_contexts=batched_contexts,
                                                    valid_mask=valid_context_mask,
                                                    is_training=is_training)

        return outputs, final_states

    def path_rnn_last_state(self, path_embed, path_lengths, valid_contexts_mask, is_training):
        # path_embed:           (batch, max_contexts, max_path_length+1, dim)
        # path_length:          (batch, max_contexts)
        # valid_contexts_mask:  (batch, max_contexts)
        max_contexts = tf.shape(path_embed)[1]

        # (batch * max_contexts, max_path_length+1, dim)
        flat_paths = tf.reshape(path_embed, shape=[-1, self.config.MAX_PATH_LENGTH,
                                                   self.config.EMBEDDINGS_SIZE])

        flat_valid_contexts_mask = tf.expand_dims(
            tf.sequence_mask(tf.reshape(path_lengths, [-1]), maxlen=self.config.MAX_PATH_LENGTH,
                             dtype=tf.float32), axis=-1)

        # https://github.com/tensorflow/tensorflow/issues/26974
        if self.config.BIRNN:
            res = self.rnn(inputs=flat_paths, mask=flat_valid_contexts_mask,
                           training=is_training)
            _, state_fw, _, state_bw, _ = res  # state = [mem, carry]
            final_rnn_state = tf.concat([state_fw, state_bw], axis=-1)  # (batch * max_contexts, rnn_size)
        else:
            _, state, _ = self.rnn(inputs=flat_paths, mask=flat_valid_contexts_mask, training=is_training)
            final_rnn_state = state

        return tf.reshape(final_rnn_state,
                          shape=[-1, max_contexts, self.config.RNN_SIZE])  # (batch, max_contexts, rnn_size)

    def compute_contexts(self, subtoken_vocab, nodes_vocab, source_input, nodes_input,
                         target_input, valid_mask, path_source_lengths, path_lengths, path_target_lengths, is_training):

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
                                                          valid_mask, is_training)  # (batch, max_contexts, rnn_size)
        target_words_sum = tf.reduce_sum(target_word_embed * target_word_mask, axis=2)  # (batch, max_contexts, dim)

        context_embed = tf.concat([source_words_sum, path_nodes_aggregation, target_words_sum],
                                  axis=-1)  # (batch, max_contexts, dim * 2 + rnn_size)
        if is_training:
            context_embed = tf.nn.dropout(context_embed, rate=1 - self.config.EMBEDDINGS_DROPOUT_KEEP_PROB)

        batched_embed = self.embed_dense_layer(inputs=context_embed)

        if not is_training and self.config.BEAM_WIDTH > 0:
            batched_embed = tfa.seq2seq.tile_batch(batched_embed, multiplier=self.config.BEAM_WIDTH)

        return batched_embed

    def decode_outputs(self, target_words_vocab, target_input, batch_size, batched_contexts, valid_mask, is_training):
        num_contexts_per_example = tf.math.count_nonzero(valid_mask, axis=-1)

        start_fill = tf.fill([batch_size],
                             self.target_to_index[Common.SOS])  # (batch, )

        contexts_sum = tf.reduce_sum(batched_contexts * tf.expand_dims(valid_mask, -1),
                                     axis=1)  # (batch_size, dim * 2 + rnn_size)
        contexts_average = tf.divide(contexts_sum, tf.cast(tf.expand_dims(num_contexts_per_example, -1), tf.float32))

        fake_encoder_state = tuple([contexts_average, contexts_average] for _ in
                                   range(self.config.NUM_DECODER_LAYERS))

        if not is_training:
            target_words_embedding = target_words_vocab
            if self.config.BEAM_WIDTH > 0:
                # https://medium.com/@dhirensk/tensorflow-addons-seq2seq-example-using-attention-and-beam-search-9f463b58bc6b
                decoder_initial_state = self.decoder_cell.get_initial_state(dtype=tf.float32,
                                                                            batch_size=batch_size * self.config.BEAM_WIDTH)
                decoder_initial_state = decoder_initial_state.clone(
                    cell_state=tfa.seq2seq.tile_batch(fake_encoder_state, multiplier=self.config.BEAM_WIDTH))
            else:
                decoder_initial_state = self.decoder_cell.get_initial_state(batch_size=batch_size, dtype=tf.float32)
                decoder_initial_state = decoder_initial_state.clone(cell_state=fake_encoder_state)

        else:
            # (batch, max_target_parts, dim * 2 + rnn_size)
            target_words_embedding = tf.nn.embedding_lookup(target_words_vocab,
                                                            tf.concat([tf.expand_dims(start_fill, -1), target_input],
                                                                      axis=-1))

            decoder_initial_state = self.decoder_cell.get_initial_state(batch_size=batch_size,
                                                                        dtype=tf.float32)
            decoder_initial_state = decoder_initial_state.clone(cell_state=fake_encoder_state)

        if is_training:
            outputs, final_states, final_sequence_lengths = self.train_decoder(
                target_words_embedding,
                training=is_training,
                initial_state=decoder_initial_state,
                sequence_length=tf.ones([batch_size], dtype=tf.int32) * (self.config.MAX_TARGET_PARTS + 1))
        else:
            if self.config.BEAM_WIDTH > 0:
                outputs, final_states, final_sequence_lengths = self.eval_decoder(
                    target_words_embedding,
                    training=is_training,
                    initial_state=decoder_initial_state,
                    start_token=start_fill,
                    end_token=self.target_to_index[Common.PAD],
                    sequence_length=tf.ones([batch_size], dtype=tf.int32) * (self.config.MAX_TARGET_PARTS + 1))
            else:
                outputs, final_states, final_sequence_lengths = self.eval_decoder(
                    target_words_embedding,
                    training=is_training,
                    initial_state=decoder_initial_state,
                    start_tokens=start_fill,
                    end_token=0)

        return outputs, final_states
