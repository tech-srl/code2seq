class Config:
    @staticmethod
    def get_default_config(args):
        config = Config()
        config.NUM_EPOCHS = 3000
        config.SAVE_EVERY_EPOCHS = 1
        config.PATIENCE = 10
        config.BATCH_SIZE = 512
        config.TEST_BATCH_SIZE = 256
        config.PREFETCH_NUM_BATCHES = 10
        config.NUM_BATCHING_THREADS = 7
        config.BATCH_QUEUE_SIZE = 30000
        config.TRAIN_PATH = args.data_path
        config.TEST_PATH = args.test_path if args.test_path is not None else ''
        config.DATA_NUM_CONTEXTS = 0
        config.MAX_CONTEXTS = 200
        config.SUBTOKENS_VOCAB_MAX_SIZE = 190000
        config.TARGET_VOCAB_MAX_SIZE = 27000
        config.EMBEDDINGS_SIZE = 128
        config.RNN_SIZE = 128 * 2 # Two LSTMs to embed paths, each of size 128
        config.DECODER_SIZE = 320
        config.NUM_DECODER_LAYERS = 1
        config.SAVE_PATH = args.save_path
        config.LOAD_PATH = args.load_path
        config.MAX_PATH_LENGTH = 8 + 1
        config.MAX_NAME_PARTS = 5
        config.MAX_TARGET_PARTS = 6
        config.EMBEDDINGS_DROPOUT_KEEP_PROB = 0.75
        config.RNN_DROPOUT_KEEP_PROB = 0.5
        config.BIRNN = True
        config.RANDOM_CONTEXTS = True
        config.BEAM_WIDTH = 0
        config.USE_NESTEROV = True
        config.RELEASE = args.release
        return config

    def take_model_hyperparams_from(self, otherConfig):
        self.EMBEDDINGS_SIZE = otherConfig.EMBEDDINGS_SIZE
        self.RNN_SIZE = otherConfig.RNN_SIZE
        self.DECODER_SIZE = otherConfig.DECODER_SIZE
        self.NUM_DECODER_LAYERS = otherConfig.NUM_DECODER_LAYERS
        self.BIRNN = otherConfig.BIRNN
        if self.DATA_NUM_CONTEXTS <= 0:
            self.DATA_NUM_CONTEXTS = otherConfig.DATA_NUM_CONTEXTS

    def __init__(self):
        self.NUM_EPOCHS = 0
        self.SAVE_EVERY_EPOCHS = 0
        self.PATIENCE = 0
        self.BATCH_SIZE = 0
        self.TEST_BATCH_SIZE = 0
        self.PREFETCH_NUM_BATCHES = 0
        self.NUM_BATCHING_THREADS = 0
        self.BATCH_QUEUE_SIZE = 0
        self.TRAIN_PATH = ''
        self.TEST_PATH = ''
        self.DATA_NUM_CONTEXTS = 0
        self.MAX_CONTEXTS = 0
        self.SUBTOKENS_VOCAB_MAX_SIZE = 0
        self.TARGET_VOCAB_MAX_SIZE = 0
        self.EMBEDDINGS_SIZE = 0
        self.RNN_SIZE = 0
        self.DECODER_SIZE = 0
        self.NUM_DECODER_LAYERS = 0
        self.SAVE_PATH = ''
        self.LOAD_PATH = ''
        self.MAX_PATH_LENGTH = 0
        self.MAX_NAME_PARTS = 0
        self.MAX_TARGET_PARTS = 0
        self.EMBEDDINGS_DROPOUT_KEEP_PROB = 0
        self.RNN_DROPOUT_KEEP_PROB = 0
        self.BIRNN = False
        self.RANDOM_CONTEXTS = True
        self.BEAM_WIDTH = 1
        self.USE_NESTEROV = True
        self.RELEASE = False
