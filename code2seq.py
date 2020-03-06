import numpy as np
import tensorflow as tf

from config import Config
from interactive_predict import InteractivePredictor
from modelrunner import ModelRunner
from args import read_args

if __name__ == '__main__':
    args = read_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    if args.debug:
        config = Config.get_debug_config(args)
        tf.config.experimental_run_functions_eagerly(True)
    else:
        config = Config.get_default_config(args)

    print('Created model')
    if config.TRAIN_PATH:
        model = ModelRunner(config)
        model.train()
    # TODO: restore
    # if config.TEST_PATH and not args.data_path:
    #     model = ModelRunner(config, is_training=False)
    #     results, precision, recall, f1, rouge = model.evaluate()
    #     print('Accuracy: ' + str(results))
    #     print('Precision: ' + str(precision) + ', recall: ' + str(recall) + ', F1: ' + str(f1))
    #     print('Rouge: ', rouge)
    # if args.predict:
    #     model = ModelRunner(config, is_training=False)
    #     predictor = InteractivePredictor(config, model)
    #     predictor.predict()
    # if args.release and args.load_path:
    #     model = ModelRunner(config, is_training=False)
    #     model.evaluate(release=True)
