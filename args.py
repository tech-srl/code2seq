from argparse import ArgumentParser


def read_args():
    parser = ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-d", "--data", dest="data_path",
                       help="path to preprocessed dataset")
    group.add_argument("-l", "--load", dest="load_path",
                       help="path to saved file", metavar="FILE")

    parser.add_argument("-m", "--model_path", dest="model_path",
                        help="path to save and load checkpoint", metavar="FILE", required=False)
    parser.add_argument("-s", "--save_path", dest="save_path",
                        help="path to save file", metavar="FILE", required=False)

    parser.add_argument("-t", "--test", dest="test_path",
                        help="path to test file", metavar="FILE", required=False)

    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', type=int, default=239)
    return parser.parse_args()