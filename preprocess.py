import pickle
from argparse import ArgumentParser

import numpy as np

import common

'''
This script preprocesses the data from MethodPaths. It truncates methods with too many contexts,
and pads methods with less paths with spaces.
'''


def save_dictionaries(dataset_name, subtoken_to_count, node_to_count, target_to_count, max_contexts, num_examples):
    save_dict_file_path = '{}.dict.c2s'.format(dataset_name)
    with open(save_dict_file_path, 'wb') as file:
        pickle.dump(subtoken_to_count, file)
        pickle.dump(node_to_count, file)
        pickle.dump(target_to_count, file)
        pickle.dump(max_contexts, file)
        pickle.dump(num_examples, file)
        print('Dictionaries saved to: {}'.format(save_dict_file_path))


def process_file(file_path, data_file_role, dataset_name, max_contexts, max_data_contexts):
    sum_total = 0
    sum_sampled = 0
    total = 0
    max_unfiltered = 0
    max_contexts_to_sample = max_data_contexts if data_file_role == 'train' else max_contexts
    output_path = '{}.{}.c2s'.format(dataset_name, data_file_role)
    with open(output_path, 'w') as outfile:
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.rstrip('\n').split(' ')
                target_name = parts[0]
                contexts = parts[1:]

                if len(contexts) > max_unfiltered:
                    max_unfiltered = len(contexts)

                sum_total += len(contexts)
                if len(contexts) > max_contexts_to_sample:
                    contexts = np.random.choice(contexts, max_contexts_to_sample, replace=False)

                sum_sampled += len(contexts)

                csv_padding = " " * (max_data_contexts - len(contexts))
                total += 1
                outfile.write(target_name + ' ' + " ".join(contexts) + csv_padding + '\n')

    print('File: ' + data_file_path)
    print('Average total contexts: ' + str(float(sum_total) / total))
    print('Average final (after sampling) contexts: ' + str(float(sum_sampled) / total))
    print('Total examples: ' + str(total))
    print('Max number of contexts per word: ' + str(max_unfiltered))
    return total


def context_full_found(context_parts, word_to_count, path_to_count):
    return context_parts[0] in word_to_count \
           and context_parts[1] in path_to_count and context_parts[2] in word_to_count


def context_partial_found(context_parts, word_to_count, path_to_count):
    return context_parts[0] in word_to_count \
           or context_parts[1] in path_to_count or context_parts[2] in word_to_count


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-trd", "--train_data", dest="train_data_path",
                        help="path to training data file", required=True)
    parser.add_argument("-ted", "--test_data", dest="test_data_path",
                        help="path to test data file", required=True)
    parser.add_argument("-vd", "--val_data", dest="val_data_path",
                        help="path to validation data file", required=True)
    parser.add_argument("-mc", "--max_contexts", dest="max_contexts", default=200,
                        help="number of max contexts to keep in test+validation", required=False)
    parser.add_argument("-mdc", "--max_data_contexts", dest="max_data_contexts", default=1000,
                        help="number of max contexts to keep in the dataset", required=False)
    parser.add_argument("-svs", "--subtoken_vocab_size", dest="subtoken_vocab_size", default=186277,
                        help="Max number of source subtokens to keep in the vocabulary", required=False)
    parser.add_argument("-tvs", "--target_vocab_size", dest="target_vocab_size", default=26347,
                        help="Max number of target words to keep in the vocabulary", required=False)
    parser.add_argument("-sh", "--subtoken_histogram", dest="subtoken_histogram",
                        help="subtoken histogram file", metavar="FILE", required=True)
    parser.add_argument("-nh", "--node_histogram", dest="node_histogram",
                        help="node_histogram file", metavar="FILE", required=True)
    parser.add_argument("-th", "--target_histogram", dest="target_histogram",
                        help="target histogram file", metavar="FILE", required=True)
    parser.add_argument("-o", "--output_name", dest="output_name",
                        help="output name - the base name for the created dataset", required=True, default='data')
    args = parser.parse_args()

    train_data_path = args.train_data_path
    test_data_path = args.test_data_path
    val_data_path = args.val_data_path
    subtoken_histogram_path = args.subtoken_histogram
    node_histogram_path = args.node_histogram

    subtoken_to_count = common.Common.load_histogram(subtoken_histogram_path,
                                                     max_size=int(args.subtoken_vocab_size))
    node_to_count = common.Common.load_histogram(node_histogram_path,
                                                 max_size=None)
    target_to_count = common.Common.load_histogram(args.target_histogram,
                                                   max_size=int(args.target_vocab_size))
    print('subtoken vocab size: ', len(subtoken_to_count))
    print('node vocab size: ', len(node_to_count))
    print('target vocab size: ', len(target_to_count))

    num_training_examples = 0
    for data_file_path, data_role in zip([test_data_path, val_data_path, train_data_path], ['test', 'val', 'train']):
        num_examples = process_file(file_path=data_file_path, data_file_role=data_role, dataset_name=args.output_name,
                                    max_contexts=int(args.max_contexts), max_data_contexts=int(args.max_data_contexts))
        if data_role == 'train':
            num_training_examples = num_examples

    save_dictionaries(dataset_name=args.output_name, subtoken_to_count=subtoken_to_count,
                      node_to_count=node_to_count, target_to_count=target_to_count,
                      max_contexts=int(args.max_data_contexts), num_examples=num_training_examples)
