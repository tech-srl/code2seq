import json
import re
import subprocess
import sys


class Common:
    internal_delimiter = '|'
    SOS = '<S>'
    EOS = '</S>'
    PAD = '<PAD>'
    UNK = '<UNK>'

    @staticmethod
    def normalize_word(word):
        stripped = re.sub(r'[^a-zA-Z]', '', word)
        if len(stripped) == 0:
            return word.lower()
        else:
            return stripped.lower()

    @staticmethod
    def process_test_input(test_file):
        programs = []
        with open(test_file, 'r') as file:
            line_number = 0
            for line in file:
                line_number += 1
                line = line.rstrip('\n')
                current_program, id_to_var = {}, {}
                try:
                    single_program_object = json.loads(line)
                except ValueError:
                    print >> sys.stderr, 'Bad JSON: ' + str(line)
                    continue
                assign = single_program_object['assign']
                query = single_program_object['query']
                for var_object in assign:
                    name = ''
                    infer_type = ''
                    if Common.INF in var_object:
                        name, infer_type = var_object[Common.INF], Common.INF
                    elif Common.GIV in var_object:
                        name, infer_type = var_object[Common.GIV], Common.GIV
                    id = var_object['v']
                    id_to_var[id] = (name, infer_type)
                for feature in query:
                    if not ('a' in feature and 'b' in feature and 'f2' in feature):
                        continue
                    try:
                        name, type1 = id_to_var[feature['a']]
                        name2, type2 = id_to_var[feature['b']]
                    except KeyError:
                        print('Key error in line: ' + str(line_number))
                        print(line)
                        sys.exit(0)
                    if feature['a'] == feature['b']:
                        name2 = 'self'
                    path = feature['f2']
                    context = str(path) + ',' + name2
                    if (name, type1) in current_program:
                        current_program[(name, type1)].append(context)
                    else:
                        current_program[(name, type1)] = [context]
                programs.append(current_program)
        return programs

    @staticmethod
    def load_histogram(path, max_size=None):
        histogram = {}
        with open(path, 'r') as file:
            for line in file.readlines():
                parts = line.split(' ')
                if not len(parts) == 2:
                    continue
                histogram[parts[0]] = int(parts[1])
        sorted_histogram = [(k, histogram[k]) for k in sorted(histogram, key=histogram.get, reverse=True)]
        return dict(sorted_histogram[:max_size])

    @staticmethod
    def load_vocab_from_dict(word_to_count, add_values=[], max_size=None):
        word_to_index, index_to_word = {}, {}
        current_index = 0
        for value in add_values:
            word_to_index[value] = current_index
            index_to_word[current_index] = value
            current_index += 1
        sorted_counts = [(k, word_to_count[k]) for k in sorted(word_to_count, key=word_to_count.get, reverse=True)]
        limited_sorted = dict(sorted_counts[:max_size])
        for word, count in limited_sorted.items():
            word_to_index[word] = current_index
            index_to_word[current_index] = word
            current_index += 1
        return word_to_index, index_to_word, current_index

    @staticmethod
    def calculate_max_contexts(file):
        contexts_per_word = Common.process_test_input(file)
        return max(
            [max(l, default=0) for l in [[len(contexts) for contexts in prog.values()] for prog in contexts_per_word]],
            default=0)

    @staticmethod
    def binary_to_string(binary_string):
        return binary_string.decode("utf-8")

    @staticmethod
    def binary_to_string_list(binary_string_list):
        return [Common.binary_to_string(w) for w in binary_string_list]

    @staticmethod
    def binary_to_string_matrix(binary_string_matrix):
        return [Common.binary_to_string_list(l) for l in binary_string_matrix]

    @staticmethod
    def binary_to_string_3d(binary_string_tensor):
        return [Common.binary_to_string_matrix(l) for l in binary_string_tensor]

    @staticmethod
    def load_file_lines(path):
        with open(path, 'r') as f:
            return f.read().splitlines()

    @staticmethod
    def load_token_to_subtoken(path):
        print('Loading token-to-subtoken mapping from: ' + path, file=sys.stderr)
        mapping = {}
        with open(path, 'r') as file:
            for line in file:
                try:
                    json_object = json.loads(line)
                except ValueError:
                    print >> sys.stderr, 'Bad JSON: ' + str(line)
                    continue
                token = json_object['token']
                subtokens = json_object['subtokens']
                mapping[token] = subtokens
        print('Finished loading token-to-subtoken mapping, found: ' + str(len(mapping)) + ' tokens', file=sys.stderr)
        return mapping

    @staticmethod
    def split_to_batches(data_lines, batch_size):
        return [data_lines[x:x + batch_size] for x in range(0, len(data_lines), batch_size)]

    @staticmethod
    def legal_method_names_checker(name):
        return not name in [Common.UNK, Common.PAD,
                            Common.EOS]  # and re.match('^_*[a-zA-Z0-9]+$', name.replace(common.internalDelimiter, ''))

    @staticmethod
    def filter_impossible_names(top_words):
        result = list(filter(Common.legal_method_names_checker, top_words))
        return result

    @staticmethod
    def unique(sequence):
        unique = []
        [unique.append(item) for item in sequence if item not in unique]
        return unique

    @staticmethod
    def parse_results(result, pc_info_dict, topk=5):
        prediction_results = {}
        results_counter = 0
        for single_method in result:
            original_name, top_suggestions, top_scores, attention_per_context = list(single_method)
            current_method_prediction_results = PredictionResults(original_name)
            if attention_per_context is not None:
                word_attention_pairs = [(word, attention) for word, attention in
                                        zip(top_suggestions[0], attention_per_context) if
                                        Common.legal_method_names_checker(word)]
                for predicted_word, attention_timestep in word_attention_pairs:
                    current_timestep_paths = []
                    for context, attention in [(key, attention_timestep[key]) for key in
                                               sorted(attention_timestep, key=attention_timestep.get, reverse=True)][
                                              :topk]:
                        if context in pc_info_dict:
                            pc_info = pc_info_dict[context]
                            current_timestep_paths.append((attention.item(), pc_info))

                    current_method_prediction_results.append_prediction(predicted_word, current_timestep_paths)
            else:
                for predicted_seq in top_suggestions:
                    filtered_seq = [word for word in predicted_seq if Common.legal_method_names_checker(word)]
                    current_method_prediction_results.append_prediction(filtered_seq, None)

            prediction_results[results_counter] = current_method_prediction_results
            results_counter += 1
        return prediction_results

    @staticmethod
    def compute_bleu(ref_file_name, predicted_file_name):
        with open(predicted_file_name) as predicted_file:
            pipe = subprocess.Popen(["perl", "scripts/multi-bleu.perl", ref_file_name], stdin=predicted_file,
                                    stdout=sys.stdout, stderr=sys.stderr)


class PredictionResults:
    def __init__(self, original_name):
        self.original_name = original_name
        self.predictions = list()

    def append_prediction(self, name, current_timestep_paths):
        self.predictions.append(SingleTimeStepPrediction(name, current_timestep_paths))

    def get_token_representation(self, name, node_id, token_occurr):
        return {'name': name, 'node_id': node_id}

    def append_attention_paths_for_timestep(self, current_timestep_paths):
        current_timestep_results = []

        self.attention_paths.append(current_timestep_results)


class SingleTimeStepPrediction:
    def __init__(self, prediction, attention_paths):
        self.prediction = prediction
        if attention_paths is not None:
            paths_with_scores = []
            for attention_score, pc_info in attention_paths:
                path_context_dict = {'score': attention_score,
                                     'path': pc_info.longPath,
                                     'token1': self.create_token_dict(pc_info.word1, pc_info.word1NodeId),
                                     'token2': self.create_token_dict(pc_info.word2, pc_info.word2NodeId)}
                paths_with_scores.append(path_context_dict)
            self.attention_paths = paths_with_scores

    def create_token_dict(self, name, node_id):
        return {'name': name, 'node_id': node_id}
