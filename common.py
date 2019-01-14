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
    def legal_method_names_checker(name):
        return not name in [Common.UNK, Common.PAD, Common.EOS]

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
                                        zip(top_suggestions, attention_per_context) if
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

class SingleTimeStepPrediction:
    def __init__(self, prediction, attention_paths):
        self.prediction = prediction
        if attention_paths is not None:
            paths_with_scores = []
            for attention_score, pc_info in attention_paths:
                path_context_dict = {'score': attention_score,
                                     'path': pc_info.longPath,
                                     'token1': pc_info.token1,
                                     'token2': pc_info.token2}
                paths_with_scores.append(path_context_dict)
            self.attention_paths = paths_with_scores


class PathContextInformation:
    def __init__(self, context):
        self.token1 = context['name1']
        self.longPath = context['path']
        self.shortPath = context['shortPath']
        self.token2 = context['name2']

    def __str__(self):
        return '%s,%s,%s' % (self.token1, self.shortPath, self.token2)
