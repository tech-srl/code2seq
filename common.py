import re
import json
import sys

import subprocess


class Config:
    @staticmethod
    def get_default_config(args):
        config = Config()
        config.NUM_EPOCHS = 5000
        config.SAVE_EVERY_EPOCHS = 1
        config.BATCH_SIZE = 512
        config.TEST_BATCH_SIZE = 256
        config.PREFETCH_NUM_BATCHES = 10
        config.NUM_BATCHING_THREADS = 7
        config.BATCH_QUEUE_SIZE = 100000
        config.TRAIN_PATH = args.data_path
        config.TEST_PATH = args.test_path
        config.DATA_NUM_CONTEXTS = 1000
        config.MAX_CONTEXTS = 200
        config.WORDS_MIN_COUNT = 150
        config.TARGET_WORDS_MIN_COUNT = 10
        config.EMBEDDINGS_SIZE = 128
        config.RNN_SIZE = 128 * 2
        config.DECODER_SIZE = 320
        config.WORDS_HISTOGRAM_PATH = args.words_histogram
        config.TARGET_WORDS_HISTOGRAM_PATH = args.target_words_histogram
        config.NODES_HISTOGRAM_PATH = args.nodes_histogram
        config.SAVE_PATH = args.save_path
        config.SAVE_W2V = args.save_w2v
        config.LOAD_PATH = args.load_path
        config.TRACE = args.trace
        config.MAX_PATH_LENGTH = 8 + 1
        config.MAX_NAME_PARTS = 5
        config.MAX_TARGET_PARTS = 5
        config.EMBEDDINGS_DROPOUT_KEEP_PROB = 0.75
        config.RNN_DROPOUT_KEEP_PROB = 0.5
        config.BIRNN = True
        config.RANDOM_CONTEXTS = True
        config.BEAM_WIDTH = 0
        config.EXTRACTION_API_CHECK = False
        return config

    def __init__(self):
        self.NUM_EPOCHS = 0
        self.SAVE_EVERY_EPOCHS = 0
        self.BATCH_SIZE = 0
        self.TEST_BATCH_SIZE = 0
        self.PREFETCH_NUM_BATCHES = 0
        self.NUM_BATCHING_THREADS = 0
        self.BATCH_QUEUE_SIZE = 0
        self.TRAIN_PATH = ''
        self.TEST_PATH = ''
        self.DATA_NUM_CONTEXTS = 0
        self.MAX_CONTEXTS = 0
        self.WORDS_MIN_COUNT = 0
        self.TARGET_WORDS_MIN_COUNT = 0
        #self.PATHS_MIN_COUNT = 0
        self.EMBEDDINGS_SIZE = 0
        self.RNN_SIZE = 0
        self.DECODER_SIZE = 0 
        self.NUM_EXAMPLES = 0
        self.WORDS_HISTOGRAM_PATH = ''
        self.TARGET_WORDS_HISTOGRAM_PATH = ''
        self.NODES_HISTOGRAM_PATH = ''
        self.SAVE_PATH = ''
        self.SAVE_W2V = ''
        self.LOAD_PATH = ''
        self.TRACE = ''
        self.MAX_PATH_LENGTH = 0
        self.MAX_NAME_PARTS = 0
        self.MAX_TARGET_PARTS = 0
        self.EMBEDDINGS_DROPOUT_KEEP_PROB = 0
        self.RNN_DROPOUT_KEEP_PROB = 0
        self.BIRNN = False
        self.RANDOM_CONTEXTS = True
        self.BEAM_WIDTH = 1
        self.EXTRACTION_API_CHECK = False

class common:
    GIV = 'giv'
    INF = 'inf'
    noSuchWord = "NoSuchWord"
    blank_target_padding = 'BLANK'
    internal_delimiter = '|'
    PRED_START = '<S>'
    PRED_END = '</S>'

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
                    if common.INF in var_object:
                        name, infer_type = var_object[common.INF], common.INF
                    elif common.GIV in var_object:
                        name, infer_type = var_object[common.GIV], common.GIV
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
    def load_vocab_from_histogram(path, min_count=0, start_from=0, add_values=[]):
        with open(path, 'r') as file:
            word_to_index = {}
            index_to_word = {}
            next_index = start_from
            for value in add_values:
                word_to_index[value] = next_index
                index_to_word[next_index] = value
                next_index += 1
            for line in file:
                line_values = line.rstrip().split(' ')
                if len(line_values) != 2:
                    continue
                word = line_values[0]
                count = int(line_values[1])
                if count < min_count:
                    continue
                if word in word_to_index:
                    continue
                word_to_index[word] = next_index
                index_to_word[next_index] = word
                next_index += 1
        return word_to_index, index_to_word, next_index - start_from

    @staticmethod
    def load_json(json_file):
        data = []
        with open(json_file, 'r') as file:
            for line in file:
                current_program = common.process_single_json_line(line)
                if current_program is None:
                    continue
                for element, scope in current_program.items():
                    data.append((element, scope))
        return data

    @staticmethod
    def load_json_streaming(json_file):
        with open(json_file, 'r') as file:
            for line in file:
                current_program = common.process_single_json_line(line)
                if current_program is None:
                    continue
                for element, scope in current_program.items():
                    yield (element, scope)

    @staticmethod
    def process_single_json_line(line):
        line = line.rstrip('\n')
        id_to_var, current_program = {}, {}
        try:
            single_program_object = json.loads(line)
        except ValueError:
            print('Bad JSON: ' + str(line), file=sys.stderr)
            return None
        assign = single_program_object['assign']
        query = single_program_object['query']
        for var_object in assign:
            name1 = ''
            infer_type = ''
            if common.INF in var_object:
                name1, infer_type = var_object[common.INF], common.INF
            elif common.GIV in var_object:
                name1, infer_type = var_object[common.GIV], common.GIV
            id = var_object['v']
            id_to_var[id] = (name1, infer_type)
        for feature in query:
            if not ('a' in feature and 'b' in feature and 'f2' in feature):
                continue
            try:
                name1, type1 = id_to_var[feature['a']]
                name2, type2 = id_to_var[feature['b']]
            except KeyError:
                print('Key error')
                print(line)
                sys.exit(0)
            if feature['a'] == feature['b']:
                name2 = 'self'
            path = feature['f2']
            context = str(path) + ',' + name2
            if (name1, type1) in current_program:
                current_program[(name1, type1)].append(context)
            else:
                current_program[(name1, type1)] = [context]
        return current_program

    @staticmethod
    def save_word2vec_file(file, vocab_size, dimension, index_to_word, vectors):
        file.write('%d %d\n' % (vocab_size, dimension))
        for i in range(1,vocab_size+1):
            if i in index_to_word:
                file.write(index_to_word[i] + ' ')
                file.write(' '.join(map(str, vectors[i])) + '\n')

    @staticmethod
    def calculate_max_contexts(file):
        contexts_per_word = common.process_test_input(file)
        return max(
            [max(l, default=0) for l in [[len(contexts) for contexts in prog.values()] for prog in contexts_per_word]],
            default=0)

    @staticmethod
    def binary_to_string(binary_string):
        return binary_string.decode("utf-8")

    @staticmethod
    def binary_to_string_list(binary_string_list):
        return [common.binary_to_string(w) for w in binary_string_list]

    @staticmethod
    def binary_to_string_matrix(binary_string_matrix):
        return [common.binary_to_string_list(l) for l in binary_string_matrix]
    
    @staticmethod
    def binary_to_string_3d(binary_string_tensor):
        return [common.binary_to_string_matrix(l) for l in binary_string_tensor]

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
        # This allows legal method names such as: "_4" (it's legal and common)
        return not name in [common.noSuchWord, common.blank_target_padding, common.PRED_END] # and re.match('^_*[a-zA-Z0-9]+$', name.replace(common.internalDelimiter, ''))
        #return name != common.noSuchWord and re.match('^[a-zA-Z]+$', name)

    @staticmethod
    def filter_impossible_names(top_words):
        result = list(filter(common.legal_method_names_checker, top_words))
        return result
    
    @staticmethod
    def unique(sequence):
        unique = []
        [unique.append(item) for item in sequence if item not in unique]
        return unique
    
    @staticmethod
    def parse_results(result, pc_info_dict, ast_jsons, topk=5):
        prediction_results = {}
        results_counter = 0
        for single_method, ast in zip(result, ast_jsons):
            original_name, top_suggestions, top_scores, attention_per_context = list(single_method)
            #original_name, top_suggestions = list(single_method)
            current_method_prediction_results = PredictionResults(original_name, ast)
            if attention_per_context is not None:
                word_attention_pairs = [(word, attention) for word, attention in zip(top_suggestions, attention_per_context) if common.legal_method_names_checker(word)]
                if len(word_attention_pairs) == 0:
                    current_method_prediction_results.append_prediction('unknown', [])
                for predicted_word, attention_timestep in word_attention_pairs:
                    current_timestep_paths = []
                    for context, attention in [(key, attention_timestep[key]) for key in sorted(attention_timestep, key=attention_timestep.get, reverse=True)][:topk]:
                        if context in pc_info_dict:
                            pc_info = pc_info_dict[context]
                            current_timestep_paths.append((attention.item(), pc_info))
                    
                    current_method_prediction_results.append_prediction(predicted_word, current_timestep_paths)
            else:
                 for predicted_seq in top_suggestions:
                     filtered_seq = [word for word in predicted_seq if common.legal_method_names_checker(word)]
                     current_method_prediction_results.append_prediction(filtered_seq, None)                   

            prediction_results[results_counter] = current_method_prediction_results
            results_counter += 1
        return prediction_results
    
    @staticmethod
    def compute_bleu(ref_file_name, predicted_file_name):
        with open(predicted_file_name) as predicted_file:
            pipe = subprocess.Popen(["perl", "scripts/multi-bleu.perl", ref_file_name], stdin=predicted_file, stdout=sys.stdout, stderr=sys.stderr)
            
        
class PredictionResults:    
    def __init__(self, original_name, ast):
        self.original_name = original_name
        self.predictions = list()
        self.AST = ast
    
    def append_prediction(self, name, current_timestep_paths):
        self.predictions.append(SingleTimeStepPrediction(name, current_timestep_paths))
        
    def get_token_representation(self, name, node_id, token_occurr):
        return {'name': name, 'node_id': node_id} #, 'occurr': token_occurr}
    
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
    
