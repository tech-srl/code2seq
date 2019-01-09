import traceback

from common import Common
from extractor import Extractor
import json
from flask import Flask, request, Response, redirect
from flask.views import MethodView
import logging

SHOW_TOP_CONTEXTS = 5
EXTRACTION_API='https://ff655m4ut8.execute-api.us-east-1.amazonaws.com/production/extractmethods'

def obj_dict(obj):
    return obj.__dict__

cache = {}
logging.basicConfig(level=logging.DEBUG, filename='serverlog.txt', format='%(asctime)s %(message)s')

class HttpRequestRedirector(MethodView):
    def get(self):
        return redirect("http://www.code2seq.org", code=302)

class HttpRequestHandler(MethodView):
    def __init__(self, path_extractor, model):
        self.path_extractor = path_extractor
        self.model = model
        self.logger = logging.getLogger(__name__)
    
    def get_env_var(self, var):
        if var in request.environ:
            return request.environ[var]
        else:
            return var
    
    def post(self):
        response_code = 200
        content_type = 'application/json'
        should_cache = True
        user_ip = str(self.get_env_var('HTTP_X_FORWARDED_FOR'))
        
        request_body = request.data.decode()
        self.logger.info('Request from %s: %s' % (user_ip, request_body))
        if request_body in cache:
            cached_response = cache[request_body]
            self.logger.info('%s Responding from cache, code: %s\n%s' % (user_ip, cached_response.status, cached_response))
            return cached_response
        try:
            results_object = self.predict(request_body)
            response_body = json.dumps(results_object, default=obj_dict)
            self.logger.info('%s responding: %s' % (user_ip, response_body))
        except ValueError as e:
            print(e)
            traceback.print_exc()
            print(request_body)
            response_code = 400
            response_body = str(e)
            self.logger.error('Value error for IP: %s, body: %s' % (user_ip, request_body))
            self.logger.error(str(e))
        except TimeoutError as e:
            print(e)
            traceback.print_exc()
            print(request_body)
            self.logger.error('Timeout error for IP: %s, body: %s' % (user_ip, request_body))
            self.logger.error(str(e))
            response_code = 500
            content_type = 'text/plain'
            response_body = 'Timeout'
            should_cache = False
        except Exception as e:
            print(e)
            traceback.print_exc()
            print(request_body)
            self.logger.error('Exception for IP: %s, body: %s' % (user_ip, request_body))
            self.logger.critical(str(e))
            response_code = 500
            response_body = str(e)
            should_cache = False
        
        response = Response(response_body, status=response_code, content_type=content_type, headers={'Access-Control-Allow-Origin': '*'})
        if should_cache:
            cache[request_body] = response
        return response

    
    def predict(self, code_string):
        paths_input, ast_jsons, pc_info_dict = self.path_extractor.extract_paths(code_string)
        model_results = self.model.predict(paths_input)
        parsed_results = Common.parse_results(model_results, pc_info_dict, topk=SHOW_TOP_CONTEXTS, ast_jsons=ast_jsons)
        if parsed_results is None:
            raise ValueError('Error')
        return parsed_results
        

class Server:
    exit_keywords = ['exit', 'quit', 'q']

    def __init__(self, config, model):
        model.predict([])
        logger = logging.getLogger(__name__)
        logger.info('Loaded model')
        self.model = model
        self.config = config
        self.port = 8080
        self.path_extractor = Extractor(config, EXTRACTION_API, self.config.MAX_PATH_LENGTH, max_path_width=2)
        logger.info('Started Extractor')
        self.app = Flask(__name__)
        self.app.add_url_rule('/predict', view_func=HttpRequestHandler.as_view('predict', 
            path_extractor=self.path_extractor,
            model=self.model))
        logger.info('Starting server')
    
    def read_file(self, input_filename):
        with open(input_filename, 'r') as file:
            return file.readlines()
        
    def offline_serve(self):
        input_filename = 'Input.java'
        print('Serving')
        while True:
            print('Modify the file: "' + input_filename + '" and press any key when ready, or "q" / "exit" to exit')
            user_input = input()
            if user_input.lower() in self.exit_keywords:
                print('Exiting...')
                return
            user_input = ' '.join(self.read_file(input_filename))
            try:
                predict_lines, ast_objects, pc_info_dict = self.path_extractor.extract_paths(user_input)
                #predict_lines, ast_objects, pc_info_dict = self.path_extractor.get_paths_from_file(user_input)
            except ValueError:
                continue
            model_results = self.model.predict(predict_lines)
            
            prediction_results = Common.parse_results(model_results, pc_info_dict, ast_jsons=ast_objects, topk=SHOW_TOP_CONTEXTS)
            for index, method_prediction in prediction_results.items():
                print('Original name:\t' + method_prediction.original_name)
                if self.config.BEAM_WIDTH == 0:
                    print('Predicted:\t%s' % [step.prediction for step in method_prediction.predictions])
                    for timestep, single_timestep_prediction in enumerate(method_prediction.predictions):
                            print('Attention:')
                            print('TIMESTEP: %d\t: %s' % (timestep, single_timestep_prediction.prediction))
                            for attention_obj in single_timestep_prediction.attention_paths:
                                print('%f\tcontext: %s,%s,%s' % (attention_obj['score'], attention_obj['token1']['name'], attention_obj['path'], attention_obj['token2']['name']))
                else:
                    print('Predicted:')
                    for predicted_seq in method_prediction.predictions:
                        print('\t%s' % predicted_seq.prediction)
                
    
    def http_serve(self):
        print('HTTP server starting on port: ' + str(self.port))
        self.app.run(port=self.port, host='0.0.0.0')
    
    def get_app(self):
        return self.app


