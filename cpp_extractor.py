import os

from cppminer.cpp_parser import AstParser
from cppminer.cpp_parser.sample import make_str_key
from common import PathContextInformation
import tempfile


class CppExtractor:
    def __init__(self, config, ):
        self.config = config
        self.parser = AstParser(max_contexts_num=self.config.MAX_CONTEXTS,
                                max_path_len=self.config.MAX_PATH_LENGTH,
                                max_subtokens_num=self.config.MAX_NAME_PARTS,
                                max_ast_depth=100,
                                out_path=None)

    def extract_paths(self, code_string):
        tmp = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.cc')
        try:
            tmp.write(code_string)
            tmp.close()

            self.parser.parse(compiler_args=[], file_path=tmp.name)
            pc_info_dict = {}
            result = []
            for sample in self.parser.samples:
                for context in sample.contexts:
                    info_context = {'name1': make_str_key(context.start_token),
                                    'name2': make_str_key(context.end_token),
                                    'path': make_str_key(context.path.tokens),
                                    'shortPath': make_str_key(context.path.tokens)}
                    pc_info = PathContextInformation(info_context)
                    pc_info_dict[(pc_info.token1, pc_info.shortPath, pc_info.token2)] = pc_info
                result_line = str(sample)
                space_padding = ' ' * (self.config.DATA_NUM_CONTEXTS - len(sample.contexts))
                result_line += space_padding
                result.append(result_line)
        finally:
            os.unlink(tmp.name)

        return result, pc_info_dict
