class PathContextInformation:
    def __init__(self, context_json_object):
        #self.word1, self.path, self.word2 = context_string.split(',')
        self.word1 = context_json_object['name1']
        self.longPath = context_json_object['path']
        self.shortPath = context_json_object['shortPath']
        self.word2 = context_json_object['name2']
        self.word1NodeId = context_json_object['name1NodeId']
        self.word2NodeId = context_json_object['name2NodeId']
        self.word1TokenNum = context_json_object['name1TokenNum']
        self.word2TokenNum = context_json_object['name2TokenNum']
        
    def __str__(self):
        return '%s,%s,%s' % (self.word1, self.shortPath, self.word2)