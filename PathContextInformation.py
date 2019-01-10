class PathContextInformation:
    def __init__(self, context):
        self.token1 = context['name1']
        self.longPath = context['path']
        self.shortPath = context['shortPath']
        self.token2 = context['name2']

    def __str__(self):
        return '%s,%s,%s' % (self.token1, self.shortPath, self.token2)