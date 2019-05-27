#!/usr/bin/python

import javalang
import sys
import re


modifiers = ['public', 'private', 'protected', 'static']

RE_WORDS = re.compile(r'''
    # Find words in a string. Order matters!
    [A-Z]+(?=[A-Z][a-z]) |  # All upper case before a capitalized word
    [A-Z]?[a-z]+ |  # Capitalized words / all lower case
    [A-Z]+ |  # All upper case
    \d+ | # Numbers
    .+
''', re.VERBOSE)

def split_subtokens(str):
    return [subtok for subtok in RE_WORDS.findall(str) if not subtok == '_']

def tokenizeFile(file_path):
  lines = 0
  with open(file_path, 'r', encoding="utf-8") as file:
    with open(file_path + 'method_names.txt', 'w') as method_names_file:
      with open(file_path + 'method_subtokens_content.txt', 'w') as method_contents_file:
        for line in file:
          lines += 1
          line = line.rstrip()
          parts = line.split('|', 1)
          method_name = parts[0]
          method_content = parts[1]
          try:
            tokens = list(javalang.tokenizer.tokenize(method_content))
          except:
            print('ERROR in tokenizing: ' + method_content)
            #tokens = method_content.split(' ')
          if len(method_name) > 0 and len(tokens) > 0:
            method_names_file.write(method_name + '\n')
            method_contents_file.write(' '.join([' '.join(split_subtokens(i.value)) for i in tokens if not i.value in modifiers]) + '\n')
          else:
            print('ERROR in len of: ' + method_name + ', tokens: ' + str(tokens))
  print(str(lines))


if __name__ == '__main__':
  file = sys.argv[1]
  tokenizeFile(file)


