#! /usr/bin/env python3
import nltk
import re
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import sys
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

from collections import defaultdict, Counter
import bz2
import _pickle as pickle

import gc

def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '#' * filled_len + ' ' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s %s\r' % (bar, percents, '%', status))
    sys.stdout.flush()

def save_file_pklbz2(filename:str, data):
    gc.disable()
    with bz2.BZ2File(f'{filename}.pkl.bz2', 'wb') as bzf:
        pickle.dump(data, bzf, protocol=-1)
    gc.enable()

def litteral_processing(litteral):
  #removes special chars
  litteral = re.sub('[^A-Za-z0-9]+', ' ', litteral)
  #removes non ASCII
  litteral = re.sub('[^\x00-\x7F]+', ' ', litteral)
  #lowering litteral
  litteral = litteral.lower()
  #removes numbers
  litteral = re.sub('\d+', ' ', litteral)
  #removes stopwords
  litteral = ' '.join([word for word in litteral.split() if word not in stop_words])
  #lemmaziter
  litteral = ' '.join([lemmatizer.lemmatize(word) for word in litteral.split()])
  #removes double spaces
  litteral = re.sub(' +', ' ', litteral)

  return litteral


# loading dictionairy of virtual docs

lib = pickle.load(bz2.BZ2File('library_vdocs_12M.pkl.bz2', 'rb'))
size_lib = len(lib)
tf_matrix = defaultdict(Counter)
inverted_index = defaultdict(list)

for id in lib.keys():
  progress(id, size_lib)
  tokens = litteral_processing(lib[id]).split()
  tf_matrix[id] = dict(Counter(tokens))

  term_set = set(tokens)
  for term in term_set:
    inverted_index[term].append(id)

print('\n\nSaving...')
save_file_pklbz2('term_freq_matrix_12M', tf_matrix)
save_file_pklbz2('inverted_index_12M', inverted_index)
print('Complete! Can exit the script.')