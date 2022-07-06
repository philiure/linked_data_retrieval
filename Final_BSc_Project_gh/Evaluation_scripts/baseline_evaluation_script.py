import bz2
import _pickle as pickle
import gc

import nltk
import re
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from math import log10

import sys

import time

import pandas as pd

import signal
from contextlib import contextmanager


@contextmanager
def timeout(time):
    # Register a function to raise a TimeoutError on the signal.
    signal.signal(signal.SIGALRM, raise_timeout)
    # Schedule the signal to be sent after ``time``.
    signal.alarm(time)

    try:
        yield
    except TimeoutError:
        pass
    finally:
        # Unregister the signal so it won't be triggered
        # if the timeout is not reached.
        signal.signal(signal.SIGALRM, signal.SIG_IGN)

def raise_timeout(signum, frame):
    raise TimeoutError


def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '#' * filled_len + ' ' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s %s\r' % (bar, percents, '%', status))
    sys.stdout.flush()

# loading data

progress(0, 2, f'Loading Data')


eval_queries = pickle.load(bz2.BZ2File('eval_queries.pkl.bz2', 'rb'))
dbpedia_dict = pickle.load(bz2.BZ2File('10-2015_DBpedia_12M.pkl.bz2', 'rb'))
inverted_index = pickle.load(bz2.BZ2File('inverted_index_12M.pkl.bz2', 'rb'))
tf_matrix = pickle.load(bz2.BZ2File('term_freq_matrix_12M.pkl.bz2', 'rb'))


index = dbpedia_dict

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
num_documents = float(len(dbpedia_dict))

### Functions

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

def save_file_pklbz2(filename:str, data):
    gc.disable()
    with bz2.BZ2File(f'{filename}.pkl.bz2', 'wb') as bzf:
        pickle.dump(data, bzf, protocol=-1)
    gc.enable()

import re

def clean_subject(subj):
    resource = re.split('#|/',subj)[-1]
    return f'<dbpedia:{resource}>'

## Retrieval Functions

def or_merge(sorted_list1, sorted_list2):  
    merged_list = []  
    list1 = list(sorted_list1)  
    list2 = list(sorted_list2)  
    while (True):  
        if (not list1):  
            merged_list.extend(list2)  
            break  
        if (not list2):  
            merged_list.extend(list1)  
            break  
        if (list1[0] < list2[0]):  
            merged_list.append(list1[0])  
            list1.pop(0)  
        elif (list1[0] > list2[0]):  
            merged_list.append(list2[0])  
            list2.pop(0)  
        else:  
            merged_list.append(list1[0])  
            list1.pop(0)  
            list2.pop(0)  
    return merged_list  

def tf(t,d):
    try:
        x = float(tf_matrix[d][t])
        return x
    except:
        return float(0)

def df(t):
    return float(len(inverted_index[t]))

def idf(t):
    return log10((num_documents + 1)/(df(t) + 1))

def tfidf(t,d):
    return tf(t,d) * idf(t)

def score_ntn_nnn(query_words, doc_id):  
    score = 0  
    for t in query_words:  
        score += tfidf(t,doc_id)
    return score  

def query_ntn_nnn(query_string, n):
    results = []
    query_words = litteral_processing(query_string).split() 
    first_word = query_words[0]  
    remaining_words = query_words[1:]
    or_list = inverted_index[first_word]  
    for t in remaining_words:  
        or_list = or_merge(or_list, inverted_index[t])
    
    for doc_id in sorted(or_list, key=lambda i: score_ntn_nnn(query_words,i), reverse=True)[:n]:  
        s = index[doc_id]['subject']
        s_clean = clean_subject(s)
        results.append(s_clean)
    
    return results



### Running Evaluation

retrieved_results = []
query_times = []
ids = []

# run query test and append to retrieved results
total_num_queries = len(eval_queries)

progress(0, total_num_queries, f'')

for idx, row in eval_queries.iterrows():
    query_num = idx
    query = row['query']
    relevant_entities = row['relevant_entities']
    very_relevant_entities = row['very_relevant_entities']
    len_relevant = len(relevant_entities)
    len_very_relevant = len(very_relevant_entities)
    n = len_relevant+len_very_relevant


    
    progress(query_num, total_num_queries, f'Query No. {query_num}/{total_num_queries}')

    time_start = time.time()

    with timeout(200):      
        # retrieved results
        retrieved = query_ntn_nnn(query, n)
        time_done = time.time() - time_start
        retrieved_results.append(retrieved)
        query_times.append(time_done)
        ids.append(query_num)

progress(total_num_queries, total_num_queries, f'Complete')

print(f'\n\n\nSAVING...')
#### Saving Evaluation

dic = {'id':ids, 'retrieved':retrieved_results, 'query_time':query_times}

baseline_results = pd.DataFrame(dic)

save_file_pklbz2('baseline_results_script', baseline_results)

failed = [id for id in range(total_num_queries) if id not in ids]
print(f'\n\n\n\nFinished!\nFailed query IDs:\n {failed}')