#! /usr/bin/env python3

###########################  PUSH  #################################
# WARNING Docker server has slow performance RUST server in --release mode is better
# START DOCKER SERVER 
# docker pull valeriansaliou/sonic:v1.3.2
# docker run -p 1491:1491 -v /users/path/to/sonic/config.cfg:/etc/sonic.cfg -v /users/path/to/sonic/store/:/var/lib/sonic/store/ valeriansaliou/sonic:v1.3.2

# platform errors? use:  --platform NAME_PLATFORM (e.g. linux/amd64/v8)
# docker run --platform linux/amd64/v8 -p 1491:1491 -v /users/philippe/desktop/bsc\ project/git/evaluation/sonic/config.cfg:/etc/sonic.cfg -v /users/philippe/desktop/bsc\ project/git/evaluation/sonic/store:/var/lib/sonic/store/ valeriansaliou/sonic:v1.3.2

# Once Rust installed (see Sonic GitHub)
# Start server using: sonic

from sonic import IngestClient
import _pickle as pickle
import bz2
import time
import sys
import numpy as np
from timeit import default_timer as timer
from sonic import SearchClient
import gc
import re
import pandas as pd
import sys
import math

########### Running Evaluation ###########

 # Push functions

def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '#' * filled_len + ' ' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s %s\r' % (bar, percents, '%', status))
    sys.stdout.flush()

# Search functions
def save_file_pklbz2(filename:str, data):
    gc.disable()
    with bz2.BZ2File(f'{filename}.pkl.bz2', 'wb') as bzf:
        pickle.dump(data, bzf, protocol=-1)
    gc.enable()


def clean_subject(subj):
    resource = re.split('#|/',subj)[-1]
    return f'<dbpedia:{resource}>'


start_global_push = timer()
print('Loading Documents...')
data = pickle.load(bz2.BZ2File('library_vdocs_12M.pkl.bz2', 'rb'))
print(f'Loading Complete - Size Documents: {sys.getsizeof(data)*(10**-9)} GB\n')


# ###########################  Push  #################################
collection = 'EVAL'
bucket = 'Default'
start = timer()
times = []
failed_ids = []
num_docs = len(data)

with IngestClient("::1", 1491, "SecretPassword") as ingestcl:
    print(f'Ping: {ingestcl.ping()} Protocol: ingest {ingestcl.protocol} Buffer {ingestcl.bufsize}')
    for id in range(len(data))[1889881:]:
        try:
            start_push = timer()
            ingestcl.push(collection, bucket, str(id), data[id], lang='Eng')
            end = timer()
            time_difference = end-start_push
            times.append(time_difference)

        except:
            failed_ids.append(id)

        progress(id+1, num_docs, f'[{id+1}/{num_docs}] [{round(time_difference*1000,4)} ms/doc]\n Num Failed [{len(failed_ids)}]')

    print(f'\n\nFinished pushing {num_docs-len(failed_ids)} documents in {timer()-start} seconds.\nFailed pushes:\n{failed_ids}')

    ingestcl.quit()

print('Push Finished')


######### Search #########

print('\nLoading...\n')
index = pickle.load(bz2.BZ2File('10-2015_DBpedia_12M.pkl.bz2', 'rb'))
print('Index dataset loaded')
eval_queries = pickle.load(bz2.BZ2File('eval_queries.pkl.bz2', 'rb'))
print('Query dataset loaded')
tf_matrix = pickle.load(bz2.BZ2File('term_freq_matrix_12M.pkl.bz2', 'rb'))
print('TF Matrix loaded')
inverted_index = pickle.load(bz2.BZ2File('inverted_index_12M.pkl.bz2', 'rb'))
print('Inverted Index loaded')
print('\nLoading Complete\n')

num_documents = float(len(index))
# Ranking Functions
def tf(t,d):
    try:
        x = float(tf_matrix[d][t])
        return x
    except:
        return float(0)

def df(t):
    return float(len(inverted_index[t]))

def idf(t):
    return math.log10((num_documents + 1)/(df(t) + 1))

def tfidf(t,d):
    return tf(t,d) * idf(t)

def scoreTFIDF(query_tokens:list, doc_id):
    score = 0  
    for t in query_words:  
        score += tfidf(t,doc_id)
    return score  


###########################

query_ids = []
failed = []
retrieved_results = []
query_times = []
complete_query_times = []

total_num_queries = len(eval_queries)
collection = 'EVAL'
bucket = 'Default'

print('Start Search\n')

with SearchClient("::1", 1491, "SecretPassword") as querycl:

    # insert for loop over queries
    for idx, row in eval_queries.iterrows():
        query_num = idx
        query = row['query']
        relevant_entities = row['relevant_entities']
        very_relevant_entities = row['very_relevant_entities']
        len_relevant = len(relevant_entities)
        len_very_relevant = len(very_relevant_entities)
        n = len_relevant+len_very_relevant
        
        retrieved_ids_query = []
        query_time = 0

        #Retrieve all documents per term, append to list
        query_words = query.split()

        start_global_time = timer()

        for term in query_words:    
            # retrieved results
            time_start = timer()
            retrieved_ids_token = querycl.query(collection, bucket, term)
            time_token = timer() - time_start
            query_time += time_token
            time.sleep(0.01)

            for i in retrieved_ids_token:
                retrieved_ids_query.append(i)

        ranked_retrieved = []
        
        # TF IDF rank with the found results
        for doc_id in sorted(set(retrieved_ids_query), key=lambda i: scoreTFIDF(query_words,i), reverse=True)[:n]:  
            # INSERT MULTIPLE TRIES ATTEMPT  
            try:          
                s = index[int(doc_id)]['subject']
                s_clean = clean_subject(s)
                ranked_retrieved.append(s_clean)
                continue
            except Exception as e:
                print(e)
        with_rank_query_time = timer() - start_global_time

        complete_query_times.append(with_rank_query_time)

        query_times.append(query_time)

        # return n results if possible
        if len(ranked_retrieved) >= n: retrieved_results.append(ranked_retrieved[:n])
        else: 
            print(f'Results for query "{query}" is smaller than {n}, len(results):{len(ranked_retrieved)}')
            retrieved_results.append(ranked_retrieved)
        
        query_ids.append(query_num)

        
    #### Saving Evaluation
    print(f'\n\n\nSAVING...')

    dic = {'id': query_ids, 'retrieved':retrieved_results, 'query_time':query_times, 'query_time_with_rank_time': complete_query_times}

    sonic_results = pd.DataFrame(dic)

    save_file_pklbz2('sonic_results_script', sonic_results)

    print('\n\n\n EVALUATION COMPLETE!')
    print(f'Failed Query IDs: \n {failed}')




    