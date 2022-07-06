#! /usr/bin/env python3

# Imports: bz2, sentence_transformers, numpy, pickle, gc, sys, os, time, scann, math, re, pandas

import bz2
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import numpy as np
import pickle
import gc
import sys
import os
import time
import scann
import math
import re
import pandas as pd

############ FUNCTIONS ##############

def print_sizes_objects(objects):
    for o in objects:
        size = sys.getsizeof(o)
        print(f'Object: {type(o)}\n{size} Bytes\n{size*(10**-9)} GB')
        print('---------------------------')


def print_sizes_files(files):
    for f in files:
        size = os.path.getsize(f)
        print(f'Filename: {f}\n{size} Bytes\n{size*(10**-9)} GB')
        print('---------------------------')


def save_file_pklbz2(filename: str, data):
    gc.disable()
    with bz2.BZ2File(f'{filename}.pkl.bz2', 'wb') as bzf:
        pickle.dump(data, bzf, protocol=-1)
    gc.enable()

def clean_subject(subj):
    resource = re.split('#|/',subj)[-1]
    return f'<dbpedia:{resource}>'


print('Vector Search\n')

########## VECTOR SEARCH #############


# LOAD FILES
print('Loading queries and index...')
doc_embedding_asym = pickle.load(bz2.BZ2File('/nfs/scratch/renzen/work/doc_embed_asym.pkl.bz2', 'rb'))
# Initialize ScaNN
k = int(math.sqrt(len(doc_embedding_asym)))
dim_per_block = 2
reorder = 50

# ADDED .astype(np.uint8)

#doc_embedding_asym /= np.linalg.norm(doc_embedding_asym, axis=1)[:, np.newaxis]

norm_vector = np.linalg.norm(doc_embedding_asym, axis=1, keepdims=True)
doc_embedding_asym /= norm_vector

searcher_asym = scann.scann_ops_pybind.builder(doc_embedding_asym, 10, "dot_product").tree(
    num_leaves=k, num_leaves_to_search=300, training_sample_size=400000).score_ah(
    2, anisotropic_quantization_threshold=0.2).reorder(100).build()

del doc_embedding_asym
gc.collect()
index = pickle.load(bz2.BZ2File('/nfs/scratch/renzen/work/10-2015_DBpedia_12M.pkl.bz2', 'rb'))
eval_queries = pickle.load(bz2.BZ2File('/nfs/scratch/renzen/work/eval_queries.pkl.bz2', 'rb'))
print('Loading Complete\n')

# total_num_queries = len(eval_queries)

failed_search = []
failed_query_ids = []

print('Starting Search\n')

# BERT models

model_asym = SentenceTransformer('msmarco-distilbert-base-dot-prod-v3')
model_asym.max_seq_length = 100


retrieved_results = []
query_ids = []
times = []

# SEARCH
for idx, row in eval_queries.iterrows():
    
    gc.collect()

    query_num = idx
    query = row['query']
    relevant_entities = row['relevant_entities']
    very_relevant_entities = row['very_relevant_entities']
    len_relevant = len(relevant_entities)
    len_very_relevant = len(very_relevant_entities)
    n_total = len_relevant+len_very_relevant


    #ASSYMETRIC SEMANTIC VECTOR SEARCH
    try:
        start = time.time()
        neighbors, distances = searcher_asym.search(model_asym.encode(query), final_num_neighbors=n_total, pre_reorder_num_neighbors=reorder)
        end = time.time()
        search_time = end - start
        retrieved = []
        for id in neighbors:
            try:
                sub = index[id]['subject']
                retrieved.append(clean_subject(sub))
            except:
                print(f'ERROR Retrieve Index {query_num} - Failed to retrieve ID: {id} - {index[id]}')
                failed_query_ids.append(id)
        retrieved_results.append(retrieved)
        query_ids.append(query_num)
        times.append(search_time)
        
    except:
        print(f'ERROR Search Failed - ID {query_num}')
        failed_search.append(query_num)

del index
del eval_queries
del model_asym
gc.collect()

#### Saving Evaluation
print(f'\n\n\nSAVING...')

dic = {'id': query_ids, 'retrieved':retrieved_results, 'query_time':times}

vector_results = pd.DataFrame(dic)

save_file_pklbz2('vector_results_script', vector_results)

print('\n\n\n EVALUATION COMPLETE!')

print(f'Total Failed Searches :\n{failed_search}')
print(f'Failed Query IDs: \n {failed_query_ids}')



