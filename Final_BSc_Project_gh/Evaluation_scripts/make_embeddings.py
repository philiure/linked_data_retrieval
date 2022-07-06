#! /usr/bin/env python3

import bz2
from sentence_transformers import SentenceTransformer
import pickle
import gc
import sys
import os
import time
import re
import numpy as np

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


########### MAKING EMBEDDINGS ###############

print('Loading Data...')
vdocs = pickle.load(bz2.BZ2File('/scratch/work/library_vdocs_12M.pkl.bz2', 'rb'))

len_vdocs = len(vdocs)

# BERT models

model_asym = SentenceTransformer('msmarco-distilbert-base-dot-prod-v3', device='cuda')
model_asym.max_seq_length = 100

print('Starting Document Embeddings Encoding\n')

doc_embedding_asym = []

time_per_encoding = []

for id in range(len(vdocs)) :
    # encode_multi_process
    time_start = time.time()
    embedding = model_asym.encode(vdocs[id], batch_size=50)
    time_end = time.time()

    time_encoding = time_end - time_start
    time_per_encoding.append(time_encoding)
    doc_embedding_asym.append(embedding)

print('\nRun Complete!')
print(f'Mean encoding time: {round(np.mean(time_per_encoding)*1000,4)} ms/doc encoding')

print('\nSaving Embeddings...')
# save doc emberddings
save_file_pklbz2('doc_embed_asym', doc_embedding_asym)

print('\nSaved!')
print_sizes_objects([doc_embedding_asym])
print_sizes_files(['doc_embed_asym.pkl.bz2'])

print('\nDone Encoding.')
