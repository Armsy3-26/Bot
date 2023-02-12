#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 12:21:53 2023

@author: armsy326
"""

import gensim
from gensim import corpora
from dataset import data


sentence  = input("Enter something: ")

corpus  = []

def load_dataset():
    for queries in data['questions']:

        corpus.append(queries)

load_dataset()

###use simple preprocess

sentence = gensim.utils.simple_preprocess(sentence)
token_corpus   =[gensim.utils.simple_preprocess(doc) for doc in corpus]

##create a dictionary #take corpus as an argument

dictionary = corpora.Dictionary(token_corpus)

#convert to bow

sent_bow  = dictionary.doc2bow(sentence)
corpus_bow = [dictionary.doc2bow(doc) for doc in token_corpus]


#convert to tf-idf

tfidf = gensim.models.TfidfModel(corpus_bow, dictionary=dictionary)

sent_tfidf = tfidf[sent_bow]

corpus_tfidf  = tfidf[corpus_bow]

##check similarity 

similarity  = [gensim.matutils.cossim(sent_tfidf, doc) for doc in corpus_tfidf]

#filtered_sent  = [doc for sim,doc in zip(similarity, corpus) if sim >= 0.5]

accurate_queries = {}
threshold  = 0.5 
def sort_data():
    for sim,doc in zip(similarity, corpus):
        if sim >= threshold:
            accurate_queries[doc] = sim
    #sorting the dictionaries
    if len(accurate_queries) > 1:
        sorted_queries = sorted(accurate_queries.items(), key=lambda x: x[1], reverse=True)
        print(accurate_queries)
        print(sorted_queries[0][0])
    else:
        key = list(accurate_queries.keys())[0]

        print(key)

sort_data()