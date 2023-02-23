#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 03:33:04 2023

@author: armsy326
"""

import gensim
from gensim import corpora

threshold = 0.5

sent = "Hello"

corpus  = [["fucks","Hello"]]
tfidf_corpus = []
#a fxn dedicated to incoming message

def preprocess_msg(msg: str)->str:
    #pass through a simple_preprocess

    sent  = gensim.utils.simple_preprocess(msg)

    #create a dictionary

    dictionary = corpora.Dictionary([sent])

    #convert to bow
    sent_bow  = dictionary.doc2bow(sent)

    #convert to tfidf 
    sent_tfidf  = gensim.models.TfidfModel([sent_bow], dictionary=dictionary)

    return sent_tfidf

def preprocess_corpus(corpus: list)->str:
    
    corpus  = [gensim.utils.simple_preprocess(doc) for doc in corpus]

    #create a dictionary object

    dictionary = corpora.Dictionary(corpus)

    corpus_bow = [dictionary.doc2bow(doc) for doc in corpus]

    #convert to term frequenct inverse document frequency

    tfidf  = gensim.models.TfidfModel(corpus_bow,dictionary=dictionary)

    corpus_tfidf = tfidf[corpus_bow]
    tfidf_corpus.append(corpus_tfidf)

def process_intents(corpus: list)->str:

    for corpus in corpus:
        
        preprocess_corpus(corpus)
process_intents(corpus)

def check_similarity():
    msg = preprocess_msg("Hello")
    print(msg)
    #for tfidf_tokens in tfidf_corpus:
        
        #check similarity using the cosine similarity fxn
        #offered by gensim by mautils.cossim
        
    #similarity = [gensim.matutils.cossim(msg, doc) for doc in tfidf_corpus]

    """for sim,doc in zip(similarity,corpus):

        if sim >= threshold:
            print(doc)

            break"""

check_similarity()

