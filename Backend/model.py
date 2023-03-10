#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 09:13:43 2023

@author: armsy326
"""

#just a file for random tests
import json
import gensim
from gensim import corpora
import random
from dataset import unknown_query_responses


with open('data.json') as f:
    #loading from a file using json.load()
    dataset = json.load(f)

corpus = []
corpus_index = []
text_similarity = {}

#have the data  as a list all of it
def load_corpus():
    i = 0
    for data in dataset['intents']:
        
        for patterns in data['patterns']:
            corpus.append(patterns)
            corpus_index.append(i)

        i = i+1

def check_similarity(query):
    #print("Here")
    text_similarity.clear()
    #create tokens for the query and the corpus
    sent   = gensim.utils.simple_preprocess(query)

    corpus_token = [gensim.utils.simple_preprocess(doc)for doc in corpus]

    #create a dictionary

    dictionary = corpora.Dictionary(corpus_token)

    #create bow from all

    sent_bow  = dictionary.doc2bow(sent)

    corpus_bow  = [dictionary.doc2bow(doc) for doc in corpus_token]

    #create a tfidf instance

    tfidf  = gensim.models.TfidfModel(corpus_bow, dictionary=dictionary)

    sent_tfidf = tfidf[sent_bow]

    corpus_tfidf  = tfidf[corpus_bow]


    #check similarity 

    similarity = [gensim.matutils.cossim(sent_tfidf, doc) for doc in corpus_tfidf]
    threshold  = 0.5
    for sim,index  in zip(similarity, corpus_index):

        if sim > threshold:
            text_similarity[sim] = index


def get_feedback():
    
    #process_corpus()
    if len(text_similarity) == 0:
        feedback = random.choices(unknown_query_responses)
        return feedback[0]
        
    elif len(text_similarity) > 1:
        #sort it , with highest similarity coming first
        sort = sorted(text_similarity.items(), key=lambda x: x[1], reverse=True)

        index = sort[0][1]

        intent_responses = dataset['intents'][index]['responses']

        response = random.choice(intent_responses)

        return response
    else:
        index = list(text_similarity.values())[0]
        intent_responses = dataset['intents'][index]['responses']

        response = random.choice(intent_responses)

        return response

