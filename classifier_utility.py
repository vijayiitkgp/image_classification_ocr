# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 00:36:16 2019

@author: VIJAY
"""

from sklearn import  metrics
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import cross_val_score


def train_model(classifier, feature_vector_train, label, feature_vector_valid, valid_y, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    scores = cross_val_score(classifier, feature_vector_train, label, cv=5)
    
    return metrics.accuracy_score(predictions, valid_y), scores.mean(), scores.std() * 2


    
def countVectorizer(input, train_x, valid_x):
    
    # create a count vectorizer object
    count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
    count_vect.fit(input)

    # transform the training and validation data using count vectorizer object
    xtrain_count =  count_vect.transform(train_x)
    xvalid_count =  count_vect.transform(valid_x) 
    
    return xtrain_count, xvalid_count

def wordLevelTfIdf(input, train_x, valid_x):

    # word level tf-idf
    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
    tfidf_vect.fit(input)
    xtrain_tfidf =  tfidf_vect.transform(train_x)
    xvalid_tfidf =  tfidf_vect.transform(valid_x)
    
    return xtrain_tfidf, xvalid_tfidf

def ngramLevelTfIdf(input, train_x, valid_x) :    

    # ngram level tf-idf 
    tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
    tfidf_vect_ngram.fit(input)
    xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
    xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)
    
    return xtrain_tfidf_ngram, xvalid_tfidf_ngram

def characterLevelTfIdf(input, train_x, valid_x):
    
    # characters level tf-idf
    tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
    tfidf_vect_ngram_chars.fit(input)
    xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x) 
    xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x) 
    
    return xtrain_tfidf_ngram_chars, xvalid_tfidf_ngram_chars
