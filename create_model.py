# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 17:36:18 2019

@author: VIJAY
"""

import pandas as pd
import re
import csv, xgboost
import string
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, svm
from sklearn import ensemble
from classifier_utility import train_model, countVectorizer, wordLevelTfIdf, ngramLevelTfIdf, characterLevelTfIdf

#Following is the argument line code

""""
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="This program create models from the set of documents.")
    parser.add_argument("-i", "--input_file", help="Input file for the files to be loaded")
    args = parser.parse_args()

    input_file = args.input_file
"""

input_file = 'finaldata.csv'

labels, texts = [], []

#reading training data from file
csvFile  = open(input_file, "r")
entries = csv.reader(csvFile)

for row in entries:    
    temp = ''.join(row)
    if temp.strip() == '':
        continue
    x=temp.split(' | ');
    l=len(x)
    labels.append(x[l-1])   
    temp1=''.join(x[:l-1])
    #spliting based on the alphanumeric word
    content = re.split(r'\W+', temp1)
    #removing punctuation from data    
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in content]
    #removing empty string from list of strings   
    filter(None, stripped)
    print(stripped)
    texts.append(" ".join(stripped[:]))
csvFile.close() 
    
trainDF = pd.DataFrame()
trainDF['text'] = texts
trainDF['label'] = labels

# split the dataset into training and validation datasets 
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'], test_size=0.2)

# label encode the target variable 
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)

# transform the training and validation data using count vectorizer object
xtrain_count, xvalid_count =  countVectorizer(trainDF['text'], train_x, valid_x )
  
# word level tf-idf
xtrain_tfidf, xvalid_tfidf = wordLevelTfIdf(trainDF['text'], train_x, valid_x)

# ngram level tf-idf 
xtrain_tfidf_ngram, xvalid_tfidf_ngram =  ngramLevelTfIdf(trainDF['text'], train_x, valid_x)

# characters level tf-idf
xtrain_tfidf_ngram_chars, xvalid_tfidf_ngram_chars =  characterLevelTfIdf(trainDF['text'], train_x, valid_x) 

# Naive Bayes on Count Vectors
accuracy, k_score, k_std = train_model(naive_bayes.MultinomialNB(), xtrain_count, train_y, xvalid_count, valid_y)
print ( "Naive Bayes, Count Vectors: ")
print("Accuracy= %0.2f, K-fold Cross Validation Accuracy: %0.2f (+/- %0.2f)"% (accuracy, k_score, k_std))

# Naive Bayes on Word Level TF IDF Vectors
accuracy, k_score, k_std  = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xvalid_tfidf, valid_y)
print ("Naive Bayes, WordLevel TF-IDF: ")
print("Accuracy= %0.2f, K-fold Cross Validation Accuracy: %0.2f (+/- %0.2f)"% (accuracy, k_score, k_std))

# Naive Bayes on Ngram Level TF IDF Vectors
accuracy, k_score, k_std  = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram, valid_y)
print ("Naive Bayes, N-Gram Vectors: ")
print("Accuracy= %0.2f, K-fold Cross Validation Accuracy: %0.2f (+/- %0.2f)"% (accuracy, k_score, k_std))

# Naive Bayes on Character Level TF IDF Vectors
accuracy, k_score, k_std  = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars, valid_y)
print ("Naive Bayes, CharLevel Vectors: ")
print("Accuracy= %0.2f, K-fold Cross Validation Accuracy: %0.2f (+/- %0.2f)"% (accuracy, k_score, k_std))

# Linear Classifier on Count Vectors
accuracy, k_score, k_std = train_model(linear_model.LogisticRegression(), xtrain_count, train_y, xvalid_count, valid_y)
print ("Logistic Regression, Count Vectors: ")
print("Accuracy= %0.2f, K-fold Cross Validation Accuracy: %0.2f (+/- %0.2f)"% (accuracy, k_score, k_std))

# Linear Classifier on Word Level TF IDF Vectors
accuracy, k_score, k_std = train_model(linear_model.LogisticRegression(), xtrain_tfidf, train_y, xvalid_tfidf, valid_y)
print ("Logistic Regression, WordLevel TF-IDF: ")
print("Accuracy= %0.2f, K-fold Cross Validation Accuracy: %0.2f (+/- %0.2f)"% (accuracy, k_score, k_std))

# Linear Classifier on Ngram Level TF IDF Vectors
accuracy, k_score, k_std = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram, valid_y)
print ("Logistic Regression, N-Gram Vectors: ")
print("Accuracy= %0.2f, K-fold Cross Validation Accuracy: %0.2f (+/- %0.2f)"% (accuracy, k_score, k_std))

# Linear Classifier on Character Level TF IDF Vectors
accuracy, k_score, k_std = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars, valid_y)
print ("Logistic Regression, CharLevel Vectors: ")
print("Accuracy= %0.2f, K-fold Cross Validation Accuracy: %0.2f (+/- %0.2f)"% (accuracy, k_score, k_std))

# SVM on Ngram Level TF IDF Vectors
accuracy, k_score, k_std = train_model(svm.SVC(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram, valid_y)
print ("SVM, N-Gram Vectors: ")
print("Accuracy= %0.2f, K-fold Cross Validation Accuracy: %0.2f (+/- %0.2f)"% (accuracy, k_score, k_std))

# RF on Count Vectors
accuracy, k_score, k_std = train_model(ensemble.RandomForestClassifier(), xtrain_count, train_y, xvalid_count,valid_y)
print ("Random Forest, Count Vectors: ")
print("Accuracy= %0.2f, K-fold Cross Validation Accuracy: %0.2f (+/- %0.2f)"% (accuracy, k_score, k_std))

# RF on Word Level TF IDF Vectors
accuracy, k_score, k_std = train_model(ensemble.RandomForestClassifier(), xtrain_tfidf, train_y, xvalid_tfidf, valid_y)
print ("Random Forest, WordLevel TF-IDF: ")
print("Accuracy= %0.2f, K-fold Cross Validation Accuracy: %0.2f (+/- %0.2f)"% (accuracy, k_score, k_std))
