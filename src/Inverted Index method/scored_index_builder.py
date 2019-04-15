import math
import numpy as np
import pandas as pd
import re
import string
import nltk
from collections import defaultdict


class Index:
    def __init__(self, tokenizer, stemmer = None, stopwords = None):
        
        self.tokenizer = tokenizer
        self.stemmer = stemmer
        self.index = defaultdict(list)
        self.documents = {}    # store indexed documents
        self.unique_id = 0     # intialize doc_id
        self.doclengths = {}   # store length of documents
        self.tot_length = 0    # sum the document lengths to compute average for bm25
        self.boost_dic = {}    # select relevant terms from documents for boosting
        if not stopwords:
            self.stopwords = set()
        else:
            self.stopwords = set(stopwords)

    # Add documents to index by words 
        # Document is converted to lower case, punctuations & stop-words removed
        # the words are stemmed and indexed. 
    # The index will be a dictionary containing words as keys & [doc-id, term_frequency] as value.
        # Inverted index format --> {word1:[(doc_id, tf), (doc_id,tf),..], word2:[(),(),...]}

    def tokenize(self, text):
        '''Tokenize text into words'''
        tokens = self.tokenizer(text)
        return tokens
    
    def clean_and_stem(self, tokens):
        '''converts to lower case, remove stopwords & punctuations, stem the words'''
        tokens = [t.lower() for t in tokens \
                    if t not in self.stopwords \
                    and re.match('^[A-Za-z]{2,}|^[A-Za-z]-[0-9]+', t)]
        if self.stemmer:
            tokens = [self.stemmer.stem(t) for t in tokens]
        return tokens 

    def pos_tag(self,tokens):
        '''POS-Tag the words, add the NNP (proper nouns) & NN (nouns) 
           to a dictionary for boosting those terms in query'''
        tags = nltk.pos_tag(tokens)
        for tag in tags:
            if tag[1] == 'NNP' and tag[0] not in self.boost_dic:
                self.boost_dic[tag[0]] = 2.0   
        else:
            if tag[1] == 'NN' and tag[0] not in self.boost_dic:
                self.boost_dic[tag[0]] = 1.5

    def add_doc(self, document):
        '''adds documents to the index'''
        tokens = self.tokenize(document)
        self.pos_tag(tokens)
                
        tokens = self.clean_and_stem(tokens)

        for token in tokens:    
            tf = tokens.count(token)    # counting term frequency
            if self.unique_id not in [idx[0] for idx in self.index[token]]:
                self.index[token].append((self.unique_id, tf))
                                
        self.documents[self.unique_id] = document  # id the document
        self.unique_id += 1    
        self.doclengths[self.unique_id] = len(document)  # add the length to id of document
        self.tot_length += len(document)                 # sum document_length for bm25

    def count_docs(self):
        '''counts the the total number of documents in the corpus'''
        doc_count = 1 + self.unique_id
        return doc_count

    # Search matches for query 
        # Query is converted to tokens, lower-case and stemmed. 
        # Query_vec dictionary is created with tokens as keys and tf-idfs as values
              
    def query_vectorize(self, query):
        '''vectorizes and scores the given query with selected terms boosted'''
        tokens = self.tokenize(query)
        tokens = self.clean_and_stem(tokens)       
        doc_count = self.count_docs()
        query_vec = {}
        
        for token in tokens:
            boost = 1.0 
            if token in self.index:
                if token in self.boost_dic:
                    boost = self.boost_dic[token]
                query_tf = 1 + math.log(tokens.count(token))
                query_idf = math.log(doc_count/len(self.index.get(token)))
                query_vec[token] = query_tf*query_idf*boost 
        return tokens, query_vec

    def query_lookup(self, query, k=1.2, b=0.75, scoring='bm25'):
        '''looks in the index for specified words in the query,
            extract the matching (doc_id,tf) and doc_len to dictionary'''
        tokens, query_vec = self.query_vectorize(query)      
        score_list = defaultdict(list)
        doc_count = self.count_docs()
        
        for token in tokens:           
            if token in self.index:
                self.score_doc(token, k, b, scoring, doc_count, query_vec, score_list)

        self.rank_docs(query, score_list, scoring)
        
    def score_doc(self, token, k, b, scoring, doc_count, query_vec, score_list):
        '''calculates the scores for each matching token-document pair and
            creates a ditionary with doc as key and list of scores as values'''
        doc_freq = len(self.index.get(token))
        
        for idx,tf in self.index.get(token):
            
            if scoring=='classic':
                idf = 1.0 + math.log(doc_count/(doc_freq+1.0))
                score_list[idx].append(query_vec[token]*idf*np.sqrt(tf)/np.sqrt(self.doclengths[idx]))

            if scoring=='bm25':
                idf = math.log(1.0 + (doc_count - doc_freq + 0.5)/(doc_freq+0.5))
                L = np.divide(self.doclengths[idx], np.divide(self.tot_length, doc_count))
                tf_norm = np.divide((k+1)*tf, tf+k*(1.0-b+b*L))
                score_list[idx].append(query_vec[token]*tf_norm*idf)
    

    def rank_docs(self, query, score_list, scoring):
        '''sums the scores and ranks the matching docs and returns the top 3'''
        matches = self.tokenize(query)
        matches = self.clean_and_stem(matches)
        print(matches, '\n')
        print('The input log is:', '\n')
        self.highlight_many(query, matches)

        total_scores = {}
        for key,val in score_list.items():
            total_scores[key] = round(sum(val),3)
        print('The top 3 similar documents are: ', '\n')
        
        for item in sorted(total_scores.items(), key=lambda x: -x[1])[1:4]:
            print('DOC_ID:', item[0], scoring, ':', item[1], '\n')
            self.highlight_many(self.documents[item[0]], matches)

    def highlight_many(self, text, keywords):
        replacement = lambda match: "\033[91m" + match.group() + "\033[39m"
        text = re.sub("|".join(map(re.escape, keywords)), replacement, text, flags=re.I)
        print(text)
            
