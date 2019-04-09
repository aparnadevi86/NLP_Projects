import sys
import math
import numpy as np
import pandas as pd
import string
import nltk
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import pickle as pkl

# import time
# start = time.time()

class Index:
    def __init__(self, tokenizer, stemmer = None, stopwords = None):
        
        self.tokenizer = tokenizer
        self.stemmer = stemmer
        self.index = defaultdict(list)
        self.documents = {}    # store indexed documents
        self.unique_id = 0     # intialize doc_id
        self.doclengths = {}   # store length of documents
        self.tot_length = 0    # sum the document lengths to compute average for bm25
        self.boost_dic = {}

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
        '''Tokenize the text into words'''
        tokens = self.tokenizer(text)
        return tokens
    
    def cleanStem(self, tokens):
        '''Remove stopwords & punctuations, stem the words'''
        tokens = [t.lower() for t in tokens if t not in self.stopwords and t not in string.punctuation]
        if self.stemmer:
            tokens = [self.stemmer.stem(t) for t in tokens]
        return tokens 

    def posTag(self,tokens):
        '''POS-Tag the words, add the NNP (proper nouns) & NN (nouns) 
           to a dictionary for boosting those terms in query'''
        tags = nltk.pos_tag(tokens)
        for tag in tags:
            if tag[1] == 'NNP' and tag[0] not in self.boost_dic:
                self.boost_dic[tag[0]] = 3.0
    
        else:
            if tag[1] == 'NN' and tag[0] not in self.boost_dic:
                self.boost_dic[tag[0]] = 2.0   

    def add(self, document):
        '''adds documents to the index'''
        tokens = self.tokenize(document)
        self.posTag(tokens)
                
        tokens = self.cleanStem(tokens)

        for token in tokens:    
            tf = tokens.count(token)    # counting term frequency
            if self.unique_id not in [idx[0] for idx in self.index[token]]:
                self.index[token].append((self.unique_id, tf))
                                
        self.documents[self.unique_id] = document  # id the document
        self.unique_id += 1    

        self.doclengths[self.unique_id] = len(document)  # add the length to id of document
        self.tot_length += len(document)                 # sum document_length for bm25


    # Search matches for query 
        # Query is converted to tokens, lower-case and stemmed. 
        # Query_vec dictionary is created with tokens as keys and tf-idfs as values

    # For each token:
        # The matching (doc_id,tf) and doc_len are extracted. 
        # 2 types of scoring are calculated - Classical Lucene scoring & BM25 scoring
        # A new dictionary (doc_scoring) is created with doc_id and calculated tf_idfs. 
        # The tf_idfs for each document are summed and appended
        # The documents are ranked by their score, the top 3 ranked documents are printed.

    def lookup(self, query, k = 1.2, b = 0.75):
        '''looks in the index for specified words in the query'''
        tokens = self.tokenize(query)
        tokens = self.cleanStem(tokens)
        
        doc_count = self.unique_id+1        # total number of docs
        
    # creating query_vector with tf-idfs of each token in query    
        query_vec = {}
        
        for token in tokens:
            if token in self.index:
                boost = 1.0
                if token in self.boost_dic:
                    boost = self.boost_dic[token]
                query_tf = 1 + math.log(tokens.count(token))
                query_idf = math.log(doc_count/len(self.index.get(token)))
                query_vec[token] = query_tf*query_idf*boost
        
    # ranking docs
        lucene_scoring = defaultdict(list)  # initialize dictionary for lucene score per doc as key
        bm25_scoring = defaultdict(list)    # initialize  dictionary for BM25 score per doc as key
        
        for token in tokens:
            if token in self.index:
                doc_freq = len(self.index.get(token))       # number of matching documents for token
                
                for idx,tf in self.index.get(token):
                   
                    # for classic lucene scoring
                    idf_lucene = 1.0 + math.log(doc_count/(doc_freq+1.0))
                    lucene_scoring[idx].append(query_vec[token]*idf_lucene*np.sqrt(tf)/np.sqrt(self.doclengths[idx]))
                                        
                    # for bm25_scoring
                    idf_bm25 = math.log(1.0 + (doc_count-doc_freq+0.5)/(doc_freq+0.5))
                    L = np.divide(self.doclengths[idx], np.divide(self.tot_length, doc_count))
                    tf_norm = np.divide((k+1)*tf, tf+k*(1.0-b+b*L))
                    bm25_scoring[idx].append(query_vec[token]*tf_norm*idf_bm25)
        
        # sorting documents by classic lucene score
        lucene_scores = {}
        for key,val in lucene_scoring.items():
            lucene_scores[key] = round(sum(val),3)
      
        print('The top 3 similar documents are: ', '\n')
        for item in sorted(lucene_scores.items(), key=lambda x: -x[1])[1:4]:
            print('DOC_ID:',item[0], ', TF-IDF_Score:', item[1],
                  '\n', self.documents[item[0]], '\n')
            
        # sorting documents by BM25 score    
        bm25_scores = {}
        for key,val in bm25_scoring.items():
            bm25_scores[key] = round(sum(val),3)
      
        print('The top 3 similar documents are: ', '\n')
        for item in sorted(bm25_scores.items(), key=lambda x: -x[1])[1:4]:
            print('DOC_ID:',item[0], ', BM25_Score:', item[1],
                  '\n', self.documents[item[0]], '\n')
            
index = Index(word_tokenize, PorterStemmer(), stopwords.words("english"))


# read data from the file specified
# if len(sys.argv) < 2:
#     print("Please provide input file path")
#     sys.exit()
# print("Starting execution...")  

# df = pd.read_csv(sys.argv[1])
# df.columns = ['Description', 'Site', 'Area', 'WorkCenter', 'WorkUnit']
# df = df.fillna(value= 'None')
# data = df.loc[(df['Site'] == 'SEC-4') & (df['Description'].notnull()), 'Description']

# for d in data:
#     index.add(d)

# # end = time.time()
# # print(end - start)

# # save the created index
# with open("indexdir/pkl_indices/dict.pickle","wb") as f:
#     pkl.dump(index, f, protocol=pkl.HIGHEST_PROTOCOL)

# open pickled index
with open('indexdir/pkl_indices/dict.pickle','rb') as f:
    index_saved = pkl.load(f)

# lookup similar documents   
query = "  "
index_saved.lookup(query)

