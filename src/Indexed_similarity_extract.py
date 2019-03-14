from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import math
import string

class Index:
    def __init__(self, tokenizer, stemmer = None, stopwords = None):
        
        self.tokenizer = tokenizer
        self.stemmer = stemmer
        self.index = defaultdict(list)
        self.documents = {}    # store indexed documents
        self.unique_id = 0     # intialize doc_id
        if not stopwords:
            self.stopwords = set()
        else:
            self.stopwords = set(stopwords)

    # add documents to index by words (document is converted to lower case, punctuations & 
    # stop-words removed, then the words are stemmed and indexed. The index will be a 
    # dictionary containing words as keys and doc-id & fequency of word (term_frequncy) as value.
    # Inverted index format --> {word1:[(doc_id, tf), (doc_id,tf),..], word2:[(),(),...]}

    def add(self, document):
        for token in [t.lower() for t in self.tokenizer(document) if t not in string.punctuation]:
            if token in self.stopwords:
                continue
                
            if self.stemmer:
                token = self.stemmer.stem(token)
            
            tf = document.lower().count(token)  
            if self.unique_id not in [idx[0] for idx in self.index[token]]:
                self.index[token].append((self.unique_id, tf))
                                
        self.documents[self.unique_id] = document
        self.unique_id += 1    


    # search matches for new_document (New document is converted to tokens, lower-case and stemmed.
    # For each token, the matching (doc_id,tf) is extracted. A new dictionary (doc_scoring) is
    # created with doc_id and calculated tf_idfs. The tf_idfs for each document are summed and 
    # the documents are ranked by their score, the top 3 ranked documents are printed.

    def lookup(self, newdocument):
        tokens = [t.lower() for t in self.tokenizer(newdocument) if t not in self.stopwords and t not in string.punctuation]
                
        if self.stemmer:
            tokens = [self.stemmer.stem(t) for t  in tokens]
        
        doc_scoring = defaultdict(list)
        for token in tokens:
            for k,v in self.index.get(token):
                IDF = math.log(self.unique_id/len(self.index.get(token)))
                doc_scoring[k].append((1+math.log(0.01+v))*IDF)
        
        scores = {}
        for k,v in doc_scoring.items():
            scores[k] = sum(v)

        print('The top 3 similar documents are: ', '\n')    
        for item in sorted(scores.items(), key=lambda x: -x[1])[:3]:
            print('DOC_ID:',item[0], ', TF-IDF_Score:', item[1], '\n', self.documents[item[0]], '\n')
            
index = Index(word_tokenize, PorterStemmer(), stopwords.words("english"))