from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import math

class Index:
    def __init__(self, tokenizer, stemmer = None, stopwords = None):
        
        self.tokenizer = tokenizer
        self.stemmer = stemmer
        self.index = defaultdict(list)
        self.documents = {}
        self.unique_id = 0
        if not stopwords:
            self.stopwords = set()
        else:
            self.stopwords = set(stopwords)

    
    def add(self, document):
        for token in [t.lower() for t in self.tokenizer(document)]:
            if token in self.stopwords:
                continue
                
            if self.stemmer:
                token = self.stemmer.stem(token)
            
            term_frequency = 0
            if document.count(token):
                term_frequency = 1 + math.log(document.count(token))
            
          
            if self.unique_id not in [x[0] for x in self.index[token]]:
                
                self.index[token].append((self.unique_id,term_frequency))
                                
        self.documents[self.unique_id] = document
        self.unique_id += 1    

    
    def lookup(self, newdocument):
        tokens = [t.lower() for t in self.tokenizer(newdocument) if t not in self.stopwords]
                
        if self.stemmer:
            tokens = [self.stemmer.stem(t) for t  in tokens]
        
        doc_scoring = defaultdict(list)
        for word in tokens:
            for k,v in self.index.get(word):
                IDF = math.log(self.unique_id/len(self.index.get(word)))
                doc_scoring[k].append(round(v*IDF,3))
        
        scores = {}
        for k,v in doc_scoring.items():
            scores[k] = sum(v)
            
        for item in sorted(scores.items(), key=lambda x: -x[1])[:3]:
            print('DOC_ID:',item[0], ', TF-IDF_Score:', item[1], '\n', self.documents[item[0]], '\n')
            
index = Index(word_tokenize, PorterStemmer(), stopwords.words("english"))