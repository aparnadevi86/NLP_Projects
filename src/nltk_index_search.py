from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# define Inverted Index
class Index:
    def __init__(self, tokenizer, stemmer = None, stopwords = None):
        
        self.tokenizer = tokenizer      # use nltk tokenizer
        self.stemmer = stemmer          # use nltk stemmer
        self.index = defaultdict(list)  # initialize index dictionary
        self.documents = {}             # initialize documents dictionary
        self.unique_id = 0            # start id
        if not stopwords:               # set stopwords
            self.stopwords = set()
        else:
            self.stopwords = set(stopwords)
    
        
    def add(self, document):   # create inverted indexing for document
        for token in [t.lower() for t in nltk.word_tokenize(document)]:
            if token in self.stopwords:  # remove stopwords
                continue
                
            if self.stemmer:             # stemming
                token = self.stemmer.stem(token)
            
            # add doc_id to word index for one time
            if self.unique_id not in self.index[token]:
                self.index[token].append(self.unique_id)

        # id the documents        
        self.documents[self.unique_id] = document
        self.unique_id += 1    


    def lookup(self, word):             # setup search method using created Index
        word = word.lower()             # preprocess input text
        if self.stemmer:
            word = self.stemmer.stem(word)
        # return id of matching documents
        return [self.documents.get(id,None) for id in self.index.get(word)]

index = Index(nltk.word_tokenize, PorterStemmer(), stopwords.words("english"))