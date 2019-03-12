from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# create Inverted Index
class Index:
    def __init__(self, tokenizer, stemmer = None, stopwords = None):
        
        self.tokenizer = tokenizer      # use nltk tokenizer
        self.stemmer = stemmer          # use nltk stemmer
        self.index = defaultdict(list)  # initialize index dictionary
        self.documents = {}             # initialize documents dictionary
        self.unique_id = 0              # initialize doc_id
        if not stopwords:               # set stopwords
            self.stopwords = set()
        else:
            self.stopwords = set(stopwords)
    
        
    def add(self, document):   # update index with words in document
        for token in [t.lower() for t in self.tokenizer(document)]:
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


    def lookup(self, newdocument):   # lookup index for matching words from new document
        tokens = [t.lower() for t in self.tokenizer(newdocument) if t not in self.stopwords]
                
        if self.stemmer:
            words = [self.stemmer.stem(t) for t  in tokens]
        
        match_documents = []         # create list of matching documents
        for word in words:
            match_documents.extend([self.documents.get(id,None) for id in self.index.get(word)])
        
        return match_documents

index = Index(word_tokenize, PorterStemmer(), stopwords.words("english"))