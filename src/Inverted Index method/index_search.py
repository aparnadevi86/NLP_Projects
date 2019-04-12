import pickle as pkl
from scored_index_builder import Index
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

index = Index(word_tokenize, PorterStemmer(), stopwords.words("english"))

# open pickled index
with open('indexdir/pkl_indices/idx_wboost.pickle','rb') as f:
    index_saved = pkl.load(f)

# lookup similar documents   
query = ""
index_saved.query_lookup(query)