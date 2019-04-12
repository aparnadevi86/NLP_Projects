import sys
import pandas as pd
import pickle as pkl
from scored_index_builder import Index
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

index = Index(word_tokenize, PorterStemmer(), stopwords.words("english"))

# read data from the file-path specified
if len(sys.argv) < 2:
    print("Please provide input file path")
    sys.exit()

df = pd.read_csv(sys.argv[1])
df.columns = ['Description', 'Site', 'Area', 'WorkCenter', 'WorkUnit']
df = df.fillna(value= 'None')
data = df.loc[df['Description'].notnull(), 'Description']

print('indexing...')
# add docs to the index
for d in data:
    index.add_doc(d)

print('pickling....')
# save the created index
with open("indexdir/pkl_indices/idx_wboost_2.pickle","wb") as f:
    pkl.dump(index, f, protocol=pkl.HIGHEST_PROTOCOL)

print('index created!')