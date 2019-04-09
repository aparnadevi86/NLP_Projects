# import packages
import os, os.path
import sys
import pandas as pd
from whoosh import qparser
from whoosh.index import create_in
from whoosh.fields import Schema, TEXT, KEYWORD, ID
from whoosh.analysis import StemmingAnalyzer, RegexTokenizer, LowercaseFilter, StopFilter
from whoosh.qparser import QueryParser, MultifieldParser
from myTokenizers import myRegexTokenizer
import pprint

# import time
# start = time.time()

# create custom analyzer
my_analyzer = myRegexTokenizer() | LowercaseFilter() | StopFilter() 

# define schema
schema = Schema(Description=TEXT(stored=True, analyzer=my_analyzer),
                Site=TEXT(stored=True, sortable=True), 
                Area=KEYWORD(stored=True),
                WorkCenter=ID(stored=True)
                )

# create new index directory and index
if not os.path.exists('indexdir'):
    os.mkdir('indexdir')
    
ix = create_in('indexdir', schema, indexname='trialindex2')  # create new index in the dir
writer = ix.writer()

# read data to be indexed
if len(sys.argv) < 2:
    print("Please provide input file path")
    sys.exit()
  
df = pd.read_csv(sys.argv[1])
df.columns = ['Description', 'Site', 'Area', 'WorkCenter', 'WorkUnit']
df = df.loc[(df['Site'] == 'SEC-4') & (df['Description'].notnull())]
df = df.fillna(value= "None")

# add documents(rows) to the index as per schema
for index,row in df.iterrows():
    writer.add_document(Description = row[0],
                        Site=row[1], 
                        Area=row[2],
                        WorkCenter=row[3])
writer.commit()

# end = time.time()
# print(end - start)