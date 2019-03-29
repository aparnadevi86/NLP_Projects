# import packages
import os, os.path
import pandas as pd
from whoosh import qparser
from whoosh.index import create_in
from whoosh.fields import Schema, TEXT, KEYWORD, ID
from whoosh.analysis import StemmingAnalyzer, RegexTokenizer, LowercaseFilter, StopFilter
from whoosh.qparser import QueryParser, MultifieldParser
import pprint

import time
start = time.time()

# create custom analyzer
my_analyzer = RegexTokenizer() | LowercaseFilter() | StopFilter() 

# define schema
schema = Schema(Description=TEXT(stored=True, analyzer=my_analyzer),
                Site=TEXT(stored=True, sortable=True), 
                Area=KEYWORD(stored=True),
                WorkCenter=ID(stored=True)
                )

# create new index directory and index
if not os.path.exists('indexdir'):
    os.mkdir('indexdir')
    
ix = create_in('indexdir', schema, indexname='trialindex')  # create new index in the dir
writer = ix.writer()

# read data to be indexed
df = pd.read_csv('D:\\Data\\RPO_Log.csv')
df = df.fillna(value= "None")

# add documents(rows) to the index as per schema
for index,row in df.iterrows():
    writer.add_document(Description = row[0],
                        Site=row[1], 
                        Area=row[2],
                        WorkCenter=row[3])
writer.commit()

# parse query for searching index    
query = QueryParser(
                     'Description',
                      schema = ix.schema,
                      #group = qparser.OrGroup.factory(0.9)
                    )
qy = query.parse('operation load')
total = []
with ix.searcher() as searcher:
    results = searcher.search(qy)#), limit=None)  
    for i in range(results.scored_length()):
                print(results[i]['Description'])
                temp = dict()
                # print(results[i].highlights("Description"))
                temp["Description"] = results[i]["Description"]
                temp["Site"] = results[i]["Site"]
                total.append(temp)

    pprint.pprint(total)

end = time.time()
print(end - start)