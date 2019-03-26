import os, os.path
import pandas as pd
from whoosh.fields import Schema, TEXT, KEYWORD, ID
from whoosh.analysis import StemmingAnalyzer, RegexTokenizer, LowercaseFilter, StopFilter
from whoosh.index import create_in
from whoosh import index
from whoosh.qparser import QueryParser
import pprint

class whooshIndexer:
    def __init__(self):
        self.analyzer = RegexTokenizer() | LowercaseFilter() | StopFilter()
        self.schema =  Schema(Description=TEXT(stored=True, analyzer=self.analyzer),
                            Site=TEXT(stored=True), 
                            Area=KEYWORD(stored=True),
                            WorkCenter=ID(stored=True))
                               
                               ### lowercases, and stem
        if not os.path.exists("index"):   ##make an index folder if one does not exist
            os.mkdir("index")
            index.create_in("index", self.schema)
        self.ix = index.open_dir("index") 

    def createIndex(self):
        """ creates the index """
        writer = self.ix.writer()
        df = pd.read_csv('D:\\Data\\testdata.csv')
        for _,row in df.iterrows():
            writer.add_document(Description = row[0],
                                Site=row[1], 
                                Area=row[2],
                                WorkCenter=row[3])
        writer.commit()
    
    def makeSearch(self, query: str):
        """ makes a search """
        qp = QueryParser("Description", schema=self.schema)
        q = qp.parse(query)
        total = []
        with self.ix.searcher() as searcher:
            results = searcher.search(q, limit=10)
            for i in range(results.scored_length()):
                print(results[i]['Description'])
                # print(results[i].highlights("Description"))
                temp = dict()
                
                temp["Description"] = results[i].highlights("Description")
                temp["Site"] = results[i]["Site"]
                total.append(temp)

        #print(total)
        return total



