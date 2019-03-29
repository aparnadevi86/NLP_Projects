import whoosh.index as index
from whoosh import qparser
from whoosh.qparser import QueryParser, MultifieldParser
import pprint


ix = index.open_dir('indexdir', indexname= 'trialindex')
# parse query for searching index    
query = QueryParser(
                     'Description',
                      schema = ix.schema,
                      group = qparser.OrGroup.factory(0.5)
                    )
qy = query.parse('operation load steam continue')
total = []
with ix.searcher() as searcher:
    results = searcher.search(qy, limit=20)  
    for i in range(results.scored_length()):
                # print(results[i]['Description'], 'Score: ', results[i].score)
                temp = dict()
                # print(results[i].highlights("Description"))
                temp['Retrieved_order'] = i
                temp["Description"] = results[i]["Description"]
                temp["Site"] = results[i]["Site"]
                temp['BM25 Score'] = results[i].score
                total.append(temp)

    pprint.pprint(total)

    # whoosh.scoring.FunctionWeighting(fn)