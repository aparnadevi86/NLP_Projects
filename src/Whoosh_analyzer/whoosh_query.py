import whoosh.index as index
from whoosh import qparser
from whoosh.qparser import QueryParser, MultifieldParser
import pprint

ix = index.open_dir('indexdir', indexname= 'trialindex2')

# parse query for searching index    
query = QueryParser(
                     'Description',
                      schema = ix.schema,
                      group = qparser.OrGroup.factory()
                    )
sent = 'maintainance \
 *procedure Drain reformate and purging ongoing: load open \
        round check at jetty area, all normal - adsorber \
        V02A/B at 0.4 bar: keep monitor'

qy = query.parse(sent)
total = []
with ix.searcher() as searcher:
    results = searcher.search(qy, limit=5)  
    for i in range(results.scored_length()):
                # print(results[i]['Description'], 'Score: ', results[i].score)
                temp = dict()
                # print(results[i].highlights("Description"))
                temp['Rank'] = i
                temp["Description"] = results[i]["Description"]
                temp["Site"] = results[i]["Site"]
                temp['BM25 Score'] = results[i].score
                total.append(temp)

    pprint.pprint(total)
