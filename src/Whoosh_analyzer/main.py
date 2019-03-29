
import sys
import pandas as pd

#schema = Schema(Description=TEXT(stored=True, analyzer=self.analyzer),
 #               Site=TEXT(stored=True), 
  #              Area=KEYWORD(stored=True),
   #             WorkCenter=ID(stored=True))


# read data from the file specified
if len(sys.argv) < 2:
    print("Please provide input file path")
    sys.exit()
  
log = pd.read_excel(sys.argv[1])
print(log.head())