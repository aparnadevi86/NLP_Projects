# NLP-project

## Objective
To pull similar documents given a query.

## Data
List of documents (rows in excel logs). 
Main field containing descriptive text to be analysed using NLP techniques. 
Additional fields for various categories (ordinal variables)

## Initial approach
- **Cleaning** 
    - The text is tokenized, converted to lower case with punctuations removed
    - Similar (usecase-specific) terms replaced using a dictionary
- **Vectorizing** 
    - Each document (each row of the descriptive column) will be converted to a vector based on the vocabulary list of the whole document corpus. 
    - In the end, the corpus will thus be converted to a matrix with rows as vectors
- **Querying**
    - A new query will  be converted to a vector based on the words identified from the document corpus.
- **Similarity calculation**
    - Cosine similarity is calculated by taking the dot product of the query vector with the vectorized matrix
    - The most similar documents will be the ones with highest values of the dot product
- Other modifications:
    - 3 types of vectorisations were done
        - *Binary vectorizer* (1 for word present/0 for word not present)
        - *Count vectorizer* (using term frequency values)
        - *TF-IDF vectorizer* (using TF-IDF values)
    - *Length Normalization* for varying length of documents 


## Advanced approach

- **Inverted Indexing**  
    - Create a unique document-id for each documents.
    - Create an index of words.
    - For every index word, add a list of document-id's with their corresponding term frequencies
    - While adding new documents, update the index with new words or with new doc-id for existing words
    
- **Querying** 
    - Create word:tf-idf pairs for the query. 
        tf = 1 + log(word_frequency)
        idf = log(total documents/number of matching documents)
    - Look for matching words in the index and the corresponding documents
    - Sum over all the matching words in the document, the product query_tfidf*document_tf_idf 

- **Scoring**
    Two types of document scoring are calculated.
    - Classical Lucene scoring

        *tf := sqrt(tf)*
        *idf := 1 + log(N/(df+1))*
    - BM25 scoring (Best Matching 25 - Okapi Weighting Scheme)
    
        *tf := (k+1)*tf/(k(1- b+ b*L)+ tf)*

        *idf := log(1+ (N-df+0.5)/(df+0.5))*

        *k= 1.2* --> tunes the impact of tf on scoring

        *b= 0.75* --> tunes the impact of document length(L) on scoring

- **Boosting**
    - Boosting selected terms if present in the query
    - Approach 1: create a dictionary of terms {'term':boost_value,...}
    - Approach 2: POSTag the terms in doc while indexing & add the proper nouns
     & nouns to a dictionary with a corresponding predefined boost value 
    - Once the query is tokenized, apply default boost=1.0 for all terms
    - If the token is a key in the dictionary, change the boost as in dict