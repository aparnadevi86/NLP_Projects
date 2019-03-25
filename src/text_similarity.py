# load packages
import sys
import pandas as pd
import re
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from nltk.stem import PorterStemmer 
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize
import dictionary # predefined regex replacements

# read data from the file specified
if len(sys.argv) < 2:
    print("Please provide input file path")
    sys.exit()
  
log = pd.read_excel(sys.argv[1])
print(log.info())

# Subset the data for analysis
data = log.loc[(log['Type'] == 'Operations') & (log['Description'].notnull()), 'Description']

# Preprocessing 

## extracting repeated patterns & replacing with single term
def clean_words(text, dict):
    for key,value in dict.items():
        text = re.sub(key, value, text)
    return text

# process the data (clean, tokenize, stem)
def clean_sentence(sentence):
    words = [w.lower() for w in word_tokenize(sentence) if w not in set(stopwords.words("english"))] 
    words = [PorterStemmer().stem(w) for w in words]
    words = [WordNetLemmatizer().lemmatize(w) for w in words]
    sent_cleaned = clean_words(" ".join(words), dictionary.NUMERIC_CODES)
    return sent_cleaned

clean_data = [clean_sentence(sent) for sent in data]

#  Vectorization 

def vectorize(data, Vectorizer, binary= False):
    vectorizer = Vectorizer(min_df=2, binary=binary) 
    vector_df = vectorizer.fit_transform(data)
    normal_df = normalize(vector_df, norm='l1', axis=1)
    return vectorizer, normal_df

## binary vector method
binary_vectorizer, binary_df = vectorize(clean_data, CountVectorizer, binary = True)

## count vector method
count_vectorizer, count_df = vectorize(clean_data, CountVectorizer)

## tf-idf vector method
tfidf_vectorizer, tfidf_df = vectorize(clean_data, TfidfVectorizer)


# Similarity identification

## choose a random sentence
sent = np.random.choice(data)
clean_sent = [clean_sentence(sent)]
print('The input log is:', '\n\n', sent)

# vectorize the input document & extract similar documents
def extract_similar(sentence, vectorizer, vector_df):
    sent_vector =  vectorizer.transform(clean_sent).toarray() # binary/count/tfidf
    sent_vector = normalize(sent_vector, norm='l1', axis=1)
    sent_similarity = sent_vector*vector_df.T  # dotproduct/cosine similarity
    top3 = (-sent_similarity).argsort()[0,1:4]
    print('The top most similar logs are:', '\n\n', \
            '1: ', data.iloc[top3[0]], '\n\n', \
            '2: ', data.iloc[top3[1]], '\n\n', \
            '3: ', data.iloc[top3[2]], '\n','~'*50)


print('With binary method', '\n')
extract_similar(clean_sent, binary_vectorizer, binary_df)
print('\n', 'With count(or tf) method', '\n')
extract_similar(clean_sent, count_vectorizer, count_df)
print('\n', 'With tf-idf method', '\n')
extract_similar(clean_sent, tfidf_vectorizer, tfidf_df)
