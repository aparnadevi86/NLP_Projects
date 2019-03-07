# load packages
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

# read data
file = "" # specify file path & name
log = pd.read_excel(file)
print(log.info())

data = log.loc[(log['Type'] == 'Operations') & (log['Description'].notnull()), 'Description']

# Preprocessing (clean, tokenize, stem)

## extracting repeated patterns & replacing with single term
def clean_words(text, dict):
    for key,value in dict.items():
        text = re.sub(key, value, text)
    return text
regex_dict = {'1[a-z][0-3][a-z0-9]+':'cal_code',
              '[0-9]+\s?h[ours\s,:.-]*|[0-9]+\s?[ap]m':'xtime ',
              '[0-9]+\s?d[egrc]*|[0-9]+\s?c\s':'xdegc ',
              '[0-9]+\s?kpa':'xkpa',
              '[0-9]+\s?mpa':'xmpa', 
              '[0-9]+\s?mm':'xmm', 
              '[0-9]+\s?bar':'xbar',
              '[0-9]+\s?kv':'xkv', 
              '[0-9]+\s?mw':'xmw',
              '9[a-z][a-z0-9]+':'inlet_code',
              '[0-9]+[a-z]*': 'num_code',
              'st[and]*bi':'standby'}

def clean_sentence(sentence):
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(sentence) ## sentence.lower().split()
    words_cleaned = [w.lower() for w in words if w not in stop_words] #and re.match('^[A-Za-z]+', w)] #and w not in string.punctuation]
    stemmer = PorterStemmer()
    vocab_stemmed = [stemmer.stem(word) for word in words_cleaned]
    lem = WordNetLemmatizer()
    vocab_lem = [lem.lemmatize(word) for word in vocab_stemmed]
    sent_cleaned = " ".join(vocab_lem) #stemmed)
    sent_cleaned = clean_words(sent_cleaned, regex_dict)
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

# vectorize the input sentence
def extract_similar(sentence, vectorizer, vector_df):
    sent_vector =  vectorizer.transform(clean_sent).toarray() # binary/count/tfidf
    norm_vector = normalize(sent_vector, norm='l1', axis=1)
    sent_similarity = norm_vector*vector_df.T  # dotproduct/cosine similarity
    idx_top3 = (-sent_similarity).argsort()[0,1:4]
    print('The top most similar logs are:', '\n\n', \
            '1: ', data.iloc[idx_top3[0]], '\n\n', \
            '2: ', data.iloc[idx_top3[1]], '\n\n', \
            '3: ', data.iloc[idx_top3[2]], '\n','~'*50)


print('With binary method', '\n')
extract_similar(clean_sent, binary_vectorizer, binary_df)
print('\n', 'With count(or tf) method', '\n')
extract_similar(clean_sent, count_vectorizer, count_df)
print('\n', 'With tf-idf method', '\n')
extract_similar(clean_sent, tfidf_vectorizer, tfidf_df)