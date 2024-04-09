import nltk
import pandas as pd
import string
import pickle
import os

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words('english')

# make lower case, remove all punctuation, and lemmatization for verbs
def preprocessing(documents):
    clean = []
    punc = str.maketrans('', '', string.punctuation)

    for doc in documents:
        doc_no_punc = doc.translate(punc) 
        words = doc_no_punc.lower().split()
        words = [lemmatizer.lemmatize(word, 'v') for word in words
                if word not in stop_words]
        clean.append(' '.join(words))
    
    return clean

docs_clean = []
directory = 'data_txt'
for filename in os.listdir(directory):
    file = os.path.join(directory, filename)
    if os.path.isfile(file) == False:
        continue

    with open(file, 'r', errors='ignore') as data_file:
        data = data_file.read().replace('\n', ' ')
    
    print(type(data))
    cleaned = preprocessing([data])
    docs_clean.append(cleaned)

tfidf = TfidfVectorizer(ngram_range=(1,2))
    # parameter "token_pattern" -> default token separators are punctuation and space
tfidf.fit(docs_clean)

pickle.dump(tfidf, open('ar_vectorizer.pkl', 'wb'))

fv_docs = tfidf.transform(docs_clean).toarray()
# fv_query = tfidf.transform(query_clean).toarray()

pickle.dump(fv_docs, open('fv_ar.pkl', 'wb'))

print('Number of feature names: ', tfidf.get_feature_names_out().size())

# cosine_sim = cosine_similarity(fv_query, fv_docs)

# cs = pd.DataFrame(
#     data = cosine_sim,
#     index = ['Cosine Similarity'],
#     columns=['text1', 'text2', 'text3']
# )

# cosine_sim = cosine_similarity(fv_query, fv_docs)

# cs = pd.DataFrame(
#     data = cosine_sim,
#     index = ['Cosine Similarity'],
#     columns=['text1', 'text2', 'text3']
# )

# print(cs)
'''
WITH MWE
                      text1  text2  text3
Cosine Similarity  0.191894    0.0    0.0
    note: text1 is the only text that has "artificial intelligence" 

WITHOUT MWE
                       text1     text2     text3
Cosine Similarity  0.294229  0.133276  0.225461

WITH UNIGRAMS AND BIGRAMS
                      text1     text2     text3
Cosine Similarity  0.254637  0.068673  0.114143
'''

# print(cs.rank(axis=1, method='max', ascending=False))
'''
                   text1  text2  text3
Cosine Similarity    1.0    3.0    2.0
'''

# cs_sorted = cs.sort_values(by='Cosine Similarity', axis=1, ascending=False)
'''
                      text1     text3     text2
Cosine Similarity  0.254637  0.114143  0.068673
'''
# print(cs_sorted.index.tolist())