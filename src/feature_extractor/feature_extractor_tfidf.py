import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

from torch import cosine_similarity

from preprocessor.index import preprocess_documents_parallel
from data.test import docs, query

docs_clean = preprocess_documents_parallel(docs)
'''
WITH MWE
docs_clean:
[text1] artificial_intelligence simulation human intelligence process machine especially computer systems specific applications ai include expert systems natural language process speech recognition machine vision
[text2] love flower recently newfound appreciation artificial flower goodquality ones look quite natural think intelligent decision use decorations want ask people think artificial flower
[text3] jimmy intelligent person everyone know probably highest intelligence wish intelligent someone make artificial brain jimmmys brain juices
'''
query_clean = preprocess_documents_parallel(query)

tfidf = TfidfVectorizer(ngram_range=(1,2))
    # parameter "token_pattern" -> default token separators are punctuation and space
tfidf.fit(docs_clean)

pickle.dump(tfidf, open('vectorizer.pkl', 'wb')) # w - write; b - binary

# print('TF-IDF feature names: ', tfidf.get_feature_names_out())
'''
WITH MWE - "artificial_intelligence"
TF-IDF feature names:  
['ai' 'applications' 'appreciation' 'artificial' 'artificial_intelligence'
 'ask' 'brain' 'computer' 'decision' 'decorations' 'especially' 'everyone'
 'expert' 'flower' 'goodquality' 'highest' 'human' 'include'
 'intelligence' 'intelligent' 'jimmmys' 'jimmy' 'juices' 'know' 'language'
 'look' 'love' 'machine' 'make' 'natural' 'newfound' 'ones' 'people'
 'person' 'probably' 'process' 'quite' 'recently' 'recognition'
 'simulation' 'someone' 'specific' 'speech' 'systems' 'think' 'use'
 'vision' 'want' 'wish']


WITHOUT MWE - "artificial", "intelligence"
TF-IDF feature names:  
['ai' 'applications' 'appreciation' 'artificial' 'ask' 'brain' 'computer'
 'decision' 'decorations' 'especially' 'everyone' 'expert' 'flower'
 'goodquality' 'highest' 'human' 'include' 'intelligence' 'intelligent'
 'jimmmys' 'jimmy' 'juices' 'know' 'language' 'look' 'love' 'machine'
 'make' 'natural' 'newfound' 'ones' 'people' 'person' 'probably' 'process'
 'quite' 'recently' 'recognition' 'simulation' 'someone' 'specific'
 'speech' 'systems' 'think' 'use' 'vision' 'want' 'wish']

WITHOUT MWE, WITH UNIGRAMS AND BIGRAMS
TF-IDF feature names:  ['ai' 'ai include' 'applications' 'applications ai' 'appreciation'
 'appreciation artificial' 'artificial' 'artificial brain'
 'artificial flower' 'artificial intelligence' 'ask' 'ask people' 'brain'
 'brain jimmmys' 'brain juices' 'computer' 'computer systems' 'decision'
 'decision use' 'decorations' 'decorations want' 'especially'
 'especially computer' 'everyone' 'everyone know' 'expert'
 'expert systems' 'flower' 'flower goodquality' 'flower recently'
 'goodquality' 'goodquality ones' 'highest' 'highest intelligence' 'human'
 'human intelligence' 'include' 'include expert' 'intelligence'
 'intelligence process' 'intelligence simulation' 'intelligence wish'
 'intelligent' 'intelligent decision' 'intelligent person'
 'intelligent someone' 'jimmmys' 'jimmmys brain' 'jimmy'
 'jimmy intelligent' 'juices' 'know' 'know probably' 'language'
 'language process' 'look' 'look quite' 'love' 'love flower' 'machine'
 'machine especially' 'machine vision' 'make' 'make artificial' 'natural'
 'natural language' 'natural think' 'newfound' 'newfound appreciation'
 'ones' 'ones look' 'people' 'people think' 'person' 'person everyone'
 'probably' 'probably highest' 'process' 'process machine'
 'process speech' 'quite' 'quite natural' 'recently' 'recently newfound'
 'recognition' 'recognition machine' 'simulation' 'simulation human'
 'someone' 'someone make' 'specific' 'specific applications' 'speech'
 'speech recognition' 'systems' 'systems natural' 'systems specific'
 'think' 'think artificial' 'think intelligent' 'use' 'use decorations'
 'vision' 'want' 'want ask' 'wish' 'wish intelligent']
'''

fv_docs = tfidf.transform(docs_clean).toarray()
fv_query = tfidf.transform(query_clean).toarray()

pickle.dump(fv_docs, open('feature_vector_docs.pkl', 'wb'))

tfidf_df = pd.DataFrame(
    data = fv_docs,
    index = ['text1', 'text2', 'text3'],
    columns= tfidf.get_feature_names_out()
)

tfidf_df_query = pd.DataFrame(
    data=fv_query,
    index=['query'],
    columns=tfidf.get_feature_names_out()
)

# tfidf_df.to_csv('fv_docs_18May-1530.csv')
# tfidf_df_query.to_csv('fv_query_ngrams_19May.csv')

# cosine_sim = cosine_similarity(fv_query, fv_docs)

# cs = pd.DataFrame(
#     data = cosine_sim,
#     index = ['Cosine Similarity'],
#     columns=['text1', 'text2', 'text3']
# )

cosine_sim = cosine_similarity(fv_query, fv_docs)

cs = pd.DataFrame(
    data = cosine_sim,
    index = ['Cosine Similarity'],
    columns=['text1', 'text2', 'text3']
)

print(cs)
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

print(cs.rank(axis=1, method='max', ascending=False))
'''
                   text1  text2  text3
Cosine Similarity    1.0    3.0    2.0
'''

cs_sorted = cs.sort_values(by='Cosine Similarity', axis=1, ascending=False)
'''
                      text1     text3     text2
Cosine Similarity  0.254637  0.114143  0.068673
'''
print(cs_sorted.index.tolist())