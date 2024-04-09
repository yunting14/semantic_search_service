from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

from preprocessor.index import preprocess_documents_parallel

app = FastAPI()

tfidf_vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
fv_docs = pickle.load(open('feature_vector_docs.pkl', 'rb'))

def get_feature_vector(query):
    query_clean = preprocess_documents_parallel([query])
    return tfidf_vectorizer.transform(query_clean).toarray()

def get_cosine_similarity_df(fv_y, fv_x):
    cosine_sim = cosine_similarity(fv_y, fv_x)

    cs = pd.DataFrame(
        data = cosine_sim,
        index = ['text1', 'text2', 'text3'],
        columns= ['Cosine Similarity']
    )
    print(cs)
    return cs

def get_ranked_list(cos_sim_df):
    cs_sorted = cos_sim_df.sort_values(by='Cosine Similarity', axis=0, ascending=False)
    return cs_sorted.index.tolist()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ROUTES
@app.get("/")
def hello_world():
    return "<p>Python API is running!<p>"

@app.get('/feature_names')
def getFeatureNames():
    feature_names = tfidf_vectorizer.get_feature_names_out().tolist()
    return jsonable_encoder(feature_names)

@app.post('/search/') # /search?query=Artificial%20Intelligence
def search(query:str):
    fv_query = get_feature_vector(query)
    cos_sim = get_cosine_similarity_df(fv_docs, fv_query)
    ranked_docs = get_ranked_list(cos_sim)
    return jsonable_encoder(ranked_docs)

if __name__ == '__main__':
    app.run(debug=True, port=8089)