import string
import asyncio
import multiprocessing.process as Process
import time
import concurrent.futures as cf
import math

import nltk
# nltk.download('stopwords')
# nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

lemmatizer = WordNetLemmatizer() # default noun, "v" for verbs, "a" for adjectives, "r" for adverbs, "s" for satellite adjectives
stop_words = stopwords.words('english')

documents = [
    "Document 1",
    "Document 2",
    "Document 3"
]

cleaned_docs = []
punctuation = str.maketrans('', '', string.punctuation)
tasks = []

'''
USING ASYNCIO
    - tasks processed on a single thread, so if there's nothing to await for there is no point using this
    - more for i/o-bound processes, like writing/reading files, db operations, network operations
'''
async def preprocess_single_document_async(doc:str):
    print('Starting:', doc)
    doc_punc_removed = doc.translate(punctuation)
    words = doc_punc_removed.lower().split()
    words = [lemmatizer.lemmatize(word, 'v') for word in words
             if word not in stop_words]
    cleaned_docs.append(' '.join(words))
    print('Processed:', doc)

async def preprocess_documents_async(documents:list[str]):
    for doc in documents:
        task = asyncio.create_task(preprocess_single_document_async(doc))
        tasks.append(task)

    await asyncio.gather(*tasks)

# asyncio.run(preprocess_documents_async(documents))

'''
USING MULTIPROCESSING
    - tasks processed on separate core/thread
    - more for cpu-bound processes, like large calculations, image processing
'''
def preprocess_single_document_parallel(doc:str):
    start = time.time()
    doc_punc_removed = doc.translate(punctuation)
    words = doc_punc_removed.lower().split()
    words = [lemmatizer.lemmatize(word, 'v') for word in words
             if word not in stop_words]
    cleaned_docs.append(' '.join(words))
    end = time.time()

def preprocess_documents_parallel(docs:list[str]):
    start = time.time()
    with cf.ProcessPoolExecutor() as executor:
        executor.map(preprocess_single_document_parallel, docs) # returns a list of result. no returns, can use submit() too
    end = time.time()
    print(f'Parallel processing ended. It took {end-start} seconds')

# preprocess_documents_parallel(documents)


''' 
Multiprocessing is not worth it for simple processes. 
Overheads like inter-process communication. 

* Small calculations
[1000]. Took 4.38690185546875e-05 seconds.
[2000]. Took 0.00013327598571777344 seconds.
[3000]. Took 0.0003249645233154297 seconds.
[4000]. Took 0.0005040168762207031 seconds.
[5000]. Took 0.000782012939453125 seconds.
Parallel processing ended. It took 0.14430809020996094 seconds

[1000]. Took 6.29425048828125e-05 seconds.
[2000]. Took 0.0001430511474609375 seconds.
[3000]. Took 0.00034689903259277344 seconds.
[4000]. Took 0.00046896934509277344 seconds.
[5000]. Took 0.0007796287536621094 seconds.
Synchronous processing ended. It took 0.001837015151977539 seconds

* Large calculations
[100000]. Took 0.14834904670715332 seconds.
[200000]. Took 0.5042500495910645 seconds.
[300000]. Took 0.9536728858947754 seconds.
[400000]. Took 1.7633140087127686 seconds.
[500000]. Took 2.354001998901367 seconds.
Parallel processing ended. It took 2.5013160705566406 seconds

[100000]. Took 0.14200592041015625 seconds.
[200000]. Took 0.4870262145996094 seconds.
[300000]. Took 0.9026510715484619 seconds.
[400000]. Took 1.6830389499664307 seconds.
[500000]. Took 2.280336856842041 seconds.
Synchronous processing ended. It took 5.495201826095581 seconds
'''
nums = [100000, 200000, 300000, 400000, 500000]

def factorial(num):
    start = time.time()
    math.factorial(num)
    end = time.time()
    print(f'[{num}]. Took {end-start} seconds.')

def calculate_factorials(nums):
    start = time.time()
    with cf.ProcessPoolExecutor() as executor:
        executor.map(factorial, nums)
    end = time.time()
    print(f'Parallel processing ended. It took {end-start} seconds\n\n')

def calculate_factorials_sync(nums):
    start = time.time()
    for num in nums:
        factorial(num)
    end = time.time()
    print(f'Synchronous processing ended. It took {end-start} seconds\n\n')

# if __name__ ==  '__main__':
    # calculate_factorials(nums)
    # calculate_factorials_sync(nums)