# https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/
# https://towardsdatascience.com/nlp-extract-contextualized-word-embeddings-from-bert-keras-tf-67ef29f60a7b

import torch
from transformers import BertTokenizer, BertModel

from data.test import docs, query, text1

# Tokenizer from pre-trained model 
bertTokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def mark_text(text_string):
    marked_text = ''
    sentences = text_string.strip().split('.')
    for i in range(len(sentences)):
        # first sentence needs [CLS] token
        if sentences[i] == '' or sentences[i].isspace():
            continue
        if i == 0:
            marked_text += '[CLS] ' + sentences[i] + ' [SEP] '
        else:
            marked_text += sentences[i] + ' [SEP] '
    return marked_text[:-1] 

# Step 1: mark sentences with [CLS] and [SEP] as bert needs them 
marked_text = mark_text(text1)

# Step 2: Split sentences into tokens 
tokenized_text = bertTokenizer.tokenize(marked_text)
'''
['[CLS]', 'artificial', 'intelligence', 'is', 'the', 'simulation', 'of', 'human', 'intelligence',
 'processes', 'by', 'machines', ',', 'especially', 'computer', 'systems', '[SEP]', 'specific', 
 'applications', 'of', 'ai', 'include', 'expert', 'systems', ',', 'natural', 'language', 
 'processing', ',', 'speech', 'recognition', 'and', 'machine', 'vision', '[SEP]']
'''

# Step 3: Map token strings to vocabulary indeces
indexed_tokens = bertTokenizer.convert_tokens_to_ids(tokenized_text)
'''
[101, 7976, 4454, 2003, 1996, 12504, 1997, 2529, 4454, 6194, 2011, 6681, 1010, 2926, 
3274, 3001, 102, 3563, 5097, 1997, 9932, 2421, 6739, 3001, 1010, 3019, 2653, 6364, 1010, 
4613, 5038, 1998, 3698, 4432, 102]
'''

# Step 4: Include Segment IDs for the senteces
segment_ids = [1] * len(tokenized_text)

# Step 5: extract embeddings
tokens_tensor = torch.tensor([indexed_tokens])
segment_tensor = torch.tensor([segment_ids])

bertModel = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
bertModel.eval()

with torch.no_grad():
    outputs = bertModel(tokens_tensor, segment_tensor)
    hidden_states = outputs[2] # hidden states is the 3rd item of the output in BERT model. See "BERT specific output" in https://huggingface.co/docs/transformers/model_doc/bert#bertmodel

# understand the embeddings extracted
print('Number of layers in BERT: ', len(hidden_states))
print('Number of batches: ', len(hidden_states[0]))
print('Number of tokens: ', len(hidden_states[0][0]))
print('Number of hidden units: ', len(hidden_states[0][0][0]))
'''
The output is a 4-dimensional tuple 
feature_vector = hidden_states[layer_i][batch_i][token_i]

layer_i : 0-12 (1 input embedding layer + 12 bert layers)
batch_i : 0
token_i: 0-34

Number of layers in BERT:  13   # layers
Number of batches:  1           # batches
Number of tokens:  35           # tokens
Number of hidden units:  768    # features (standard in base-bert; 768-axes vector space)

Currently, the values are grouped by layer. We want it to be grouped by tokens. 
'''

# Step 6: Grouping hidden values by tokens

# create new dimension in the tensor
token_embeddings = torch.stack(hidden_states, dim=0) 
print(token_embeddings.size()) # torch.Size([13, 1, 35, 768])

# remove "batch" dimension (dimension = 1)
token_embeddings = torch.squeeze(token_embeddings, dim=1)
print(token_embeddings.size()) # torch.Size([13, 35, 768])

# switch the layers and tokens dimensions around such that the values are grouped by tokens
token_embeddings = token_embeddings.permute(1, 0, 2)
print(token_embeddings.size()) # torch.Size([35, 13, 768])

'''
After step 6, we will have successfully grouped the values by tokens. 
35 -> 13 -> 768
Now, for each token, we have 13 separate vectors, each consisting of 768 features (vector axes)
We need to have 1 vector for 1 sentence/input. This means we need to combine the layers.

Which layers or combination of layers provide the best representation?
- To explore; there is no straightforward rule
'''

# Step 7: Combining the layers

# Combination 1: To get one vector for each token, sum all the layers together
token_embeddings_sum = []

for token in token_embeddings: 
    sum_layers_vec = torch.sum(token, dim=0) # sum (12 x 768)
    token_embeddings_sum.append(sum_layers_vec)

print('Shape after summing all layers:', len(token_embeddings_sum), 'x', len(token_embeddings_sum[0])) 
'''
Shape after summing all layers: 35 x 768
'''

# Combination 2: To get one vector for each sentence/document, average the 2nd to last layer (768)

token_vectors = hidden_states[2][0] # ?? hidden_states [13 x 1 x 35 x 768] -> [35, 768]
sentence_embedding_ave = torch.mean(token_vectors, dim=0)

print('Shape after averaging all layers for all tokens:', sentence_embedding_ave.size())
'''
Shape after averaging all layers for all tokens: torch.Size([768])
'''

# Step 8: Compare the similarity using cosine similarity
for i, token_str in enumerate(tokenized_text):
    print(i, token_str)

'''
0 [CLS]
1 artificial
2 intelligence
3 is
4 the
5 simulation
6 of
7 human
8 intelligence
9 processes
10 by
11 machines
12 ,
13 especially
14 computer
15 systems
16 [SEP]
17 specific
18 applications
19 of
20 ai
21 include
22 expert
23 systems
24 ,
25 natural
26 language
27 processing
28 ,
29 speech
30 recognition
31 and
32 machine
33 vision
34 [SEP]
'''