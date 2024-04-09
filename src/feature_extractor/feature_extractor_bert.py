# https://discuss.huggingface.co/t/how-to-get-embedding-matrix-of-bert-in-hugging-face/10261/2

text1 = 'Artificial intelligence is the simulation of human intelligence processes by machines, especially computer systems. Specific applications of AI include expert systems, natural language processing, speech recognition and machine vision.'
text2 = 'I love flowers. Recently I have a new-found appreciation for artificial flowers, because good-quality ones can look quite natural. I think it\'s an intelligent decision to use them for decorations. I want to ask other people what they think about artificial flowers.'
text3 = 'Jimmy is an intelligent person. Out of everyone I know, he probably has the highest intelligence. I wish I can be as intelligent as him. Someone, make me an artificial brain that has Jimmmy\'s brain juices!'
query = ['Artificial intelligence']
data_docs = [text1, text2, text3, query]

import torch
from transformers import BertTokenizer, BertModel

bertTokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
encoded_input = bertTokenizer(text1, return_tensors='pt')
output = model(**encoded_input)

# print(encoded_input)
'''
{   'input_ids': tensor([[  101,  7976,  4454,  2003,  1996, 12504,  1997,  2529,  4454,  6194,
          2011,  6681,  1010,  2926,  3274,  3001,  1012,  3563,  5097,  1997,
          9932,  2421,  6739,  3001,  1010,  3019,  2653,  6364,  1010,  4613,
          5038,  1998,  3698,  4432,  1012,   102]]), 
    'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), 
    'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
}
'''

# print(output)
'''
BaseModelOutputWithPoolingAndCrossAttentions(
     last_hidden_state=tensor([[[-0.0623, -0.3527, -0.5576,  ..., -0.0101, -0.1844,  0.5327],
         [ 0.2346,  0.5921, -0.4877,  ...,  0.1900,  0.6903,  0.5983],
         [-0.1684,  0.7258, -0.6291,  ..., -0.9630, -0.1117,  0.5228],
         ...,
         [-0.1381,  0.1396, -0.5438,  ..., -0.8875, -0.3788,  0.3332],
         [ 0.7441,  0.0501, -0.3979,  ..., -0.0028, -0.5767, -0.1452],
         [ 0.2867,  0.0729, -0.1420,  ...,  0.0998, -0.6188, -0.3201]]],
          grad_fn=<NativeLayerNormBackward0>), 
     pooler_output=tensor([[-0.9234, -0.4856, -0.9194,  0.7224,  0.8136, -0.5384,  0.5352,  0.3555,
         -0.8439, -1.0000, -0.7559,  0.9771,  0.9748,  0.1335,  0.7299, -0.7069,
         -0.5949, -0.5332,  0.4685,  0.5131,  0.5890,  1.0000, -0.4003,  0.5013,
          0.4261,  0.9920, -0.8339,  0.8839,  0.9279,  0.7992, -0.5159,  0.4236,
         ...
         -0.8848, -0.3527,  0.8461,  0.8850, -0.5900, -0.7525,  0.5050, -0.6872,
          0.9919,  0.6547, -0.9356,  0.1381,  0.6217, -0.9188, -0.7113,  0.7645]],
       grad_fn=<TanhBackward0>), hidden_states=None, past_key_values=None, attentions=None, cross_attentions=None)


print(len(output)) ==> 2

1. last_hidden_state contains the hidden representations for each token in each sequence of the batch. 
So the size is (batch_size, seq_len, hidden_size)       

2. pooler_output contains a "representation" of each sequence in the batch, and is of size (batch_size, hidden_size). 
What it basically does is take the hidden representation of the [CLS] token of each sequence in the batch 
(which is a vector of size hidden_size), and then run that through the BertPooler nn.Module. 
This consists of a linear layer followed by a Tanh activation function. 
The weights of this linear layer are already pretrained on the next sentence prediction task 
(note that BERT is pretrained on 2 tasks: masked language modeling and next sentence prediction). 
I assume that the authors of the Transformers library have taken the weights from the original TF implementation, and initialized the layer with them. 
In theory, they would come from BertForPretraining - which is BERT with the 2 pretraining heads on top. 
'''

print(output.last_hidden_state.size()) # torch.Size([1, 36, 768]) ==> (batch_size, seq_len(no. of tokens), hidden_size)

print('First token:', output.last_hidden_state[:,0,:].mean(1)) 

# print('Second token:', output.last_hidden_state[:,1,:]) # torch.Size([1, 768])

sentence_embedding = torch.mean(output.last_hidden_state.squeeze(dim=0), dim=0)
print('Sentence Embedding Size:', sentence_embedding.size()) # torch.Size([768])
'''
First token: tensor([-0.0127], grad_fn=<MeanBackward1>)
torch.Size([1])

Second token: tensor([[ 2.3455e-01,  5.9205e-01, -4.8770e-01,  1.7207e-02,  3.4896e-01,
          2.7281e-01,  2.9207e-02,  8.0254e-02,  5.3580e-01, -5.0182e-01,
         -2.4469e-01,  2.7015e-01, -2.5979e-01,  8.3177e-02, -3.4946e-01,
         -3.3609e-01,  6.7124e-02,  1.6864e-02, -8.6530e-02,  2.1881e-01,
         -3.7153e-01, -1.9262e-02, -1.0820e-01,  7.0706e-01, -8.3148e-02,
         ...,
         2.5434e-01,  3.5197e-01,  3.0604e-01, -2.9270e-01, -2.6135e-01,
          1.8996e-01,  6.9034e-01,  5.9832e-01]], grad_fn=<SliceBackward0>)
torch.Size([1, 768])
'''

last_hidden_states = output.last_hidden_state
token_embeddings = torch.squeeze(last_hidden_states, dim=0) # torch.Size([36, 768])

embedding_matrix = model.embeddings.word_embeddings.weight
'''
##### text1
Word Embedding Matrix: Parameter containing:
tensor([[-0.0102, -0.0615, -0.0265,  ..., -0.0199, -0.0372, -0.0098],
        [-0.0117, -0.0600, -0.0323,  ..., -0.0168, -0.0401, -0.0107],
        [-0.0198, -0.0627, -0.0326,  ..., -0.0165, -0.0420, -0.0032],
        ...,
        [-0.0218, -0.0556, -0.0135,  ..., -0.0043, -0.0151, -0.0249],
        [-0.0462, -0.0565, -0.0019,  ...,  0.0157, -0.0139, -0.0095],
        [ 0.0015, -0.0821, -0.0160,  ..., -0.0081, -0.0475,  0.0753]],
       requires_grad=True)
'''
# averaging all tokens to get sentence embedding, forming document embedding of ([768])
def get_feature_vector(doc_list):
     feature_vector_list = []
     for doc in doc_list:
          encoded_input = bertTokenizer(doc, return_tensors='pt')
          output = model(**encoded_input)
          last_hidden_states = output.last_hidden_state # torch.size([1, no. of tokens, 768])
          token_embeddings = torch.squeeze(last_hidden_states, dim=0) # torch.size([no. of tokens, 768])
          sentence_embedding = torch.mean(token_embeddings, dim=0) # torch.size([768])
          feature_vector_list.append(sentence_embedding)
     return feature_vector_list

docs_feature_vector = get_feature_vector([text1, text2, text3]) # this is an array of tensors (multi-dim matrix, each of length 768)
query_feature_vector = get_feature_vector([query])

from scipy.spatial.distance import cosine

def compare_cosine_similarity(fv_docs, fv_query):
     cos_sim_dict = {}
     for i in range(len(fv_docs)):
          cos_sim = 1 - cosine(fv_docs[i].detach().numpy(), fv_query[0].detach().numpy())
               # RuntimeError: Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.
               # ValueError: Input vector should be 1-D.
          cos_sim_dict['text'+ str(i+1)] = cos_sim
     return cos_sim_dict

cs = compare_cosine_similarity(docs_feature_vector, query_feature_vector)
print(cs)
'''
{'text1': 0.5494999289512634, 'text2': 0.3853999376296997, 'text3': 0.40591856837272644}
text1 > text3 > text2
by human judgment, this is accurate 
'''