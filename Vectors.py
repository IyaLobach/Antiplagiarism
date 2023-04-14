import numpy as np
import pandas as pd
from opt_einsum.backends import torch

from transformers import AutoTokenizer, AutoModel


text = pd.read_csv('C:/Users/iyush/PycharmProjects/BERT/dataset/Text.csv', header=None, sep=";")
text[3] = None
print(text[2])

tokenizer = AutoTokenizer.from_pretrained("Geotrend/distilbert-base-ru-cased")
model = AutoModel.from_pretrained("Geotrend/distilbert-base-ru-cased")
tokenized = text[2].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))


# # Accessing the model configuration
# configuration = model.config
# print(model.config)

max_len = 0
for i in tokenized.values:
    if len(i) > max_len:
        max_len = len(i)
print(max_len)

if max_len < 512:
    padded = np.array([i + [0]*(512-len(i)) for i in tokenized.values])
else:
    print("max_len > 512")

input_ids = torch.tensor(np.array(padded))
with torch.no_grad():
    last_hidden_states = model(input_ids)

features = last_hidden_states[0][:,0,:].numpy()

vectors = []
for i in features:
    tmp = []
    for j in i:
        tmp.append(j)
    vectors.append(tmp)
text[3] = vectors
print(text)
text.to_csv(r'C:/Users/iyush/PycharmProjects/BERT/dataset/Text_vector.csv', index=False, header=None)

# print("Количество строк", len(features))
# print("Размер вектора", len(features[0]))

