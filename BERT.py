import numpy as np
import pandas as pd
import sklearn
import torch
import transformers as ppb # pytorch transformers
from numpy import dot
from numpy.linalg import norm
from scipy import spatial
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
import time




start = time.time()

text = pd.read_csv('C:/Users/iyush/PycharmProjects/BERT/dataset/Text.csv', header=None, sep=";")
# text[3] = None
print(text[2])

# pair = pd.read_csv('C:/Users/iyush/PycharmProjects/BERT/dataset/plagiarism.csv', header=None, sep=";")
# print(pair)

text_vector = pd.read_csv('C:/Users/iyush/PycharmProjects/BERT/dataset/Text_vector.csv', header=None, sep=",")
print(text_vector[3])
# # для русского
# tokenizer = AutoTokenizer.from_pretrained("Geotrend/distilbert-base-ru-cased")
# model = AutoModel.from_pretrained("Geotrend/distilbert-base-ru-cased")
#
# # токенизация
# tokenized = text[2].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
#
#
# max_len = 0
# for i in tokenized.values:
#     if len(i) > max_len:
#         max_len = len(i)
#
#
# print(max_len)
#
# if max_len < 512:
#     padded = np.array([i + [0]*(512-len(i)) for i in tokenized.values])
# else:
#     print("max_len > 512")
#
# input_ids = torch.tensor(np.array(padded))
# with torch.no_grad():
#     last_hidden_states = model(input_ids)
#
#
#
# # Разрежьте выход для первой позиции во всех последовательностях, возьмите все выходы скрытых нейронок
# features = last_hidden_states[0][:,0,:].numpy()
#
# vectors = []
# for i in features:
#     tmp = []
#     for j in i:
#         tmp.append(j)
#     vectors.append(tmp)
# text[3] = vectors
# print(text)
# text.to_csv(r'C:/Users/iyush/PycharmProjects/BERT/dataset/Text_vector.csv', index=False, header=None)
#
#
# print("Количество строк", len(features))
# print("Размер вектора", len(features[0]))


end = time.time()
print("Время ", (end - start)/60)

