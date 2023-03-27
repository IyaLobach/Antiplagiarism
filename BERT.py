import numpy as np
import pandas as pd
import torch
import transformers as ppb # pytorch transformers
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
import time




start = time.time()

# из примера
#df = pd.read_csv('https://github.com/clairett/pytorch-sentiment-classification/raw/master/data/SST2/train.tsv', delimiter='\t', header=None)
df = pd.read_csv('C:/Users/iyush/PycharmProjects/Word2Vec/Text.csv', header=None)
print(df)

## Хотите BERT вместо distilBERT? Раскомментируйте следующую строку:
#model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')
# model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
#
# # Загрузка предобученной модели/токенизатора
# tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
# model = model_class.from_pretrained(pretrained_weights)



#для русского
tokenizer = AutoTokenizer.from_pretrained("Geotrend/distilbert-base-ru-cased")
model = AutoModel.from_pretrained("Geotrend/distilbert-base-ru-cased")


tokenized = df[0].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))


max_len = 0
for i in tokenized.values:
    if len(i) > max_len:
        max_len = len(i)

padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])

input_ids = torch.tensor(np.array(padded))

with torch.no_grad():
    last_hidden_states = model(input_ids)

# Разрежьте выход для первой позиции во всех последовательностях, возьмите все выходы скрытых нейронок
features = last_hidden_states[0][:,0,:].numpy()
end = time.time()

print(len(features))
print(len(features[0]))
print(features)
print((end - start)/60)

