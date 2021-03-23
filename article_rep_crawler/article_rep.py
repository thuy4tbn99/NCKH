'''
# requirements:
pip install vncorenlp
pip install keras 
pip install pickle

'''
import os
import numpy as np
import pickle
import torch 

# os.environ["CUDA_VISIBLE_DEVICES"]="0"

import keras
from keras.layers import *
from keras.models import Model
from keras import backend as K
from keras.optimizers import *
from sklearn.metrics import roc_auc_score
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
import pickle 

# load word_dict 
file = open('phobert_news_preprocess.pkl', 'rb')
word_dict, news_words, news_index, news = pickle.load(file)
file.close()

# load modelExtractor
modelExtractor = keras.models.load_model('new_embedding_extractor')

# load vncorenlp
from vncorenlp import VnCoreNLP
VnCoreNLP_jar_file = './vncorenlp/VnCoreNLP-1.1.1.jar'
rdrsegmenter = VnCoreNLP(VnCoreNLP_jar_file, annotators='wseg')

def news_word2index(word_dict, sapo):
    news_tokenizer = rdrsegmenter.tokenize(sapo)[0]
    
    word_id = []
    for word in news_tokenizer: # quét các tokens
        if word in word_dict:
            word_id.append(word_dict[word][0])
    word_id = word_id[:30] # lấy word_id của article (embedd)
    news_words = (word_id + [0]*(30-len(word_id))) # max 30 tokens, <30 cho =0
    
    return news_words

def article_represent(articleID, sapo, modelExtractor):
    article_sapo = sapo
    print(article_sapo)
    sapo_index = news_word2index(word_dict, article_sapo)

    article = np.array([sapo_index], dtype='int32')
    print(article.shape)

    represent = modelExtractor.predict(article)
    
    return represent

sapo_test = 'Trung vệ Leonardo Bonucci tiết lộ Cristiano Ronaldo rất phấn khích trước lượt về vòng 1/8 Champions League với Porto.'
reprentation = article_represent(1, sapo_test, modelExtractor)
print('Article representation:\n', reprentation)