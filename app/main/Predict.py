import hgtk
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences
import tensorflow_hub as hub
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle
from transformers import *
import json
from tensorflow.keras import backend as K
warnings.filterwarnings('ignore')

## 모델 불러오기
def recall(y_target, y_pred):
    y_target_yn = K.round(K.clip(y_target, 0, 1))
    y_pred_yn = K.round(K.clip(y_pred, 0, 1))
    count_true_positive = K.sum(y_target_yn * y_pred_yn)
    count_true_positive_false_negative = K.sum(y_target_yn)
    recall = count_true_positive / (count_true_positive_false_negative + K.epsilon())
    return recall


def precision(y_target, y_pred):
    y_pred_yn = K.round(K.clip(y_pred, 0, 1))
    y_target_yn = K.round(K.clip(y_target, 0, 1))
    count_true_positive = K.sum(y_target_yn * y_pred_yn)
    count_true_positive_false_positive = K.sum(y_pred_yn)
    precision = count_true_positive / (count_true_positive_false_positive + K.epsilon())
    return precision


def f1score(y_target, y_pred):
    _rebcall = recall(y_target, y_pred)
    _precision = precision(y_target, y_pred)
    _f1score = ( 2 * _recall * _precision) / (_recall + _precision+ K.epsilon())
    return _f1score


### Char - Level - Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# loading
with open('./app/main/tokenizer.pickle', 'rb') as handle:
    tk = pickle.load(handle)

input_size = 512
vocab_size = len(tk.word_index)
embedding_size = 52

conv_layers = [
               [256, 7, 3],
               [256, 7, 3],
               [256, 3, -1],
               [256, 3, -1],
               [256, 3, -1],
               [256, 3, 3]
               ]

fully_connected_layers = [256, 256]
dropout = 0.3

# Embedding weights
embedding_weights = []
embedding_weights.append(np.zeros(vocab_size))

for char, i in tk.word_index.items():
    onehot = np.zeros(vocab_size)
    onehot[i - 1] = 1
    embedding_weights.append(onehot)

embedding_weights = np.array(embedding_weights)

### Ko-ELECTRA
from transformers import TFElectraModel
from transformers import ElectraTokenizer

tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
model = TFElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator", from_pt=True)

SEQ_LEN = 128
BATCH_SIZE = 16

token_inputs = tf.keras.layers.Input((SEQ_LEN,), dtype=tf.int32, name='input_word_ids')
mask_inputs = tf.keras.layers.Input((SEQ_LEN,), dtype=tf.int32, name='input_masks')
segment_inputs = tf.keras.layers.Input((SEQ_LEN,), dtype=tf.int32, name='input_segment')

ELEC_outputs = model([token_inputs, mask_inputs, segment_inputs])

# print(ELEC_outputs[0])

## 여기서 불러와야함! (아니면 에러 뜸)
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout, LSTM, Bidirectional, MaxPooling1D, Reshape, Flatten, Conv1D, GlobalMaxPooling1D, BatchNormalization, LayerNormalization, Concatenate, AveragePooling1D
from keras.optimizers import Adam, SGD, RMSprop, Nadam, Adamax
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,LearningRateScheduler
import tensorflow_hub as hub
import re
from tqdm import tqdm
from keras.models import Model
from keras.layers import Dense, Lambda, Input
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


## 여기서 불러와야함! (아니면 에러 뜸)
from keras.models import Sequential
from keras.layers import Activation, Dense, Input, Dropout, LSTM, Bidirectional, MaxPooling1D, Reshape, Flatten, Conv1D, GlobalMaxPooling1D, BatchNormalization, LayerNormalization, Concatenate, AveragePooling1D
from keras.optimizers import Adam, SGD, RMSprop, Nadam, Adamax
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Dense, Lambda, Input


def build_model():
    inputs1 = Input(shape=(input_size,), name='input1', dtype='int64')
    inputs2 = ELEC_outputs[0]

    x = Embedding(vocab_size + 1,
                  embedding_size,
                  input_length=input_size,
                  weights=[embedding_weights])(inputs1)
    for filter_num, filter_size, pooling_size in conv_layers:
        x = Conv1D(filter_num, filter_size)(x)
        x = Activation('relu')(x)

        if pooling_size != -1:
            x = MaxPooling1D(pool_size=pooling_size)(x)
    flat1 = Flatten()(x)

    Bi_LSTM = Bidirectional(LSTM(128, dropout=0.2, return_sequences=True))(inputs2)
    maxpooling1 = MaxPooling1D(2, None, padding='valid')(Bi_LSTM)
    conv = Conv1D(filters=256, kernel_size=3, activation='relu')(maxpooling1)
    drop_out = Dropout(0.4)(conv)
    maxpooling2 = MaxPooling1D(2, None, padding='valid')(drop_out)
    flat2 = Flatten()(maxpooling2)

    x = Concatenate()([flat1, flat2])
    for dense_size in fully_connected_layers:
        x = Dense(dense_size, activation='relu')(x)
        x = Dropout(dropout)(x)

    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[inputs1, token_inputs, mask_inputs, segment_inputs], outputs=predictions)
    model.compile(Adam(lr=4e-6), loss='binary_crossentropy', metrics=['accuracy', f1score])
    return model

model = build_model()
# model.summary()
model.load_weights('./app/main/KoELEC+CharCNN_genre1.h5')

print('모델 불러오기 완료')
#################################################################################################
### TEST DATA 바꾸기

def convert_data(text):
    global tokenizer

    SEQ_LEN = 128  # SEQ_LEN : 버트에 들어갈 인풋의 길이
    tokens, masks, segments = [], [], []

    # token : 문장을 토큰화함
    token = tokenizer.encode(text, max_length=SEQ_LEN, pad_to_max_length=True)

    # 마스크는 토큰화한 문장에서 패딩이 아닌 부분은 1, 패딩인 부분은 0으로 통일
    num_zeros = token.count(0)
    mask = [1] * (SEQ_LEN - num_zeros) + [0] * num_zeros

    # 문장의 전후관계를 구분해주는 세그먼트는 문장이 1개밖에 없으므로 모두 0
    segment = [0] * SEQ_LEN

    tokens.append(token)
    masks.append(mask)
    segments.append(segment)

    # tokens, masks, segments, 정답 변수 targets를 numpy array로 지정
    tokens = np.array(tokens)
    masks = np.array(masks)
    segments = np.array(segments)

    return [tokens, masks, segments]


def get_test_data(genre, text):
    # ELECTRA Input
    text = genre + ' ' + text # 장르 추가
    text = re.sub("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "", text)
    text_x = convert_data(text)

    # Char_Embedding Input
    text_char = hgtk.text.decompose(text)
    text_char = re.sub('ᴥ', "", text_char)
    test_text = sum(tk.texts_to_sequences(text_char), [])
    test_text = test_text + [0] * (512 - len(test_text))
    test_text = np.array(test_text, dtype='float32')
    test_text = np.array([test_text])

    return [test_text, text_x]

def isitspo(genre, text):
    test = get_test_data(genre, text)
    pred = model.predict(test)

    return pred