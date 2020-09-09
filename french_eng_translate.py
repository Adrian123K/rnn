import pandas as pd
import urllib3
import zipfile
import shutil
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

http = urllib3.PoolManager()
url = 'http://www.manythings.org/anki/fra-eng.zip'
filename = 'fra-eng.zip'
path = os.getcwd()
zipfilename = os.path.join(path, filename)
with http.request('GET', url, preload_content=False) as r, open(zipfilename, 'wb') as out_file:
    shutil.copyfileobj(r, out_file)

with zipfile.ZipFile(zipfilename, 'r') as zip_ref:
    zip_ref.extractall(path)

lines = pd.read_csv('fra.txt', names=['src', 'tar', 'cc'], sep='\t')
print(len(lines))

#
# lines = lines.loc[:, 'src':'tar']
# lines = lines[0:60000]  # 6만개만 저장
# print(lines.sample(10))
#
# lines.tar = lines.tar.apply(lambda x: '\t ' + x + ' \n')
# print(lines.sample(10))
#
# # 글자 집합 구축
# src_vocab = set()
# for line in lines.src:  # 1줄씩 읽음
#     for char in line:  # 1개의 글자씩 읽음
#         src_vocab.add(char)
#
# tar_vocab = set()
# for line in lines.tar:
#     for char in line:
#         tar_vocab.add(char)
#
# print(src_vocab)
#
# src_vocab_size = len(src_vocab) + 1
# tar_vocab_size = len(tar_vocab) + 1
# print(src_vocab_size)
# print(tar_vocab_size)
#
# src_vocab = sorted(list(src_vocab))
# tar_vocab = sorted(list(tar_vocab))
# print(src_vocab[45:75])
# print(tar_vocab[45:75])
#
# src_to_index = dict([(word, i + 1) for i, word in enumerate(src_vocab)])
# tar_to_index = dict([(word, i + 1) for i, word in enumerate(tar_vocab)])
# print(src_to_index)
# print(tar_to_index)
#
# encoder_input = []
# for line in lines.src:  # 입력 데이터에서 1줄씩 문장을 읽음
#     temp_X = []
#     for w in line:  # 각 줄에서 1개씩 글자를 읽음
#         temp_X.append(src_to_index[w])  # 글자를 해당되는 정수로 변환
#     encoder_input.append(temp_X)
#
# print(encoder_input[:5])
#
# decoder_input = []
# for line in lines.tar:
#     temp_X = []
#     for w in line:
#         temp_X.append(tar_to_index[w])
#     decoder_input.append(temp_X)
# print(decoder_input[:5])
#
# decoder_target = []
# for line in lines.tar:
#     t = 0
#     temp_X = []
#     for w in line:
#         if t > 0:
#             temp_X.append(tar_to_index[w])
#         t = t + 1
#     decoder_target.append(temp_X)
# print(decoder_target[:5])
#
# max_src_len = max([len(line) for line in lines.src])
# max_tar_len = max([len(line) for line in lines.tar])
# print(max_src_len)
# print(max_tar_len)
#
# encoder_input = pad_sequences(encoder_input, maxlen=max_src_len, padding='post')
# decoder_input = pad_sequences(decoder_input, maxlen=max_tar_len, padding='post')
# decoder_target = pad_sequences(decoder_target, maxlen=max_tar_len, padding='post')
#
# print(encoder_input)
#
# encoder_input = to_categorical(encoder_input)
# decoder_input = to_categorical(decoder_input)
# decoder_target = to_categorical(decoder_target)
#
# from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
# from tensorflow.keras.models import Model
# import numpy as np
#
# encoder_inputs = Input(shape=(None, src_vocab_size))
# encoder_lstm = LSTM(units=256, return_state=True)
# encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
# # encoder_outputs도 같이 리턴받기는 했지만 여기서는 필요없으므로 이 값은 버림.
# encoder_states = [state_h, state_c]
# # LSTM은 바닐라 RNN과는 달리 상태가 두 개. 바로 은닉 상태와 셀 상태.
#
# decoder_inputs = Input(shape=(None, tar_vocab_size))
# decoder_lstm = LSTM(units=256, return_sequences=True, return_state=True)
# decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
# # 디코더의 첫 상태를 인코더의 은닉 상태, 셀 상태로 합니다.
# decoder_softmax_layer = Dense(tar_vocab_size, activation='softmax')
# decoder_outputs = decoder_softmax_layer(decoder_outputs)
#
# model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
# model.compile(optimizer="rmsprop", loss="categorical_crossentropy")
#
# model.fit(x=[encoder_input, decoder_input], y=decoder_target, batch_size=64, epochs=50, validation_split=0.2)
#
