# coding: utf-8
import sys
sys.path.append('D:/Desktop/Itwill ws/rnn')
from dataset import sequence

(x_train, t_train), (x_test, t_test) = sequence.load_data('addition.txt', seed=1984)

char_to_id, id_to_char = sequence.get_vocab()

print(x_train.shape, t_train.shape)
print(x_test.shape, t_test.shape)

print(x_train[0])
print(t_train[0])

print(''.join([id_to_char[c] for c in x_train[0]]))
print(''.join([id_to_char[c] for c in t_train[0]]))


# 띄어쓰기
sent = '오늘끗나면모하지공부할까 곱창전골먹을까 베고프다.'
sent = sent.replace(' ', '')

# print(new_sent)

from pykospacing import spacing
sent = spacing(new_sent)
print(sent)

# print(kospacing_sent)
# 맞춤법
from hanspell import spell_checker
sent = spell_checker.check(sent)
sent = spelled_sent.checked
print(sent)

# text를 숫자로 변환하는 모듈
from tensorflow.keras.preprocessing.text import Tokenizer

# 패딩하여 일관된 문장 길이로 만들어지기 위한 모듈
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# onehot표현을 위한 모듈
from tensorflow.keras.utils import to_categorical

# 문자를 숫자로 변환하는 클래스 인스턴스화
t = Tokenizer()

# text로 corpus, word_to_id, id_to_word 생성
t.fit_on_texts([sent])

# word_to_id 출력
print(t.word_index)

sequences = list()

# \n을 기준으로 문장 토큰화
for line in sent.split('\n'):
    encoded = t.texts_to_sequences([line])

    for i in range(1,len(encoded[0])):
        sequence = encoded[0][0:i+1]
        sequences.append(sequence)

max_len = max(len(i) for i in sequences)
print(max_len)

from tensorflow.keras.preprocessing.sequence import pad_sequences
sequences = pad_sequences(sequences, maxlen = max_len, padding = 'pre')

# print(sequences)

# 훈련데이터 / 그의 라벨로 분리
import numpy as np

sequences = np.array(sequences)

X = sequences[:,0:-1]
y = sequences[:, -1]

print(X)
print(y)
