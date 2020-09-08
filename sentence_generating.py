from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
"""
문장마다 길이가 다르므로 패딩을 해서 일관된 문장 길이로 만들어주는 모듈
신경망에 배치처리를 하려면 일관된 사이즈로 문장을 만들어줘야 하기 때문에 필요
"""
from tensorflow.keras.preprocessing.text import Tokenizer # text를 숫자로 변환하는 모듈

text = """경마장에 있는 말이 뛰고 있다\n 그의 말이 법이다\n 가는 말이 고와야 오는 말이 곱다"""

t = Tokenizer()
t.fit_on_texts([text]) # text 문자를 가지고 corpus, word_to_id, id_to_word를 생성

sequences = list()
for line in text.split('\n'):
    encoded = t.texts_to_sequences([line])[0]
    for i in range(1,len(encoded)):
        sequence = encoded[:i+1]
        sequences.append(sequence)

max_len = max(len(l) for l in sequences)

from tensorflow.keras.preprocessing.sequence import pad_sequences
sequences = pad_sequences(sequences, maxlen=6, padding='pre')

from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
sequences = np.array(pad_sequences(sequences, maxlen=6, padding='pre'))
X = sequences[:,:-1]
y = sequences[:,-1]
X, y

vocab_size = len(t.word_index) + 1
y = to_categorical(y, num_classes = vocab_size)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, SimpleRNN

vocab_size = len(t.word_index) + 1

model = Sequential()
model.add(Embedding(vocab_size, 10, input_length=max_len-1)) # 레이블을 분리하였으므로 이제 X의 길이는 5
model.add(SimpleRNN(32))
model.add(Dense(vocab_size, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=200, verbose=2)

def sentence_generation(model, t, current_word, n):  # 모델, 토크나이저, 현재 단어, 반복할 횟수
    init_word = current_word  # 처음 들어온 단어도 마지막에 같이 출력하기위해 저장
    sentence = ''
    for _ in range(n):  # n번 반복
        encoded = t.texts_to_sequences([current_word])[0]  # 현재 단어에 대한 정수 인코딩
        # print(encoded)
        encoded = pad_sequences([encoded], maxlen=5,
                                padding='pre')  # 데이터에 대한 패딩
        # print(encoded)
        result = model.predict_classes(encoded, verbose=0)

    # 입력한 X(현재 단어)에 대해서 Y를 예측하고 Y(예측한 단어)를 result에 저장.
        for word, index in t.word_index.items():
            if index == result:  # 만약 예측한 단어와 인덱스와 동일한 단어가 있다면
                break  # 해당 단어가 예측 단어이므로 break

        current_word = current_word + ' ' + word  # 현재 단어 + ' ' + 예측 단어를 현재 단어로 변경
        sentence = sentence + ' ' + word  # 예측 단어를 문장에 저장

    # for문이므로 이 행동을 다시 반복

    sentence = init_word + sentence
    return sentence

print(sentence_generation(model, t, '경마장에', 4))
print(sentence_generation(model, t, '그의', 2))
print(sentence_generation(model, t, '가는', 4))

print(sentence_generation(model, t, '오는', 6))


