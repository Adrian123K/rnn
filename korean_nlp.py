# !pip install git+https://github.com/haven-jeon/PyKoSpacing.git --user
from pykospacing import spacing

sent = '김철수   는    극중 두 인격     의 사나이 이광수 역을 맡았     다. 철수는 한국 유일의 태권도 전승자를 가리는 결전의 날을 앞두고 10년간 함께 훈련한 사형인 유연재(김광수 분)를 찾으러 속세로 내려온 인물이다.'

new_sent = sent.replace(" ", '') # 띄어쓰기가 없는 문장 임의로 만들기
print(new_sent)

kospacing_sent = spacing(new_sent)
print(sent)

print(kospacing_sent)

sent2 = "아기다리 고기다리던 봄방학는 잘 안돼요"

new_sent2 = sent2.replace(" ",'')
kospacing_sent2 = spacing(new_sent2)
print(kospacing_sent2)

# !pip install git+https://github.com/ssut/py-hanspell.git --user

from hanspell import spell_checker

sent = "맞춤법 틀리면 외 않되? 쓰고싶은대로쓰면돼지 "

spelled_sent = spell_checker.check(sent)

hanspell_sent = spelled_sent.checked
print(hanspell_sent)

test_sent = spell_checker.check(sent2)
test2 = test_sent.checked
print(test2)

# !pip install konlpy
# !pip install customized_konlpy

from ckonlpy.tag import Twitter

twitter = Twitter()

twitter.morphs('은경이는 사무실로 갔습니다.')
twitter.add_dictionary('은경이', 'Noun')

print( twitter.morphs('은경이는 사무실로 갔습니다.'))

# 오늘의 마지막 문제. 문법에 맞지 않은 한글 문장을 문법검사해서 잘 구성한 다음 토큰화해서 RNN신경망에 입력되기 전 데이터인
# 훈련 데이터와 라벨로 구성되게 하시오

test = "RNN신경망은 너무어려워서 하나도모르 겠습니다.\n 외않되는지매 일공부해 봐도모르 겠습니다.\n 살려주세 요."
gram_test = spell_checker.check(test)
rs_test = gram_test.checked
print(rs_test)

from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer # text를 숫자로 변환하는 모듈

t = Tokenizer()
t.fit_on_texts([rs_test]) # text 문자를 가지고 corpus, word_to_id, id_to_word를 생성

sequences = list()
for line in rs_test.split('\n'):
    encoded = t.texts_to_sequences([line])[0]
    for i in range(1,len(encoded)):
        sequence = encoded[:i+1]
        sequences.append(sequence)

max_len = max(len(l) for l in sequences)

from tensorflow.keras.preprocessing.sequence import pad_sequences
sequences = pad_sequences(sequences, maxlen=14, padding='pre')

from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
sequences = np.array(pad_sequences(sequences, maxlen=14, padding='pre'))
X = sequences[:,:-1]
y = sequences[:,-1]
X, y