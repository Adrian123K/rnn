import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import datetime

scaler = MinMaxScaler()

# 1. 데이터 로드
data = pd.read_csv("D:/Desktop/Itwill ws/rnn/cacao5.csv")
data.info()

# 2. 훈련 데이터, 테스트 데이터 분리
data['date'] = pd.to_datetime(data['date'])
data = data.set_index('date')
data = data.drop(['Unnamed: 0','diff'], axis=1)
data.head()

# 3. 데이터 정규화
data_sc = scaler.fit_transform(data)

# 4. 데이터프레임으로 형 변환
data_sc_pd = pd.DataFrame(data_sc, index=data.index, columns=data.columns)
data_sc_pd.head()

# 5. 기준날짜로 훈련 데이터와 테스트 데이터 분리
split_date = datetime.datetime(2020, 9, 3) # 기준날짜
training_data = data_sc_pd.loc[:split_date].copy().reindex()
test_data = data_sc_pd.loc[split_date:].copy().reindex()

past_58_days = training_data.iloc[len(training_data)-59:len(training_data)-1].copy() # 58개의 데이터 추가
past_58_days.tail()

# 기존 테스트 데이터(2개)와 59개의 과거 데이터를 병합하여 61개의 데이터로 생성
test_combine = past_58_days.append(test_data, sort=True) # 61일치 test 데이터 생성
test_combine.head() # 데이터 확인
test_combine.tail()
len(test_combine) # 60

# 그래프로 확인
ax = training_data.close.plot() # 훈련 부분의 그래프
test_combine.close.plot(ax=ax) # 테스트 부분의 그래프를 다른색으로 뒷부분에 함께 표현

# 6. 종가 데이터로만 훈련 데이터 및 테스트 데이터 분리
training_close = pd.DataFrame(training_data['close'], index=training_data.index)
test_close = pd.DataFrame(test_combine['close'], index=test_combine.index)

# 7. 시계열 분석을 위해 데이터 shift
for s in range(1, 60):
    training_close['shift_{}'.format(s)] = training_close['close'].shift(s)
    test_close['shift_{}'.format(s)] = test_close['close'].shift(s)
training_close.head()
test_close.head()

# 8. 훈련데이터와 테스트 데이터의 라벨 생성
x_train = training_close.dropna().drop('close',axis=1).values
y_train = training_close.close.values

x_test = test_close.dropna().drop('close',axis=1).values
y_test = test_close.close.values

X_train = x_train.reshape(-1,59,1)
X_test = x_test.reshape(-1,59,1)
print(X_train.shape)
print(X_test.shape)

# 9. 신경망 모델을 생성

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

regression = Sequential()
regression.add(LSTM(units=50, activation="relu", return_sequences=True, input_shape=(X_train.shape[1], 1)))
regression.add(Dropout(0.2))

regression.add(LSTM(units=60, activation="relu", return_sequences=True))
regression.add(Dropout(0.3))

regression.add(LSTM(units=80, activation="relu", return_sequences=True))
regression.add(Dropout(0.4))

regression.add(LSTM(units=120, activation="relu"))
regression.add(Dropout(0.5))

regression.add(Dense(units=1))

regression.summary()

# 학습
regression.compile(optimizer='adam', loss='mean_squared_error')
regression.fit(X_train, y_train[len(y_train)-len(X_train):], epochs=20, batch_size=32)

# 10. 예측
y_pred = regression.predict(X_test)
print(y_pred)

print(scaler.scale_)

scale = 1 / scaler.scale_[0]
print(scale)

y_pred = y_pred * scale
y_test = y_test * scale
print(y_pred)
print(y_test)

#
# # Visualising the results
# plt.plot(y_test, color='red', label='Real Cacao Stock Price') # 훈련 부분의 그래프
# plt.plot(y_pred, color='blue', label='Predicted Cacao Stock Price') # 테스트 부분의 그래프를 다른색으로 뒷부분에 함께 표현
# plt.title('Cacao Stock Price Prediction')
# plt.xlabel('Time')
# plt.ylabel('Cacao Stock Price')
# plt.legend()
# plt.show()