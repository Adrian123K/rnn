import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 1. 주가 데이터를 로드 합니다.
data = pd.read_csv("D:/Desktop/Itwill ws/rnn/hynix.csv")
print(data.tail())
print(data.shape)

data = data.rename(columns= {'날짜': 'date', '종가': 'close', '전일비': 'diff', '시가': 'open', '고가': 'high', '저가': 'low', '거래량': 'volume'})
data[['close', 'diff', 'open', 'high', 'low', 'volume']] = data[['close', 'diff', 'open', 'high', 'low', 'volume']].astype(int)  # 데이터의 타입을 int형으로 바꿔줌
data['date'] = pd.to_datetime(data['date'])
data = data.sort_values(by=['date'], ascending=False)
print(data.head())

# 2. 훈련, 테스트 데이터 나누기.
import datetime

data['date'] = pd.to_datetime(data['date'])
split_date = datetime.datetime(2019, 9, 5)
training_data = data[data['date'] < split_date].copy()
test_data = data[data['date'] >= split_date].copy()

scaler = MinMaxScaler()

data = data.drop(['Unnamed: 0', 'date','diff','volume'],axis=1)
data = scaler.fit_transform(data)
data_scale = scaler.scale_[0]
print(data_scale)

# 3. 예측을 위해 필요한 데이터만 남겨둡니다.
training_data = training_data.drop(['Unnamed: 0', 'date','diff','volume'], axis=1)
# training_data = np.array(training_data)

# 4. 정규화 합니다.
training_data = scaler.fit_transform(training_data)
print(training_data.shape)  # (991, 5)

# 5.
x_train = []
y_train = []

for i in range(60, training_data.shape[0]):
    x_train.append(training_data[i - 60:i])
    y_train.append(training_data[i, 0])

# 7. 훈련 데이터와 라벨을 numpy array 로 변환합니다.
x_train, y_train = np.array(x_train), np.array(y_train)

# 8. 신경망 모델을 생성합니다.
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

regression = Sequential()
regression.add(LSTM(units=50, activation="relu", return_sequences=True, input_shape=(x_train.shape[1], 4)))
regression.add(Dropout(0.3))

regression.add(LSTM(units=60, activation="relu", return_sequences=True))
regression.add(Dropout(0.4))

regression.add(LSTM(units=80, activation="relu", return_sequences=True))
regression.add(Dropout(0.4))

regression.add(LSTM(units=120, activation="relu"))
regression.add(Dropout(0.5))

regression.add(Dense(units=1))

regression.summary()

# %% 9. 시각화 해봅니다.

# activate keras_study

# conda uninstall pydot
# conda uninstall pydotplus
# conda uninstall graphviz

# conda install pydot
# conda install pydotplus


# import tensorflow as tf

# tf.keras.utils.plot_model(regression, 'multi_input_and_output_model.png', show_shapes=True)

# 10. 학습
regression.compile(optimizer='adam', loss='mean_squared_error')
regression.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0)

# 11.
data = pd.read_csv("D:/Desktop/Itwill ws/rnn/hynix.csv")
data = data.rename(columns= {'날짜': 'date', '종가': 'close', '전일비': 'diff', '시가': 'open', '고가': 'high', '저가': 'low', '거래량': 'volume'})
data[['close', 'diff', 'open', 'high', 'low', 'volume']] = data[['close', 'diff', 'open', 'high', 'low', 'volume']].astype(int)
data['date'] = pd.to_datetime(data['date'])
data = data.sort_values(by=['date'], ascending=False)
print(data.head())

data['date'] = pd.to_datetime(data['date'])
split_date = datetime.datetime(2019, 9, 5)
training_data = data[data['date'] < split_date].copy()
test_data = data[data['date'] >= split_date].copy()

past_60_days = training_data.head(60)

print(past_60_days.head())

df = test_data.append(past_60_days, ignore_index=True)
df = df.drop(['Unnamed: 0', 'date','diff','volume'], axis=1)

# test 데이터 정규화
testing_data = scaler.fit_transform(df)
print(testing_data.shape)

# df_ar = np.array(df)
# max(df['close'])

x_test = []
y_test = []

# testing_data = np.array(testing_data)

for i in range(60, testing_data.shape[0]):
    x_test.append(testing_data[i - 60:i])
    y_test.append(testing_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
print(x_test.shape, y_test.shape)

# 예측
y_pred = regression.predict(x_test)
print(y_pred)

scale = 1 / data_scale
print(scale)
#
y_pred = y_pred * scale
y_test = y_test * scale
print(y_pred)
print(y_test)

# Visualising the results
# plt.figure(figsize=(14, 5))
plt.plot(y_test, color='red', label='Real Hynix Stock Price')
plt.plot(y_pred, color='blue', label='Predicted Hynix Stock Price')
plt.title('Hynix Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Hynix Stock Price')
plt.legend()
plt.show()

from scipy import stats
import itertools
import numpy as  np

y_pred = np.array(y_pred)
y_pred = y_pred.flatten().tolist()
print(y_pred)
print(y_test)

print(np.corrcoef(y_pred, y_test))