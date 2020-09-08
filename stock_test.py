import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras import Sequential
import numpy as np
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 1. 주가 데이터를 로드 합니다.
data = pd.read_csv("D:/Desktop/Itwill ws/rnn/goog.csv")
print(data.tail())
print(data.shape)

# 2. 훈련데이터와 테스트 데이터를 나눕니다.
data['Date'] = pd.to_datetime(data['Date'])
split_date = datetime.datetime(2009, 1, 1)
training_data = data[data['Date'] < split_date].copy()
test_data = data[data['Date'] >= split_date].copy()

print(training_data.shape)
print(test_data.shape)

print(training_data)
print(test_data)

# 3. 예측을 위해 필요한 데이터만 남겨둡니다.
training_data = training_data.drop(['Sym', 'Date', 'Adj Close'], axis=1)
training_data.head()
print(training_data)
print()

# 4. 정규화 합니다.
scaler = MinMaxScaler()
training_data = scaler.fit_transform(training_data)
print(training_data.shape)  # (941, 5)

# 5.
x_train = []
y_train = []

for i in range(60, training_data.shape[0]):
    x_train.append(training_data[i-60:i])
    y_train.append(training_data[i, 0])

print(x_train)
print(y_train)

# # 6. 위의 코드를 이해하기 위한 코드
#
# test_arr = np.arange(4705).reshape(941, 5)
# print(test_arr)
#
#
# x_train2 = []
# y_train2 = []
#
# for i in range(60, test_arr.shape[0]):
#     x_train2.append(test_arr[i-60:i])
#     y_train2.append(test_arr[i, 0])

# for i, j in zip(x_train2, y_train2):
#     print(i, '|', j)

# 7. 훈련 데이터와 라벨을 numpy array 로 변환
x_train, y_train = np.array(x_train), np.array(y_train)
print(x_train.shape)

# 8. 신경망 모델 생성
regression = Sequential()
regression.add(LSTM(units=50, activation="relu",
                    return_sequences=True, input_shape=(x_train.shape[1], 5)))
regression.add(Dropout(0.2))

regression.add(LSTM(units=60, activation="relu", return_sequences=True))
regression.add(Dropout(0.3))

regression.add(LSTM(units=80, activation="relu", return_sequences=True))
regression.add(Dropout(0.4))

regression.add(LSTM(units=120, activation="relu"))
regression.add(Dropout(0.5))

regression.add(Dense(units=1))

regression.summary()

# 9. 시각화
# activate keras_study

# conda uninstall pydot
# conda uninstall pydotplus
# conda uninstall graphviz

# conda install pydot
# conda install pydotplus


# tf.keras.utils.plot_model(
#     regression, 'multi_input_and_output_model.png', show_shapes=True)

# 10. 학습
regression.compile(optimizer='adam', loss='mean_squared_error')

regression.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0)

# 11. 다시  훈련데이터와 테스트 데이터를 원래대로 나눠 불러옵니다.
data['Date'] = pd.to_datetime(data['Date'])
split_date = datetime.datetime(2009, 1, 1)
training_data = data[data['Date'] < split_date].copy()
test_data = data[data['Date'] >= split_date].copy()

past_60_days = training_data.head(60)
print(past_60_days.head())

# 12. 60일 데이터 + test 데이터 붙이기
df = test_data.append(past_60_days, sort=True)
print(df.head())
print(df.tail())
print(len(df))
df = df.drop(['Sym', 'Date', 'Adj Close'], axis=1)

scaler = MinMaxScaler()
testing_data = scaler.fit_transform(df)

x_test = []
y_test = []

for i in range(60, testing_data.shape[0]):
    x_test.append(testing_data[i-60:i])
    y_test.append(testing_data[i, 0])


# %%

x_test, y_test = np.array(x_test), np.array(y_test)
print(x_test.shape)
# %%
y_pred = regression.predict(x_test)
print(y_pred)

# %%

print(scaler.scale_)


# %%

scale = 1/ scaler.scale_[3]
print(scale)

# %%

y_pred = y_pred*scale
y_test = y_test*scale

print(y_pred)
print(y_test)

# %%

# Visualising the results
plt.figure(figsize=(14, 5))
plt.plot(y_test, color='red', label='Real Google Stock Price')
plt.plot(y_pred, color='blue', label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()