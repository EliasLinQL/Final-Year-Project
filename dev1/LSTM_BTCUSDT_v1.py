import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import re

import requests
import time
import pandas as pd
from sklearn.preprocessing import StandardScaler
from collections import deque
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping


early_stopping = EarlyStopping(monitor='val_loss', patience=5)


pd.set_option('expand_frame_repr', False)

BASE_URL = 'https://api.binance.com'
limit = 1000
end_time = int(time.time() // (24 * 60 * 60) * (24 * 60 * 60) * 1000)
print(end_time)
start_time = int((time.time() - 2 * 365 * 24 * 60 * 60) * 1000)
print(start_time)
print(time.time())

while True:
    url = BASE_URL + '/api/v1/klines' + '?symbol=BTCUSDT&interval=1d&limit=' + str(limit) + '&startTime=' + str(start_time) + '&endTime=' + str(end_time)
    print(url)
    resp = requests.get(url)
    data = resp.json()
    df = pd.DataFrame(data, columns={'open_time': 0, 'open': 1, 'high': 2, 'low': 3, 'close': 4, 'volume': 5,
                                     'close_time': 6, 'quote_volume': 7, 'trades': 8, 'taker_base_volume': 9,
                                     'taker_quote_volume': 10, 'ignore': 11})
    df.set_index('open_time', inplace=True)

    df.index = pd.to_datetime(df.index, unit='ms', utc=True)
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms', utc=True)

    df.to_csv('BTCUSDT_temp.csv')
    print(df)

    if len(df) < 1000:
        break
    end_time = start_time
    start_time = int(end_time - limit * 24 * 60 * 60 * 1000)

df_final = pd.read_csv('BTCUSDT_temp.csv', index_col=0)

df_final['date'] = pd.to_datetime(df_final.index).date
df_final = df_final[['date', 'open', 'high', 'low', 'close', 'volume']]

df_final.to_csv('BTCUSDT_two_years_data.csv', index=False)


def Stock_Price_LSTM_Data_Process(df,mem_his_days,pre_days):
    df_final.dropna(inplace=True)
    df_final.sort_index(inplace=True)

    df_final['label'] = df_final['close'].shift(-pre_days)

    scaler = StandardScaler()

    X = df_final.drop(columns=['date', 'label']).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    deq = deque(maxlen=mem_his_days)
    X = []
    for i in X_scaled:
        deq.append(list(i))
        if len(deq) == mem_his_days:
            X.append(list(deq))
    X_lately = X[-pre_days:]
    X = X[:-pre_days]

    y = df_final['label'].values[mem_his_days-1:-pre_days]

    X = np.array(X)
    y = np.array(y)
    return X, y, X_lately

X, y, X_lately = Stock_Price_LSTM_Data_Process(df_final,10,1)

models_dir = './models'

if os.path.exists(models_dir):
    for file in os.listdir(models_dir):
        file_path = os.path.join(models_dir, file)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
            elif os.path.isdir(file_path):
                os.rmdir(file_path)
                print(f"Deleted directory: {file_path}")
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")
else:
    print(f"Directory does not exist: {models_dir}")



pre_days = 1
# mem_days = [5,10,15]
# lstm_layers = [1,2,3]
# dense_layers = [1,2,3]
# units = [8,16,32]
mem_days = [5]
lstm_layers = [1]
dense_layers = [1]
units = [32]

for the_mem_days in mem_days:
    for the_lstm_layers in lstm_layers:
        for the_dense_layers in dense_layers:
            for the_units in units:
                filepath = './models/{val_mape:.2f}_{epoch:02d}_'+f'mem_{the_lstm_layers}_lstm_{the_lstm_layers}_dense_{the_dense_layers}_unit_{the_units}.weights.h5'
                checkpoint = ModelCheckpoint(
                    filepath=filepath,
                    save_weights_only=True,
                    monitor='val_mape',
                    mode='min',
                    save_best_only=True
                )

                X, y, X_lately = Stock_Price_LSTM_Data_Process(df_final,the_mem_days,pre_days)
                X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle= False, test_size=0.1)

                model = Sequential()
                model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
                model.add(LSTM(the_units,activation='relu',return_sequences=True))
                model.add(Dropout(0.1))

                for i in range(the_lstm_layers):
                    model.add(LSTM(the_units,activation='relu',return_sequences=True))
                    model.add(Dropout(0.1))

                model.add(LSTM(the_units,activation='relu'))
                model.add(Dropout(0.1))

                for i in range(the_dense_layers):
                    model.add(Dense(the_units, activation='relu'))
                    model.add(Dropout(0.1))

                model.add(Dense(1))
                model.compile(loss='mse', optimizer='adam',metrics=['mape'])

                model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[checkpoint])




def find_best_model(models_dir='./models'):
    best_model = None
    best_mape = float('inf')

    for filename in os.listdir(models_dir):
        if filename.endswith('.weights.h5'):
            match = re.search(r'(\d+\.\d+)_\d+_', filename)
            if match:
                val_mape = float(match.group(1))
                if val_mape < best_mape:
                    best_mape = val_mape
                    best_model = filename

    return best_model, best_mape


best_model, best_mape = find_best_model()
print(f'Best model: {best_model} with val_mape: {best_mape}')

best_model_path = os.path.join('./models', best_model)
model.load_weights(best_model_path)
loaded_model = model

loss, mape = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test MAPE: {mape}')
pre = model.predict(X_test)

df_time = df_final['date'].iloc[-len(y_test):]

plt.figure(figsize=(10, 6))
plt.plot(df_time, y_test, color="red", label="Actual Price")
plt.plot(df_time, pre, color="green", label="Predicted Price")
plt.xlabel('Date')
plt.ylabel('Price (USDT)')
plt.title('Bitcoin Price Prediction vs Actual Price')
plt.legend()
plt.grid(True)
plt.show()
