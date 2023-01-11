import pymysql
import random
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint

conn = pymysql.connect(host='localhost', port=3308, user='root', password='8073298c', db='bssm', charset='utf8')
curs = conn.cursor()
sql = "select * from my_table"
curs.execute(sql)
rows = curs.fetchall()

# train_dataset = tfds.load('iris',split='train[:80%]') #80% 우리 db로 교체되어야함
# valid_dataset = tfds.load('iris',split='train[80%:]') #20%

np.array(rows)
print(rows)
# def preprocessing(data): #iris 데이터를 가공
#   x = data['features']
#   y = data['label']
#   y = tf.one_hot(y, 3) #원 핫 벡터 함수 사용
#   # 0 -> [1, 0, 0]
#   # 1 -> [0, 1, 0]
#   # 2 -> [0, 0, 1]
#   return x,y
#
#
# batch_size = 10
# train_data = train_dataset.map(preprocessing).batch(batch_size) #학습할때 한꺼번에 하면 수학적으로 안됨, 나눠서 하기
# valid_data = valid_dataset.map(preprocessing).batch(batch_size) #학습할때 한꺼번에 하면 수학적으로 안됨, 나눠서 하기
#
# for batch in train_data.take(3):
#   print(batch[0])
#   print(batch[1])
#
# model = Sequential([
#     Dense(512, activation='relu', input_shape = (4, )),
#     Dense(256, activation='relu'),
#     Dense(128, activation='relu'),
#     Dense(64, activation='relu'),
#     Dense(32, activation='relu'),
#     Dense(16, activation='relu'),
#     Dense(3, activation = 'softmax'),
# ])
# model.compile(optimizer = 'adam',
#               loss = 'categorical_crossentropy',
#               metrics = ['acc']
#               )
#
# checkpoint_path = 'checkpoint.ckpt'
# checkpoint = ModelCheckpoint(
#     filepath = checkpoint_path,
#     save_weights_only = True,
#     save_best_only = True,
#     mpnitor = 'val_loss',
#     verbose = 1
# )
#
# epochs = 20
# history = model.fit(train_data,
#                     validation_data = (valid_data),
#                     epochs = epochs,
#                     callbacks = [checkpoint]
# )
# plt.figure(figsize = (12,9))
# plt.plot(np.arange(1,epochs+1), history.history['loss'])
# plt.plot(np.arange(1,epochs+1), history.history['val_loss'])
# plt.title('Loss / Val Loss', fontsize = 20)
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend(['loss', 'val_loss'], fontsize =15)
# plt.show()