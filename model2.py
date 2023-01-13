import pymysql
import random
import numpy as np
import tensorflow as tf
import datetime
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
def model2(datq):
    days = [0, 1, 2, 3, 4, 5, 6]
    today = days[datq.weekday()]+1
    if today == 0:
        datz = [[1, 0, 0, 0, 0, 0, 0]]
    elif today == 1:
        datz = [[0, 1, 0, 0, 0, 0, 0]]
    elif today == 2:
        datz = [[0, 0, 1, 0, 0, 0, 0]]
    elif today == 3:
        datz = [[0, 0, 1, 0, 0, 0, 0]]
    elif today == 4:
        datz = [[0, 0, 0, 1, 0, 0, 0]]
    elif today == 5:
        datz = [[0, 0, 0, 0, 1, 0, 0]]
    elif today == 6:
        datz = [[0, 0, 0, 0, 0, 1, 0]]
    elif today == 7:
        datz = [[0, 0, 0, 0, 0, 0, 1]]

    conn = pymysql.connect(host='10.150.150.191', port=3307, user='djdn', password='djdn', db='djdn_warning', charset='utf8')

    curs = conn.cursor()

    sql_2 = "select * from tbl_density where camera_id = 2"

    curs.execute(sql_2)
    rows = curs.fetchall()
    rows = np.array(rows)


    train_size = int(rows.shape[0] * 0.8)
    rows_size = rows.size//4

    #최종 값인 data 생성
    data = []

    #요일로 변환하면서 필요한 값만 data 튜플에 저장
    for i in range(rows_size):
      # record = [days[(rows[i])[1].weekday()], int((rows[i])[2])]
      record = [0, 0, 0, 0, 0, 0, 0, int((rows[i])[2])]
      record[days[(rows[i])[1].weekday()]] = 1
      data.append(record)

    #np.array로 형변환
    data = np.array(data)


    train_dataset = data[:train_size]
    valid_dataset = data[train_size:]

    train_x = []
    train_y = []
    valid_x = []
    valid_y = []



    for i in range(train_dataset.size//8):
      train_x.append((train_dataset[i])[:-1])
      train_y.append([(train_dataset[i])[-1]])

    for i in range(valid_dataset.size//8):
      valid_x.append((valid_dataset[i])[:-1])
      valid_y.append([(valid_dataset[i])[-1]])
    #
    # train_x = tf.data.Dataset(train_x)
    # train_y = tf.data.Dataset(train_y)
    # valid_x = tf.data.Dataset(valid_x)
    # valid_y = tf.data.Dataset(valid_y)

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    valid_x = np.array(valid_x)
    valid_y = np.array(valid_y)

    # train_merge = tf.data.Dataset.zip((train_x, train_y))
    # valid_merge = tf.data.Dataset.zip((valid_x, valid_y))

    print(train_x)
    print('----------------------')
    print(train_y)
    print('----------------------')
    print(valid_x)
    print('----------------------')
    print(valid_y)

    #모델 학습
    model = Sequential([
        Dense(512, activation='relu', input_shape = (7, )),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1),
    ])
    model.compile(optimizer = 'adam',
                  loss = 'mse',
                  metrics = ['acc']
                  )
    #체크포인트 설정
    checkpoint_path = 'checkpoint.ckpt'
    #가장 낮은거
    checkpoint = ModelCheckpoint(
        filepath = checkpoint_path,
        save_weights_only = True,
        save_best_only = True,
        mpnitor = 'val_loss',
        verbose = 1
    )

    epochs = 5

    history = model.fit(train_x, train_y,
                        validation_data = (valid_x, valid_y),
                        epochs = epochs,
                        batch_size = 0,
                        callbacks = [checkpoint]
    )

    plt.figure(figsize = (12,9))
    plt.plot(np.arange(1,epochs+1), history.history['loss'])
    plt.plot(np.arange(1,epochs+1), history.history['val_loss'])
    plt.title('Loss / Val Loss', fontsize = 20)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['loss', 'val_loss'], fontsize =15)
    plt.show()
    model.load_weights(checkpoint_path)
    datz = np.array(datz)
    return(str(model.predict(datz)))

