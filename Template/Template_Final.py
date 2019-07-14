import keras
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import cv2
import sys

from os import listdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from Template_LSTM import LSTM_TEMPLATE as LSTM
from Template_ResNet import ResNet_TEMPLATE as ResNet
from Template_MLP import MLP
from keras.callbacks import TensorBoard
from datetime import datetime


#def main(name_param, name_modele, batch_size, epochs, lera, activation, nb_layer, nb_filtre):

if __name__ == "__main__":

    name_param = sys.argv[1]
    print("name_param = " + name_param)
    name_modele = sys.argv[2]
    print("name_modele = " +name_modele)
    batch_size = sys.argv[3]
    print("batch_size = " + batch_size)
    epochs = sys.argv[4]
    print("epochs = " + epochs)
    lera = sys.argv[5]
    print("lera = " + lera)
    activation = sys.argv[6]
    print("activation = " + activation)
    nb_layer = sys.argv[7]
    print("nb_layer = " + nb_layer)
    nb_filtre = sys.argv[8]
    print("nb_filtre = " + nb_filtre)
    nb_dropout_flag = sys.argv[9]
    print("nb_dropout_flag = " + nb_dropout_flag)
    nb_dropout_value = sys.argv[10]
    print("nb_dropout_value = " + nb_dropout_value)

    ## Setup memory use
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    
    ## Preprocessing

    train_csv = pd.read_csv('../Data/data/train.csv')

    filenames = ['../Data/data/train/' + fname for fname in train_csv['id'].tolist()]
    labels = train_csv['has_cactus'].tolist()

    train = []
    for file_name in filenames:
        img = cv2.imread(file_name)
        img = img.reshape(32*32*3,)
        img = img/255
        train.append(img)

    x_train, x_test, y_train, y_test = train_test_split(train,
                                                        labels,
                                                        train_size=0.9)


    x_train = np.array(x_train)
    if name_modele != 'MLP':
        x_train = np.reshape(x_train, (-1, 32, 32, 3))
    print(x_train.shape)

    x_test = np.array(x_test)
    if name_modele != 'MLP':
        x_test = np.reshape(x_test, (-1, 32, 32, 3))
    print(x_test.shape)

    if name_modele == 'MLP':
        inputs = Input(shape=train[0].shape)
    else:
        inputs = Input(shape=(32, 32, 3))
    print(inputs)

    y_train = keras.utils.to_categorical(y_train, 2)
    y_test = keras.utils.to_categorical(y_test, 2)


    ## Modele

    if name_modele == "LSTM":
        print(name_modele + " " + name_param)
        outputs = LSTM(inputs, nb_filtre, nb_layer, nb_dropout_flag, nb_dropout_value)
    if name_modele == "ResNet":
        print(name_modele + " " + name_param)
        outputs = ResNet(nb_layer, inputs, activation, nb_dropout_flag, nb_dropout_value)
    if name_modele == "MLP":
        outputs = MLP(inputs, activation, nb_layer, nb_filtre)

    ## Run model
    model = Model(inputs=inputs, outputs=outputs)

    if name_modele == 'MLP':
        model.compile(loss='binary_crossentropy',
                      optimizer=Adam(lr=float(lera)),
                      metrics=['accuracy'])
    else:
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=float(lera)),
                      metrics=['accuracy'])

    model.summary()

    # Prepare model model saving directory.
    save_dir = os.path.join(os.getcwd(), 'res_logs')
    date = datetime.today()
    year = date.strftime("%Y")
    month = date.strftime("%m")
    day = date.strftime("%d")
    hour = date.strftime("%H")
    minute = date.strftime("%M")
    model_name = "{}{}{}{}{}{}{}_es{}_lr{}_bs{}_{}_ly{}_nf{}" \
        .format(name_param, name_modele, year, month, day, hour, minute, epochs, lera, batch_size, activation, nb_layer, nb_filtre)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)
    callbacks = TensorBoard(log_dir=filepath)

    model.fit(x_train, y_train,
              batch_size=int(batch_size),
              epochs=int(epochs), #int(1)
              validation_data=(x_test, y_test),
              callbacks=[callbacks])

    # predicted class
    filenames_test = ['../Data/data/test/' + f for f in listdir('../Data/data/test/') if isfile(join('../Data/data/test/', f))]
    print(len(filenames_test))
    test = []
    i = 0
    for file_name in filenames_test:
        img = cv2.imread(file_name)
        # print("--------------------------------------------------------------------------------------------------"+str(i)+str(img))
        # i=i+1
        img = img.reshape(32 * 32 * 3, )
        img = img / 255
        test.append(img)
    test = np.array(test)
    test = np.reshape(test, (-1, 32, 32, 3))
    pred1 = model.predict(test)
    predf = pred1.argmax(axis=-1)
    print(predf)
    # np.savetxt("predict.csv", np.c_[filenames_test,predf], delimiter=",")
    df = pd.DataFrame({"id": filenames_test, "has_cactus": predf})
    df.to_csv("./prediction/predict_{}_{}.csv".format(name_modele, name_param), index=False)
    #keras.backend.clear_session()
    #del model
    #torch.cuda.empty_cache()
    #time.sleep(.300)
    #csvFile.close()
