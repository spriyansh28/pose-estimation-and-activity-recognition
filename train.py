import argparse
import pandas as pd
import numpy as np
import data_preprocessing as dpp

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from tensorflow import keras 

from keras.utils import np_utils
from keras.models import load_model

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from keras.layers import LeakyReLU


if __name__ == "__main__":

    key_word = "--dataset"
    parser = argparse.ArgumentParser()
    parser.add_argument(key_word, required=False, default='dataset.csv')
    input = parser.parse_args().dataset

    raw_data = pd.read_csv(input, header=0)
 
    dataset = raw_data.values
    X = dataset[:, 0:36].astype(float)
    Y = dataset[:, 36]


    X_pp = []
    for i in range(len(X)):
        X_pp.append(dpp.pose_normalization(X[i]))
    X_pp = np.array(X_pp)

    encoder = LabelEncoder()
    encoder_Y = encoder.fit_transform(Y)
    matrix_Y = np_utils.to_categorical(encoder_Y)
    print(Y[0], ": ", encoder_Y[0])
    print(Y[650], ": ", encoder_Y[650])
    print(Y[1300], ": ", encoder_Y[1300])
    print(Y[1950], ": ", encoder_Y[1950])
    print(Y[2600], ": ", encoder_Y[2600])


    X_train, X_test, Y_train, Y_test = train_test_split(X_pp, matrix_Y, test_size=0.1, random_state=42)

    model = Sequential()
    model.add(Dense(units=128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(units=64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(units=5, activation='sigmoid'))
    model.add(BatchNormalization())
    model.add(Dense(units=5, activation='softmax'))

    model.compile(optimizer=Adam(0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train, Y_train, batch_size=32, epochs=50, verbose=2, validation_data=(X_test, Y_test))

    model.save('action_poses.h5')
