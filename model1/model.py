# coding=utf-8

from keras import Input, Model
from keras.layers import Embedding, Dense, Conv1D, GlobalMaxPooling1D, Concatenate, Dropout


class model(object):
    def __init__(self, feature_len,last_activation='sigmoid'):
        self.feature_len=feature_len
        self.last_activation = last_activation

    def get_model(self):

        input = Input((self.feature_len,))
        dense1=Dense(128,activation='relu')(input)
        dense2 = Dense(64, activation='relu')(dense1)
        decoder_final = Dense(32, activation='relu')(dense2)

        # decoder layers
        decoded1 = Dense(64, activation='relu')(decoder_final)
        decoded2 = Dense(128, activation='relu')(decoded1)
        decoded = Dense(self.feature_len, activation='relu')(decoded2)

        # 搭建autoencoder模型
        self.autoencoder = Model(input=input, output=decoded)

        #  搭建encoder model for plotting,encoder是autoencoder的一部分
        self.encoder = Model(input=input, output=decoder_final)

        return self.autoencoder,self.encoder

    def compile(self):
        # 编译 autoencoder
        self.autoencoder.compile(optimizer='adam', loss='mse')

