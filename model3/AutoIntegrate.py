# coding=utf-8

from keras import Input, Model
from keras.layers import Dense, Concatenate
import numpy as np

class AutoModel(object):
    def __init__(self, arr_len,layers):
        self.con_len=np.sum(arr_len)
        self.arr_len=arr_len
        self.layers=layers


    def get_model(self):

        input_arr=[]
        encoder_cat=[]

        for lenS in self.arr_len:
            input = Input((lenS,))
            dense1=Dense(self.layers[0],activation='relu')(input)
            dense2 = Dense(self.layers[1], activation='relu')(dense1)
            decoder_final = Dense(self.layers[2])(dense2)
            input_arr.append(input)
            encoder_cat.append(decoder_final)

        x = Concatenate(axis=1)(encoder_cat)

        decoder_cat = []
        for lenS in self.arr_len:
            # decoder layers
            decoded1 = Dense(self.layers[1], activation='relu')(x)
            decoded2 = Dense(self.layers[0], activation='relu')(decoded1)
            decoded = Dense(lenS, activation='relu')(decoded2)

            decoder_cat.append(decoded)



        # 搭建autoencoder模型
        self.autoencoder = Model(input=input_arr, output=decoder_cat)

        #  搭建encoder model for plotting,encoder是autoencoder的一部分
        self.encoder = Model(input=input_arr, output=x)

        return self.autoencoder,self.encoder

    def compile(self):
        # 编译 autoencoder
        self.autoencoder.compile(optimizer='adam', loss='mse')

