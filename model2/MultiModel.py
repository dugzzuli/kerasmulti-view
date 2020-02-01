# coding=utf-8

from keras import Input, Model
from keras.layers import Embedding, Dense, Conv1D, GlobalMaxPooling1D, Concatenate, Dropout


class MModel(object):
    def __init__(self, feature_len,last_activation='sigmoid'):
        self.feature_len=feature_len
        self.last_activation = last_activation

    def get_model(self):

        input = Input((self.feature_len,))
        dense1=Dense(128,activation='relu')(input)
        dense2 = Dense(64, activation='relu')(dense1)
        decoder_final = Dense(8, activation='relu')(dense2)

        input2 = Input((self.feature_len,))
        dense21 = Dense(128, activation='relu')(input2)
        dense22 = Dense(64, activation='relu')(dense21)
        decoder_final2 = Dense(8, activation='relu')(dense22)

        input3 = Input((self.feature_len,))
        dense31 = Dense(128, activation='relu')(input3)
        dense32 = Dense(64, activation='relu')(dense31)
        decoder_final3 = Dense(8, activation='relu')(dense32)

        convs = []
        convs.append(decoder_final)
        convs.append(decoder_final2)
        convs.append(decoder_final3)

        x = Concatenate()(convs)

        # decoder layers
        decoded1 = Dense(64, activation='relu')(x)
        decoded2 = Dense(128, activation='relu')(decoded1)
        decoded = Dense(self.feature_len, activation='relu')(decoded2)

        # 搭建autoencoder模型
        self.autoencoder = Model(input=[input,input2,input3], output=decoded)

        #  搭建encoder model for plotting,encoder是autoencoder的一部分
        self.encoder = Model(input=[input,input2,input3], output=x)

        return self.autoencoder,self.encoder

    def compile(self):
        # 编译 autoencoder
        self.autoencoder.compile(optimizer='adam', loss='mse')

