# coding=utf-8

import numpy as np
import scipy.io as sio

from Util.Util import node_clustering


def get_data(path='data/BBC4view_685.mat'):
    data=sio.loadmat(path)
    a_sque = np.squeeze(data.get("data"))
    return a_sque

def get_label(path='data/BBC4view_685.mat'):
    data=sio.loadmat(path)
    a_sque = np.squeeze(data.get("truelabel"))
    return np.transpose(a_sque[0])


data=get_data()
label=get_label()
print(np.shape(label))


view1=np.transpose(data[0])
view2=np.transpose(data[1])
view3=np.transpose(data[2])
view4=np.transpose(data[3])
shape1=(np.shape(view1))
shape2=(np.shape(view2))
shape3=(np.shape(view3))
shape4=(np.shape(view4))

from AutoIntegrate import AutoModel
aeModel=AutoModel(arr_len=[shape1[1],shape2[1],shape3[1],shape4[1]],layers=[512,256,128])
aeModel.get_model()
aeModel.compile()
print(aeModel.autoencoder.summary())
# 训练
aeModel.autoencoder.fit([view1,view2,view3,view4], [view1,view2,view3,view4], nb_epoch=5000,batch_size=shape1[0], shuffle=True)

emb=aeModel.encoder.predict([view1,view2,view3,view4])

acc,nmi=node_clustering(emb,label)
print(acc,nmi)




