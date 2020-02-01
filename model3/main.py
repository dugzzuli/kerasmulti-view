# coding=utf-8

import numpy as np
import scipy.io as sio

from Util.Util import node_clustering


def get_data(path='data/3sources.mat'):
    data=sio.loadmat(path)
    a_sque = np.squeeze(data.get("data"))
    return a_sque

def get_label(path='data/3sources.mat'):
    data=sio.loadmat(path)
    a_sque = np.squeeze(data.get("truelabel"))
    return np.transpose(a_sque[0])


data=get_data()
label=get_label()
print(np.shape(label))


view1=np.transpose(data[0])
view2=np.transpose(data[1])
view3=np.transpose(data[2])
shape1=(np.shape(view1))
shape2=(np.shape(view2))
shape3=(np.shape(view3))

viewA=np.concatenate([view1,view2,view3],axis=1)
print(np.shape(viewA))
from model3.AutoIntegrate import AutoModel
aeModel=AutoModel(arr_len=[shape1[1],shape2[1],shape3[1]],layers=[512,128,64])
aeModel.get_model()
aeModel.compile()
print(aeModel.autoencoder.summary())
# 训练
aeModel.autoencoder.fit([view1,view2,view3], [view1,view2,view3], nb_epoch=10000,batch_size=169, shuffle=False)

emb=aeModel.encoder.predict([view1,view2,view3])

acc,nmi=node_clustering(emb,label)
print(acc,nmi)




