# coding=utf-8

import numpy as np

view1=np.random.random((1000,1000))
y=view1*3+1
view2=np.random.random((1000,1000))
view3=np.random.random((1000,1000))


from model1.model import model
aeModel=model(1000)
aeModel.get_model()
aeModel.compile()
print(aeModel.autoencoder)
# 训练
aeModel.autoencoder.fit(view1, view1,
                nb_epoch=20,
                batch_size=256,
                shuffle=True)



