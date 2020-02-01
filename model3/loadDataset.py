import  scipy.io as sio
import numpy as np

def get_data(path='data/3sources.mat'):
    data=sio.loadmat(path)
    a_sque = np.squeeze(data.get("data"))
    return a_sque

