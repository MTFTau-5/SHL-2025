import os
import psutil
import numpy as np
from tqdm import tqdm
from shutil import copyfile
from numpy.lib.format import open_memmap

import jax
import jax.numpy as jnp

from functools import partial

@partial(jax.jit, static_argnames=('window_length', 'step', 'fs'))
def window_rfft(signal, window_length=100, step=20, fs=100):
    """
    Computes the RFFT for segments of 1D signal using a sliding window.
    """
    n_subframes = 1 + (signal.shape[0] - window_length) // step
    
    starts = jnp.arange(n_subframes) * step
    signal = jax.vmap(lambda start: jax.lax.dynamic_slice(signal, (start,), (window_length,)))(starts)
    # 去均值
    signal = signal - jnp.mean(signal, axis=1, keepdims=True)
    # 计算RFFT
    signal = jnp.abs(jnp.fft.rfft(signal))
    signal /= (fs / 2)  # 归一化
    # 调整直流分量
    signal = signal.at[:, 0].set(signal[:, 0] / 2)
    return signal.reshape(-1)

def gen_rfft(dataset, shape, batchsize):
    source = np.load('data/npy_data/{}/data.npy'.format(dataset),mmap_mode='r')
    data = open_memmap('data/fft_data/{}/data.npy'.format(dataset),dtype=np.float16,mode='w+',shape=shape)
    for i in tqdm(range(shape[0])):
        data[i] = jax.lax.map(window_rfft,source[i].reshape(-1,500),batch_size=batchsize).reshape(shape[1],shape[2],shape[3])
            
def gen_test(batchsize, shape=(9, 92726, 1071)):
    if not os.path.exists('data/fft_data/test'):
        os.mkdir('data/fft_data/test')
    
    source = np.load('data/npy_data/test/data.npy',mmap_mode='r')
    data = open_memmap('data/fft_data/test/data.npy',dtype=np.float16,mode='w+',shape=shape)
    
    for i in tqdm(range(shape[0])):
        data[i] = jax.lax.map(window_rfft,source[i].reshape(-1,500),batch_size=batchsize).reshape(shape[1],shape[2])

if __name__ == '__main__':

    #形状对应：模态，数据源(九个通道代表Acc、Gyr、Mag的各xyz通道)，帧，500个样本
    shape_list = [(4, 9, 196072, 1071),(4, 9, 28789, 1071)]

    sets = ('train', 'valid')

    modal_list = ("Bag","Hand","Hips","Torso")

    batchsize = 65536
    
    if not os.path.exists('data/fft_data'):
        os.mkdir('data/fft_data')
        for set in sets:
            path=os.path.join('data/fft_data',set)
            if not os.path.exists(path):
                os.mkdir(path)
    
    # copy label
    for set in sets:
        copyfile('data/npy_data/{}/label.npy'.format(set),'data/fft_data/{}/label.npy'.format(set))
        print("processed {} Label".format(set))

    # gen test
    gen_test(batchsize)
    print("processed test data")

    # gen_rfft
    for i,set in enumerate(sets):
        gen_rfft(set, shape_list[i], batchsize)
        print("processed {}".format(set))
