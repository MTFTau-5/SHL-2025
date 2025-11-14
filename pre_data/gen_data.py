import os
import psutil
import numpy as np
from tqdm import tqdm
from scipy.stats import mode
from joblib import Parallel , delayed
from numpy.lib.format import open_memmap

def gen_data(dataset, modal_list, file_list, shape):
    file_list = list(file_list)

    #生成数据
    data = open_memmap('data/npy_data/{}/data.npy'.format(dataset),dtype=np.float32,mode='w+',shape=shape)
    for i,modal in enumerate(tqdm(modal_list)):
        #生成实际file位置
        tmp_list = list(file_list)
        for x,_ in enumerate(file_list):
            tmp_list[x] = os.path.join('data/raw_data',dataset,modal,file_list[x])
        if dataset == 'train':
            Parallel(n_jobs=psutil.cpu_count(), verbose=0)(delayed(lambda j,f: data[i].__setitem__(j, np.loadtxt(f)))(j,f) for j,f in enumerate(tmp_list))
            # process some nan，data[2,3,121217] -》data[2,8,121217] contain nan
            if modal == 'Hips':
                data[2,:,121217] = np.nan_to_num(data[2,:,121217],copy=False,nan=0.005)
        else:
            Parallel(n_jobs=psutil.cpu_count(), verbose=0)(delayed(lambda j,f: data[i].__setitem__(j, np.loadtxt(f,delimiter=',')))(j,f) for j,f in enumerate(tmp_list))

def gen_test(file_list, shape=(9, 92726, 500)):
    #生成实际file位置
    file_list = list(file_list)
    for i,_ in enumerate(file_list):
        file_list[i] = os.path.join('data/raw_data/test',file_list[i])
    
    if not os.path.exists('data/npy_data/test'):
        os.mkdir('data/npy_data/test')
    
    #生成数据
    data = open_memmap('data/npy_data/test/data.npy',dtype=np.float32,mode='w+',shape=shape)
    
    Parallel(n_jobs=psutil.cpu_count(), verbose=0)(delayed(lambda i: data.__setitem__(i, np.loadtxt(file_list[i],delimiter=',')))(i) for i in tqdm(range(9)))

    
def gen_label(set, modal):
    path=os.path.join('data/npy_data',set)
    if not os.path.exists(path):
        os.mkdir(path)
    
    source_path = os.path.join('data/raw_data',set,modal,'Label.txt')

    source = np.loadtxt(source_path, dtype=np.int8)
    label = np.zeros((len(source)),dtype=np.int8)
    
    label = mode(source,axis=1)[0]
    
    np.save('data/npy_data/{}/label.npy'.format(set),label)

if __name__ == '__main__':

    #形状对应：模态，数据源(九个通道代表Acc、Gyr、Mag的各xyz通道)，帧，500个样本
    shape_list = [(4, 9, 196072, 500),(4, 9, 28789, 500)]

    sets = ('train', 'valid')

    modal_list = ("Bag","Hand","Hips","Torso")

    file_list = ("Acc_x.txt", "Acc_y.txt", "Acc_z.txt","Gyr_x.txt", "Gyr_y.txt", "Gyr_z.txt","Mag_x.txt", "Mag_y.txt", "Mag_z.txt")

    if not os.path.exists('data/npy_data'):
        os.mkdir('data/npy_data')

    # gen_label
    for set in sets:
        print("processing {} Label".format(set))
        gen_label(set,modal_list[0])

    # process test data
    print("processing test data")
    gen_test(file_list=file_list)

    # gen_data
    for i,set in enumerate(sets):
        print("processing {} data".format(set))
        gen_data(set, modal_list,file_list,shape_list[i])