import os
from os import listdir
from os.path import isfile, join
import scipy.io as sio
import numpy as np
import torch

class Dataloader():
    def __init__(self, path='', batch_size=256, device='cpu'):
        
        self.batch_size = batch_size
        self.device = device
        self.files = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
        # filter
        for i,f in enumerate(self.files):
            if not f.split('.')[-1]=='mat':
                del(self.files[i])
        #print(self.files)
        self.reset()
        
    def reset(self):
        self.done = False
        self.unvisited_files = [f for f in self.files]
        self.buffer = np.zeros((0,4,8,288))
        self.buffer_label = np.zeros((0, 1))
        self.buffer_SNR = np.zeros((0, 1))
        #print(self.buffer.shape)
    
    def load(self, file):
        data = sio.loadmat(file)
        #print(data.keys())
        ch1 = data['ch1'][:, np.newaxis, :, :]
        ch2= data['ch2'][:, np.newaxis, :, :]
        ch3= data['ch3'][:, np.newaxis, :, :]
        ch4= data['ch4'][:, np.newaxis, :, :]
        channels = np.concatenate((ch1,ch2,ch3,ch4),axis=1)
        labels = data['flags']
        SNR = data['SNR']
        #print(labels.shape)
        return channels, labels, SNR
    
    def pre_process(self, channels):
        return 0
        
    def next_batch(self):
        done = False
        while self.buffer.shape[0] < self.batch_size:            
            if len(self.unvisited_files) == 0:
                done = True
                break
            #print('Load',self.unvisited_files[0])
            channels, labels, SNR = self.load(self.unvisited_files.pop(0))
            self.buffer = np.concatenate((self.buffer, channels), axis=0)
            self.buffer_label = np.concatenate((self.buffer_label, labels), axis=0)
            self.buffer_SNR = np.concatenate((self.buffer_SNR, SNR), axis=0)
        out_size = min(self.batch_size, self.buffer.shape[0])
        batch_channels = self.buffer[0:out_size,:,:,:] * 10e5
        batch_labels = np.squeeze(self.buffer_label[0:out_size,:])
        batch_labels[batch_labels == -1] = 0
        batch_SNR = np.squeeze(self.buffer_SNR[0:out_size,:])
        self.buffer = np.delete(self.buffer, np.s_[0:out_size], 0)
        self.buffer_label = np.delete(self.buffer_label, np.s_[0:out_size], 0)
        self.buffer_SNR = np.delete(self.buffer_SNR, np.s_[0:out_size], 0)
        #print('batch size:', batch_channels.shape, 'buffer size:', self.buffer.shape)
        
        batch_channels = np.float32(batch_channels)
        batch_labels = np.float32(batch_labels)
        batch_SNR = np.float32(batch_SNR)
        
        return torch.from_numpy(batch_channels).to(self.device), torch.from_numpy(batch_labels).to(self.device), torch.from_numpy(batch_SNR).to(self.device), done
        
    
        