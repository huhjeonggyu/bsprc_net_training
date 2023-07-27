# ---
# jupyter:
#   jupytext:
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import random
import glob
import pickle
import numpy as np
import torch
import torch.utils.data as data
from torch.distributions.normal import Normal
# +
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt') # Save the model
        self.val_loss_min = val_loss

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# +
def get_x_y(data_num,mode,data_dir='data') :
    
    file_num = data_num//10000
    filelist = glob.glob(f'{data_dir}/{mode}/bs_{mode}_*.pkl')
    filelist = np.sort(filelist)
    filelist = filelist[:file_num]
    
    x = []; y = []
    for i in range(file_num) :
        with open(filelist[i],'rb') as f :
            x_,y_ = pickle.load(f)
        x.append(x_)
        y.append(y_)
        
    x = [e for e in x]; x = np.array(x).reshape(-1,3)
    y = [e for e in y]; y = np.array(y).reshape(-1,1)
    return (x,y)  

def make_loader_train(data_num) :
    
    x,y = get_x_y(data_num,'train')
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()

    nn = int(0.7*data_num)
    dataset_train = data.TensorDataset(x[:nn],y[:nn])
    dataset_validate = data.TensorDataset(x[nn:],y[nn:])
    loader_train = data.DataLoader(dataset_train, batch_size=50, drop_last=True, pin_memory=True)
    loader_validate = data.DataLoader(dataset_validate, batch_size=50, drop_last=True, pin_memory=True)
        
    return (loader_train,loader_validate)

def make_loader_test(data_num) :
    
    x,y = get_x_y(data_num,'test')
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()

    dataset = data.TensorDataset(x,y)    
    loader = data.DataLoader(dataset, batch_size=50, drop_last=True, pin_memory=True)
    
    return loader

