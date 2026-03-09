from torch.utils.data import Dataset

import os
import pickle

class CashDataset(Dataset) :
    def __init__(self, cache_dir) :
        self.cache_dir = cache_dir
        self.length = sum(name.startswith('spectrogram_') and 
                          name.endswith('.pkl') for name in 
                          os.listdir(self.cache_dir))
    
    def __len__(self) :
        return self.length
    
    def __getitem__(self, index):
        with open(f'{self.cache_dir}//spectrogram_{index}.pkl', 'rb') as file :
            spectrogram = pickle.load(file)
            
        with open(f'{self.cache_dir}//labels_{index}.pkl', 'rb') as file :
            label = pickle.load(file)
            
        return spectrogram, label