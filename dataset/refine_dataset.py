import torch
from torch.utils.data import Dataset

class RefineDataset(Dataset) :
    def __init__(self, metadata):
        super().__init__()
        self.metadata = metadata
        pass
    
    def __len__(self) :
        pass
    
    def __getitem__(self, index):
        return super().__getitem__(index)