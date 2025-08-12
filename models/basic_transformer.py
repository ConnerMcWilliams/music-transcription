import torch
import torch.nn as nn
import torch.nn.functional as f

class BasicTransformerAMT(nn.Module) :
    def __init__(self) :
        super(BasicTransformerAMT, self).__init__()