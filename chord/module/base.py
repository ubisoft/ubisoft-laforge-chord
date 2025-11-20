import torch
import torch.nn as nn

class Base(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.setup()
    
    def setup(self):
        raise NotImplementedError
    
