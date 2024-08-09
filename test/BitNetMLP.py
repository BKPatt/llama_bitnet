import torch
import torch.nn as nn
import torch.nn.functional as F

from BitNetB158Config import BitNetB158Config
from QuantizedLinear import QuantizedLinear

class BitNetMLP(nn.Module):
    def __init__(self, config: BitNetB158Config):
        super(BitNetMLP, self).__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.fc1 = QuantizedLinear(self.hidden_size, self.intermediate_size)
        self.fc2 = QuantizedLinear(self.intermediate_size, self.hidden_size)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = F.relu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.fc2(hidden_states)
        
        return hidden_states
