# @author Tasuku Miura
# @brief Models used for financial data prediction.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class SimpleLSTM(nn.Module):
    """
    Simple 2 layer LSTM used for classification. 
    """ 
    def __init__(self, input_size, hidden_size, batch_size, steps, n_layers=2, dropout=0.4):
        super(SimpleLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.steps = steps
        self.n_layers = n_layers
        self.dropout = dropout
        
        # Try preprocess with FC layer
        # https://danijar.com/tips-for-training-recurrent-neural-networks/
        self.fc_pre = nn.Linear(self.input_size, self.hidden_size)
        self.rnn = nn.LSTM(
            self.hidden_size,
            self.hidden_size,
            num_layers=self.n_layers,
            dropout=self.dropout,
            batch_first=True
        )
        self.fc = nn.Linear(self.hidden_size, 2)

        
    def forward(self, x, train=True):
        out = self.fc_pre(x)
        # hidden,cell init to 0 as default
        out, _ = self.rnn(out)
        # We want the out of the last step (batch, step, out)
        return F.softmax(self.fc(out[:,-1,:]), dim=1)
