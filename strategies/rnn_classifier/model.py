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
    def __init__(self, input_size, hidden_size, n_classes, batch_size, steps, n_layers=3, dropout=0.5):
        super(SimpleLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_classes = n_classes
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
        self.fc = nn.Linear(self.hidden_size, self.n_classes)

        
    def forward(self, x, train=True):
        out = F.dropout(self.fc_pre(x))
        # hidden,cell init to 0 as default
        out, _ = self.rnn(out)
        out = F.dropout(self.fc(out[:,-1,:]))
        # We want the out of the last step (batch, step, out)
        return F.softmax(out, dim=1)


# TODO: finish MLSTM
# https://discuss.pytorch.org/t/implementation-of-multiplicative-lstm/2328/9
# https://discuss.pytorch.org/t/different-between-lstm-and-lstmcell-function/5657
# https://github.com/pytorch/benchmark/blob/master/benchmarks/models/mlstm.py
def MultiplicativeLSTMCell(input, hidden, w_xm, w_hm, w_ih, w_mh, b_xm=None, b_hm=None, b_ih=None, b_mh=None):
    # w_ih holds W_hx, W_ix, W_ox, W_fx
    # w_mh holds W_hm, W_im, W_om, W_fm

    hx, cx = hidden

    # Key difference:
    m = F.linear(input, w_xm, b_xm) * F.linear(hx, w_hm, b_hm)
    gates = F.linear(input, w_ih, b_ih) + F.linear(m, w_mh, b_mh)

    ingate, forgetgate, hiddengate, outgate = gates.chunk(4, 1)

    ingate = F.sigmoid(ingate)
    outgate = F.sigmoid(outgate)
    forgetgate = F.sigmoid(forgetgate)

    cy = (forgetgate * cx) + (ingate * hiddengate)
    hy = F.tanh(cy * outgate)

    return hy, cy
