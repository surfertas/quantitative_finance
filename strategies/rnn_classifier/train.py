# @author Tasuku Miura

import os
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model import *
from data_set import *


def train_one_epoch(optimizer, model, loss_fn, data_loader, epoch):
    """ Train one epoch. """
    model.train()
    running_loss = []
    for i, batch in enumerate(data_loader):
        inputs = batch['sequence']
        labels = batch['label']
        
        inputs = Variable(inputs.type(torch.FloatTensor))
        labels = Variable(labels.type(torch.LongTensor).squeeze(1))
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        # clip gradients
        torch.nn.utils.clip_grad_norm(model.parameters(), 0.75)
        
        optimizer.step()

        running_loss.append(loss.data[0])
    
    print('Epoch: {}  loss: {}'.format(
          epoch + 1, np.mean(np.array(running_loss))))




if __name__=="__main__":
    train_data = SequenceDataset('train_data.csv','.',25)
    test_data = SequenceDataset('test_data.csv','.',25)

    train_loader = DataLoader(train_data, batch_size=1, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)

    model = SimpleLSTM(input_size=190, hidden_size=100, batch_size=1, steps=25)
    optimizer =  torch.optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(100):
        train_one_epoch(optimizer, model, loss_fn, train_loader, epoch)


    print('Finished Training')
