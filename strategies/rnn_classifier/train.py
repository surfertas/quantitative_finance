# @author Tasuku Miura

import os
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model import *
from data_set import *

#dtype = torch.FloatTensor
float_dtype = torch.cuda.FloatTensor

#long_dtype = torch.LongTensor
long_dtype = torch.cuda.LongTensor

def train_one_epoch(optimizer, model, loss_fn, data_loader, epoch):
    """ Train one epoch. """
    model.train()
    running_loss = []
    for i, batch in enumerate(data_loader):
        inputs = batch['sequence']
        labels = batch['label']
        
        inputs = Variable(inputs.type(float_dtype))
        labels = Variable(labels.type(long_dtype).squeeze(1))
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        # clip gradients
        
        # Checking grad values
        #for param in model.parameters():
        #    print(param.grad.data)    
        torch.nn.utils.clip_grad_norm(model.parameters(), 0.75)
        
        optimizer.step()
        running_loss.append(loss.data[0])
    
    print('Epoch: {}  loss: {}'.format(
          epoch + 1, np.mean(np.array(running_loss))))

def evaluate(model, loss_fn, data_loader, epoch):
    model.eval()
    loss_sum = 0
    acc_sum = 0
    for i, batch in enumerate(data_loader):
        inputs = batch['sequence']
        labels = batch['label']
        inputs = Variable(inputs.type(float_dtype), volatile=True)
        labels = Variable(labels.type(long_dtype).squeeze(1), volatile=True)
        
        output = model(inputs)
        loss = loss_fn(output, labels)
        loss_sum += loss.data[0]

        predict = output.data.max(1)[1]
        acc = predict.eq(labels.data).cpu().sum()
        acc_sum += acc

    ave_loss = loss_sum/float(len(data_loader))
    ave_acc = acc_sum/float(len(data_loader))

    print('Epoch: {}  valid loss: {:0.6f} accuracy: {:0.6f}'.format(
          epoch + 1, ave_loss, ave_acc))





if __name__=="__main__":
    # Set random seed to 0
    np.random.seed(0)
    torch.manual_seed(0)
    train_data = SequenceDataset('train_data.csv','.',25)
    test_data = SequenceDataset('test_data.csv','.',25)

    train_loader = DataLoader(train_data, batch_size=1, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)

    model = SimpleLSTM(
        input_size=190,
        hidden_size=100,
        batch_size=1,
        steps=25
    ).cuda()
    optimizer =  torch.optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(200):
        train_one_epoch(optimizer, model, loss_fn, train_loader, epoch)
        evaluate(model, loss_fn, test_loader, epoch)

    print('Finished Training')
