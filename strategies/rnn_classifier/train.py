# @author Tasuku Miura

import os
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model import *
from data_set import *

# dtype = torch.FloatTensor
float_dtype = torch.cuda.FloatTensor

# long_dtype = torch.LongTensor
long_dtype = torch.cuda.LongTensor


def train_one_epoch_cv(optimizer, model, loss_fn, data_loader, epoch):
    k = 1500
    n_tests = (len(data_loader) - k)
    train_loss = []
    loss_sum = 0
    acc_sum = 0

    for i in range(n_tests):
        model.train()
        test = data_loader[k + i]
        train = data_loader[:k + i - 1]
        train_data = zip(train['sequence'], train['label'])
        running_loss = []
        for inputs, labels in train_data:
            inputs = Variable(inputs.type(float_dtype).view(1, -1, 230))
            # labels is not type Torch so need to convert(not sure why its not
            # consistent with other implementations, where labels is already a
            # torch tensor.
            labels = Variable(long_dtype([labels]))

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()

            loss.backward()
            # clip gradients
            torch.nn.utils.clip_grad_norm(model.parameters(), 0.75)
            optimizer.step()

            train_loss.append(loss.data[0])

        # Validate on the hold out sample.
        model.eval()
        v_inputs = test['sequence']
        v_labels = test['label']
        v_inputs = Variable(v_inputs.type(float_dtype).view(1, -1, 230), volatile=True)
        v_labels = Variable(v_labels.type(long_dtype), volatile=True)

        v_output = model(v_inputs)
        v_loss = loss_fn(v_output, v_labels)
        loss_sum += v_loss.data[0]

        v_predict = v_output.data.max(1)[1]
        acc = v_predict.eq(v_labels.data).cpu().sum()
        acc_sum += acc

    ave_loss = loss_sum / float(n_tests)
    ave_acc = acc_sum / float(n_tests)
    print('Epoch: {}  loss: {}'.format(
          epoch + 1, np.mean(np.array(train_loss))))

    print('Epoch: {}  valid loss: {:0.6f} accuracy: {:0.6f}'.format(
          epoch + 1, ave_loss, ave_acc))


def train_one_epoch(optimizer, model, loss_fn, data_loader, epoch):
    """ Train one epoch. """
    model.train()
    running_loss = []
    for i, batch in enumerate(data_loader):
        inputs = batch['sequence']
        labels = batch['label']

        inputs = Variable(inputs.type(float_dtype))
        labels = Variable(labels.type(long_dtype).squeeze(1))
        outputs = model(inputs)
        # Debugging
        if i % 100 == 0:
            print(outputs)

        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()

        loss.backward()
        # clip gradients
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

    ave_loss = loss_sum / float(len(data_loader))
    ave_acc = acc_sum / float(len(data_loader))

    print('Epoch: {}  test loss: {:0.6f} accuracy: {:0.6f}'.format(
          epoch + 1, ave_loss, ave_acc))


if __name__ == "__main__":
    # Set random seed to 0
    np.random.seed(0)
    torch.manual_seed(0)
    train_data = SequenceDataset('train_data_oc.csv', '.', 25)
    test_data = SequenceDataset('test_data_oc.csv', '.', 25)

    train_loader = DataLoader(train_data, batch_size=1, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)

    model = SimpleLSTM(
        input_size=230,
        hidden_size=200,
        n_classes=3,
        batch_size=1,
        steps=25,
        n_layers=2
    ).cuda()
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(200):
#        train_one_epoch(optimizer, model, loss_fn, train_loader, epoch)
        train_one_epoch_cv(optimizer, model, loss_fn, train_data, epoch)

    evaluate(model, loss_fn, test_loader, epoch)

    print('Finished Training')
