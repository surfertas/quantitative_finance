# @author Tasuku Miura

import os
import pickle
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model import *
from data_set import *


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
    # Training settings
    parser = argparse.ArgumentParser(description='RNN Classifier')
    parser.add_argument('--train-data', type=str, default='train_data_oc.csv',
                        help='filename containing train data')
    parser.add_argument('--test-data', type=str, default='test_data_oc.csv',
                        help='filename containing test data')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                        help='input batch size for testing (default: 32)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 0)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--steps', type=int, default=25, metavar='N',
                        help='how many time steps in sequence (if 0, use feedforward)')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        float_dtype = torch.cuda.FloatTensor
        long_dtype = torch.cuda.LongTensor
    else:
        dtype = torch.FloatTensor
        long_dtype = torch.LongTensor

    # Set random seed to 0
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    current_dir = os.path.dirname(os.path.realpath(__file__))
    train_data_seq = SequenceDataset(args.train_data, current_dir, args.steps)
    test_data_seq = SequenceDataset(args.test_data, current_dir, args.steps)

    # train_data = Dataset('train_data_oc.csv', '.')
    # test_data = Dataset('test_data_oc.csv', '.')

    train_loader = DataLoader(train_data_seq, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_data_seq, batch_size=32, shuffle=False, num_workers=4)

    model_lstm = SimpleLSTM(
        input_size=230,
        hidden_size=100,
        n_classes=3,
        batch_size=1,
        steps=args.steps,
        n_layers=1
    ).cuda()

    # model_nn = NN(
    #    input_size=230,
    #    hidden_size=50,
    #    n_classes=3
    #).cuda()

    optimizer = torch.optim.Adam(model_lstm.parameters(), args.lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        train_one_epoch(optimizer, model_lstm, loss_fn, train_loader, epoch)
#        train_one_epoch_cv(optimizer, model, loss_fn, train_data, epoch)

    evaluate(model, loss_fn, test_loader, epoch)

    print('Finished Training')
