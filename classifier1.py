
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import util
from sklearn.preprocessing import MinMaxScaler
import sys


class CLASSIFIER:
    def __init__(self, opt, _train_X, _train_Y, data_loader, _lr=0.001, _beta1=0.5, _nepoch=20,
                 _batch_size=100, generalized=True, test=True):
        self.train_X = _train_X
        self.train_Y = _train_Y
        if(test):
            self.test_seen_feature = data_loader.test_seen_feature
            self.test_seen_label = data_loader.test_seen_label
            self.test_unseen_feature = data_loader.test_unseen_feature
            self.test_unseen_label = data_loader.test_unseen_label
            self.seenclasses = data_loader.seenclasses0
            self.unseenclasses = data_loader.unseenclasses0
            if(generalized):
                self.nclass = data_loader.allclasses.size(0)
            else:
                self.nclass = data_loader.unseenclasses0.size(0)

        else:
            self.test_seen_feature = data_loader.val_seen_feature
            self.test_seen_label = util.map_label(data_loader.val_seen_label, data_loader.seenclasses0)
            self.test_unseen_feature = data_loader.val_unseen_feature
            self.test_unseen_label = util.map_label(data_loader.val_unseen_label, data_loader.seenclasses0)
            self.seenclasses = util.map_label(data_loader.seenclasses, data_loader.seenclasses0)
            self.unseenclasses = util.map_label(data_loader.unseenclasses, data_loader.seenclasses0)
            if(generalized):
                self.nclass = data_loader.seenclasses0.size(0)
            else:
                self.nclass = data_loader.unseenclasses.size(0)

        self.batch_size = _batch_size
        self.nepoch = _nepoch
        self.input_dim = _train_X.size(1)
        self.cuda = opt.cuda
        self.device = opt.device
        self.model = LINEAR_LOGSOFTMAX(self.input_dim, self.nclass)
        self.model.apply(util.weights_init)
        self.criterion = nn.NLLLoss()
        self.input = torch.FloatTensor(_batch_size, self.input_dim)
        self.label = torch.LongTensor(_batch_size)

        self.lr = _lr
        self.beta1 = _beta1
        # setup optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=_lr, betas=(_beta1, 0.999))

        if self.cuda:
            self.model.to(self.device)
            self.criterion.to(self.device)
            self.input = self.input.to(self.device)
            self.label = self.label.to(self.device)

        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.ntrain = self.train_X.size()[0]

        if generalized:
            self.acc_seen, self.acc_unseen, self.H = self.fit()
        else:
            self.acc = self.fit_zsl()


    def fit_zsl(self):
        best_acc = 0
        mean_loss = 0
        last_loss_epoch = 1e8
        for epoch in range(self.nepoch):
            for i in range(0, self.ntrain, self.batch_size):
                self.model.zero_grad()
                batch_input, batch_label = self.next_batch(self.batch_size)
                self.input.copy_(batch_input)
                self.label.copy_(batch_label)
                inputv = Variable(self.input)
                labelv = Variable(self.label)
                output = self.model(inputv)
                loss = self.criterion(output, labelv)
                mean_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            acc = self.val(self.test_unseen_feature, self.test_unseen_label, self.unseenclasses)
            if acc > best_acc:
                best_acc = acc
        return best_acc

    def fit(self):
        best_H = 0
        best_seen = 0
        best_unseen = 0
        for epoch in range(self.nepoch):
            for i in range(0, self.ntrain, self.batch_size):
                self.model.zero_grad()
                batch_input, batch_label = self.next_batch(self.batch_size)
                self.input.copy_(batch_input)
                self.label.copy_(batch_label)

                inputv = Variable(self.input)
                labelv = Variable(self.label)
                output = self.model(inputv)
                loss = self.criterion(output, labelv)
                loss.backward()
                self.optimizer.step()
            acc_seen = 0
            acc_unseen = 0
            acc_seen = self.val_gzsl(self.test_seen_feature, self.test_seen_label, self.seenclasses)
            acc_unseen = self.val_gzsl(self.test_unseen_feature, self.test_unseen_label, self.unseenclasses)
            H = 2*acc_seen*acc_unseen / (acc_seen+acc_unseen)
            if H > best_H or (H == best_H and acc_seen > best_seen):
                best_seen = acc_seen
                best_unseen = acc_unseen
                best_H = H
        return best_seen, best_unseen, best_H

    def next_batch(self, batch_size):
        start = self.index_in_epoch
        if self.epochs_completed == 0 and start == 0:
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
        # the last batch
        if start + batch_size > self.ntrain:
            self.epochs_completed += 1
            rest_num_examples = self.ntrain - start
            if rest_num_examples > 0:
                X_rest_part = self.train_X[start:self.ntrain]
                Y_rest_part = self.train_Y[start:self.ntrain]
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
            start = 0
            self.index_in_epoch = batch_size - rest_num_examples
            end = self.index_in_epoch
            X_new_part = self.train_X[start:end]
            Y_new_part = self.train_Y[start:end]
            #print(start, end)
            if rest_num_examples > 0:
                return torch.cat((X_rest_part, X_new_part), 0) , torch.cat((Y_rest_part, Y_new_part), 0)
            else:
                return X_new_part, Y_new_part
        else:
            self.index_in_epoch += batch_size
            end = self.index_in_epoch
            return self.train_X[start:end], self.train_Y[start:end]


    def val_gzsl(self, test_X, test_label, target_classes):
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start+self.batch_size)
            with torch.no_grad():
                if self.cuda:
                    output = self.model(Variable(test_X[start:end].to(self.device)))
                else:
                    output = self.model(Variable(test_X[start:end]))
            _, predicted_label[start:end] = torch.max(output.data, 1)
            start = end

        acc = self.compute_per_class_acc_gzsl(test_label, predicted_label, target_classes)
        return acc


    def val(self, test_X, test_label, target_classes):
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start+self.batch_size)
            with torch.no_grad():
                if self.cuda:
                    output = self.model(Variable(test_X[start:end].to(self.device)))
                else:
                    output = self.model(Variable(test_X[start:end]))
            _, predicted_label[start:end] = torch.max(output.data, 1)
            start = end
        acc = self.compute_per_class_acc(util.map_label(test_label, target_classes), predicted_label, target_classes.size(0))
        return acc

    def compute_per_class_acc_gzsl(self, test_label, predicted_label, target_classes):
        acc_per_class = 0
        for i in target_classes:
            idx = (test_label == i)
            if torch.sum(idx) > 0:
                acc_per_class += torch.sum(test_label[idx]==predicted_label[idx]) / torch.sum(idx)
        acc_per_class /= target_classes.size(0)
        return acc_per_class

    def compute_per_class_acc(self, test_label, predicted_label, nclass):
        acc_per_class = torch.FloatTensor(nclass).fill_(0)
        for i in range(nclass):
            idx = (test_label == i)
            acc_per_class[i] = torch.sum(test_label[idx]==predicted_label[idx]) / torch.sum(idx)
        return acc_per_class.mean()

class LINEAR_LOGSOFTMAX(nn.Module):
    def __init__(self, x_dim, s_dim, layers=''):
        super(LINEAR_LOGSOFTMAX, self).__init__()
        self.x_dim = x_dim
        self.s_dim = s_dim
        layers = layers.split()
        fcn_layers = []
        if len(layers) == 0:
            fcn_layers.append(nn.Linear(x_dim, s_dim))
            fcn_layers.append(nn.LogSoftmax(dim=1))
        else:
            for i in range(len(layers)):
                pre_hidden = int(layers[i - 1])
                num_hidden = int(layers[i])
                if i == 0:
                    fcn_layers.append(nn.Linear(x_dim, num_hidden))
                    fcn_layers.append(nn.ReLU())
                else:
                    fcn_layers.append(nn.Linear(pre_hidden, num_hidden))
                    fcn_layers.append(nn.ReLU())

                if i == len(layers) - 1:
                    fcn_layers.append(nn.Linear(num_hidden, s_dim))
                    fcn_layers.append(nn.LogSoftmax(dim=1))

        self.FCN = nn.Sequential(*fcn_layers)
    def forward(self, x):
        o = self.FCN(x)
        return o
