import torch
import torch.nn.functional as F
import torch.optim as optim
import model
import util


class CLASSIFIER:
    def __init__(self, netM_layer_sizes, lamda_1, _train_X, _train_Y, data_loader, _lr=0.0001, _beta1=0.5, _nepoch=60,
                 _batch_size=100, temperature=0.04, generalized=True, test=True):
        self.train_X = _train_X
        self.train_Y = _train_Y
        self.generalized = generalized
        if test:
            self.test_seen_feature = data_loader.test_seen_feature
            self.test_seen_label = data_loader.test_seen_label
            self.test_unseen_feature = data_loader.test_unseen_feature
            self.test_unseen_label = data_loader.test_unseen_label
            self.seenclasses = data_loader.seenclasses0.cuda()
            self.unseenclasses = data_loader.unseenclasses0.cuda()
            self.att = data_loader.attribute.cuda()
        else:
            self.test_seen_feature = data_loader.val_seen_feature
            self.test_seen_label = util.map_label(data_loader.val_seen_label, data_loader.seenclasses0)
            self.test_unseen_feature = data_loader.val_unseen_feature
            self.test_unseen_label = util.map_label(data_loader.val_unseen_label, data_loader.seenclasses0)
            self.seenclasses = util.map_label(data_loader.seenclasses, data_loader.seenclasses0).cuda()
            self.unseenclasses = util.map_label(data_loader.unseenclasses, data_loader.seenclasses0).cuda()
            self.att = data_loader.attribute[data_loader.seenclasses0].cuda()

        self.batch_size = _batch_size
        self.nepoch = _nepoch
        self.input_dim = _train_X.size(1)
        self.temperature = temperature
        self.lamda_1 = lamda_1
        self.netM = model.Mapping_net(netM_layer_sizes, self.att.size(-1)).cuda() # initialize the mapping net
        self.optimizerM = optim.Adam(self.netM.parameters(), _lr, betas=(_beta1, 0.999))
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizerM, gamma=0.1, step_size=25)
        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.ntrain = self.train_X.size(0)

        if self.generalized:
            self.acc_seen, self.acc_unseen, self.H = self.fit()
        else:
            self.acc = self.fit_zsl()

    def fit(self):
        best_H = torch.zeros(1).cuda()
        best_seen = torch.zeros(1).cuda()
        best_unseen = torch.zeros(1).cuda()
        for epoch in range(self.nepoch):
            self.netM.eval()
            acc_seen = self.val(self.test_seen_feature, self.test_seen_label, self.seenclasses)
            acc_unseen = self.val(self.test_unseen_feature, self.test_unseen_label, self.unseenclasses)
            H = 2 * acc_seen * acc_unseen / (acc_seen + acc_unseen)
            if H > best_H:
                best_seen = acc_seen
                best_unseen = acc_unseen
                best_H = H
            self.netM.train()
            for i in range(0, self.ntrain, self.batch_size):
                self.netM.zero_grad()
                batch_input, batch_label = self.next_batch(self.batch_size)
                batch_input = F.normalize(batch_input, dim=1).cuda()
                batch_label = batch_label.cuda()
                weight = F.normalize(self.netM(self.att), dim=-1)
                loss = self.revised_cross_entropy(batch_input, batch_label, weight, self.temperature, self.lamda_1,
                                                  self.seenclasses, self.unseenclasses)
                loss.backward()
                self.optimizerM.step()
            self.scheduler.step()
        return best_seen, best_unseen, best_H

    def fit_zsl(self):
        best_acc = torch.zeros(1).cuda()
        for epoch in range(self.nepoch):
            self.netM.eval()
            acc = self.val(self.test_unseen_feature, self.test_unseen_label, self.unseenclasses)
            if acc > best_acc:
                best_acc = acc
            self.netM.train()
            for i in range(0, self.ntrain, self.batch_size):
                self.netM.zero_grad()
                batch_input, batch_label = self.next_batch(self.batch_size)
                batch_input = F.normalize(batch_input, dim=1).cuda()
                batch_label = batch_label.cuda()
                weight = F.normalize(self.netM(self.att), dim=-1)
                loss = self.revised_cross_entropy(batch_input, batch_label, weight, self.temperature, self.lamda_1,
                                                  self.seenclasses, self.unseenclasses)
                loss.backward()
                self.optimizerM.step()
            self.scheduler.step()
        return best_acc


    def revised_cross_entropy(self,input,label,weight,tau,lambda_1,seenclasses,unseenclasses):
        logits = input @ weight.t() / tau
        idx_seen = torch.eq(label.reshape(-1, 1), seenclasses).sum(1).nonzero().squeeze(1)
        mask = torch.ones_like(logits)
        mask[idx_seen] = mask[idx_seen].scatter(1, unseenclasses.repeat(idx_seen.size(0), 1), 0)
        _, index = torch.max(logits, dim=1, keepdim=True)
        mask_ = torch.scatter(mask, 1, index, lambda_1)
        mask = (1 - mask) * mask_ + mask
        posi = logits[torch.arange(logits.size(0)).long(),label].view(-1,1)
        logits = logits - posi
        logits = torch.exp(logits)
        logits = mask * logits
        loss = (torch.log(logits.sum(1))).mean()
        return loss
    def val(self, test_X, test_label, target_classes):
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size()).cuda()
        test_label = test_label.cuda()
        target_classes = target_classes.cuda()
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start + self.batch_size)
            with torch.no_grad():
                test_batch =F.normalize(test_X[start:end], dim=-1).cuda()
                weight = F.normalize(self.netM(self.att), dim=-1)
                logits= test_batch@weight.t()
            predicted_label[start:end] = torch.max(logits.data, 1)[1]
            start = end
        acc = self.compute_per_class_acc(test_label, predicted_label, target_classes)
        return acc

    def compute_per_class_acc(self, test_label, predicted_label, target_classes):
        acc_per_class = 0
        test_label = test_label.cpu()
        predicted_label = predicted_label.cpu()
        target_classes = target_classes.cpu()
        for i in target_classes:
            idx = (test_label == i)
            if torch.sum(idx) > 0:
                acc_per_class += torch.sum(test_label[idx] == predicted_label[idx]).float() / torch.sum(idx).float()
        acc_per_class /= target_classes.size(0)
        return acc_per_class

    # select batch samples by randomly drawing batch_size classes
    def next_batch(self, batch_size):
        start = self.index_in_epoch
        # shuffle the data at the first epoch
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
            # shuffle the data
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
            # start next epoch
            start = 0
            self.index_in_epoch = batch_size - rest_num_examples
            end = self.index_in_epoch
            X_new_part = self.train_X[start:end]
            Y_new_part = self.train_Y[start:end]

            if rest_num_examples > 0:
                return torch.cat((X_rest_part, X_new_part), 0), torch.cat((Y_rest_part, Y_new_part), 0)
            else:
                return X_new_part, Y_new_part
        else:
            self.index_in_epoch += batch_size
            end = self.index_in_epoch
            return self.train_X[start:end], self.train_Y[start:end]
