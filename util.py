import numpy as np
import scipy.io as sio
import torch
import torch.autograd as autograd
from sklearn import preprocessing
import torch.nn.functional as F


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label == classes[i]] = i
    return mapped_label


def generate_syn_feature(opt, netG, classes, att, num, genotype=None):
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass * num, opt.resSize)
    syn_label = torch.LongTensor(nclass * num)
    syn_att = torch.FloatTensor(num, opt.attSize)
    syn_noise = torch.FloatTensor(num, opt.nz)
    if opt.cuda:
        syn_att = syn_att.to(opt.device)
        syn_noise = syn_noise.to(opt.device)

    for i in range(nclass):
        iclass = classes[i]
        iclass_att = att[iclass]
        syn_att.copy_(iclass_att.repeat(num, 1))
        syn_noise.normal_(0, 1)
        with torch.no_grad():
            if genotype is None:
                output = netG(syn_noise, syn_att)
            else:
                output = netG(syn_noise, syn_att, genotype)
        syn_feature.narrow(0, i * num, num).copy_(output.data.cpu())
        syn_label.narrow(0, i * num, num).fill_(iclass)
    return syn_feature, syn_label


def calc_gradient_penalty(opt, netD, real_data, fake_data, input_att, genotype=None):
    alpha = torch.rand(opt.batch_size, 1)
    alpha = alpha.expand(real_data.size())
    if opt.cuda:
        alpha = alpha.to(opt.device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    if opt.cuda:
        interpolates = interpolates.to(opt.device)
    interpolates.requires_grad_(True)

    if genotype is None:
        disc_interpolates = netD(interpolates, input_att)
    else:
        disc_interpolates = netD(interpolates, input_att, genotype)

    ones = torch.ones(disc_interpolates.size())
    if opt.cuda:
        ones = ones.to(opt.device)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda_0
    return gradient_penalty


class DATA_LOADER(object):
    def __init__(self, opt):
        self.read_matdataset(opt)
        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.opt = opt

    def read_matdataset(self, opt):
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat")
        feature = matcontent['features'].T
        label = matcontent['labels'].astype(int).squeeze() - 1
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + "_splits.mat")
        # numpy array index starts from 0, matlab starts from 1
        # mx = 5
        trainval_loc = matcontent['trainval_loc'].squeeze() - 1
        train_loc = matcontent['train_loc'].squeeze() - 1
        val_unseen_loc = matcontent['val_loc'].squeeze() - 1

        total_train_samples = len(train_loc)
        train_size = int(0.8 * total_train_samples)

        np.random.shuffle(train_loc)

        val_seen_loc = train_loc[train_size:]
        train_loc = train_loc[:train_size]

        test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1
        self.attribute = torch.from_numpy(matcontent['att'].T).float()
        if opt.preprocessing:
            scaler = preprocessing.MinMaxScaler()
            _trainval_feature = scaler.fit_transform(feature[trainval_loc])
            _train_feature = scaler.fit_transform(feature[train_loc])
            _val_seen_feature = scaler.fit_transform(feature[val_seen_loc])
            _val_unseen_feature = scaler.fit_transform(feature[val_unseen_loc])
            _test_seen_feature = scaler.transform(feature[test_seen_loc])
            _test_unseen_feature = scaler.transform(feature[test_unseen_loc])
            self.trainval_feature = torch.from_numpy(_trainval_feature).float()
            mx = self.trainval_feature.max()
            self.trainval_feature.mul_(1 / mx)
            self.trainval_label = torch.from_numpy(label[trainval_loc]).long()

            self.train_feature = torch.from_numpy(_train_feature).float()
            self.train_feature.mul_(1 / mx)
            self.train_label = torch.from_numpy(label[train_loc]).long()

            self.val_unseen_feature = torch.from_numpy(_val_unseen_feature).float()
            self.val_unseen_feature.mul_(1 / mx)
            self.val_unseen_label = torch.from_numpy(label[val_unseen_loc]).long()

            self.val_seen_feature = torch.from_numpy(_val_seen_feature).float()
            self.val_seen_feature.mul_(1 / mx)
            self.val_seen_label = torch.from_numpy(label[val_seen_loc]).long()
            self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
            self.test_unseen_feature.mul_(1 / mx)
            self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long()

            self.test_seen_feature = torch.from_numpy(_test_seen_feature).float()
            self.test_seen_feature.mul_(1 / mx)
            self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()
        else:
            self.trainval_feature = torch.from_numpy(feature[trainval_loc]).float()
            self.trainval_label = torch.from_numpy(label[trainval_loc]).long()

            self.train_feature = torch.from_numpy(feature[train_loc]).float()
            self.train_label = torch.from_numpy(label[train_loc]).long()

            self.val_seen_feature = torch.from_numpy(feature[val_seen_loc]).float()
            self.val_seen_label = torch.from_numpy(label[val_seen_loc]).long()

            self.val_unseen_feature = torch.from_numpy(feature[val_unseen_loc]).float()
            self.val_unseen_label = torch.from_numpy(label[val_unseen_loc]).long()

            self.test_unseen_feature = torch.from_numpy(feature[test_unseen_loc]).float()
            self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long()

            self.test_seen_feature = torch.from_numpy(feature[test_seen_loc]).float()
            self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()

        self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.val_unseen_label.numpy()))
        self.seenclasses0 = torch.from_numpy(np.unique(self.trainval_label.numpy()))
        self.unseenclasses0 = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))
        self.ntrain = self.train_feature.size()[0]
        self.ntrainval = self.trainval_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntrainval_class = self.seenclasses0.size(0)
        self.ntest_class = self.unseenclasses0.size(0)
        self.allclasses = torch.arange(0, self.ntrainval_class + self.ntest_class).long()

    def get_val(self):
        feature = self.val_unseen_feature
        label = self.val_unseen_label
        att = self.attribute[label]
        return feature, label, att

    def get_test(self):
        feature = self.test_unseen_feature
        label = self.test_unseen_label
        att = self.attribute[label]
        return feature, label, att

    def get_seen_val(self):
        idx = torch.randperm(self.val_seen_feature.size()[0])
        batch_feature = self.val_seen_feature[idx]
        batch_label = self.val_seen_label[idx]
        batch_att = self.attribute[batch_label]
        return batch_feature, batch_label, batch_att

    def get_seen_test(self):
        idx = torch.randperm(self.test_seen_feature.size()[0])
        batch_feature = self.test_seen_feature[idx]
        batch_label = self.test_seen_label[idx]
        batch_att = self.attribute[batch_label]
        return batch_feature, batch_label, batch_att

    def next_batch_trainval(self, batch_size):
        idx = torch.randperm(self.ntrainval)[0:batch_size]
        batch_feature = self.trainval_feature[idx]
        batch_label = self.trainval_label[idx]
        batch_att = self.attribute[batch_label]
        return batch_feature, batch_label, batch_att

    def next_batch_train(self, batch_size):
        idx = torch.randperm(self.ntrain)[0:batch_size]
        batch_feature = self.train_feature[idx]
        batch_label = self.train_label[idx]
        batch_att = self.attribute[batch_label]
        return batch_feature, batch_label, batch_att

    def get_class_names_by_ids(self, class_ids):
        file_path = self.opt.dataroot + "/" + self.opt.dataset + "/" + "allclasses.txt"
        class_names = {}
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split('.', 1)
                if len(parts) == 2:
                    class_id, class_name = parts
                    class_names[int(class_id)] = class_name
        return [class_names.get(class_id) for class_id in class_ids]