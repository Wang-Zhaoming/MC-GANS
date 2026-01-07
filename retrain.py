from __future__ import print_function

import argparse
import importlib
import os
import random

import numpy as np
import torch
import time

import torch.optim as optim

import classifier1
import model_retrain
import util
import classifier
import model
from config import parser, str2bool

# 管理命令行参数
parser = argparse.ArgumentParser(parents=[parser])
parser.add_argument('--preprocessing', action='store_true', default=True, help='enbale MinMaxScaler on visual features')
parser.add_argument('--nepoch', type=int, default=500, help='number of epochs to train for GANs')
parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate to train GANs ')
parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
parser.add_argument('--original', type=str2bool, default=False, help='Whether new genes are introduced during evolution')

opt = parser.parse_args()
print(opt)

# set random seed
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)

model_path = opt.geo_dir + "{}_params.pth".format(opt.dataset)

# load data
data = util.DATA_LOADER(opt)
print("# of training samples: ", data.ntrain)

if opt.classifier_module_name == 'classifier':
    opt.syn_num = int(data.test_unseen_label.shape[0]/data.unseenclasses0.shape[0])
else:
    opt.syn_num = int(data.test_unseen_label.shape[0] / data.unseenclasses0.shape[0])*10
print("syn_num=%d" % opt.syn_num)

if opt.original == True:
    netG = model.Generator(opt.netG_layer_sizes, opt.nz, opt.attSize)
    netD = model.Discriminator(opt.resSize, opt.attSize, opt.netD_layer_sizes)
elif opt.fix_alg == "D":
    netD = model.Discriminator(opt.resSize, opt.attSize, opt.netD_layer_sizes)
    loaded_data = np.load(opt.geo_dir + "/{}_G.npz".format(opt.dataset))
    genotype_G = loaded_data['best_genotype']
    netG = model_retrain.NetworkRetrain(opt, 'g', genotype_G)
    print(genotype_G)
else:
    loaded_data = np.load(opt.geo_dir + "/{}_G.npz".format(opt.dataset))
    genotype_G = loaded_data['best_genotype']
    netG = model_retrain.NetworkRetrain(opt, 'g', genotype_G)
    loaded_data = np.load(opt.geo_dir + "/{}_D.npz".format(opt.dataset))
    genotype_D = loaded_data['best_genotype']
    netD = model_retrain.NetworkRetrain(opt, 'd', genotype_D)
    print(genotype_G)
    print(genotype_D)

best_gzsl_acc = 0
best_zsl_acc = 0
best_epoch = 0
best_cls_gzsl = 0
best_cls_zsl = 0
input_res = torch.FloatTensor(opt.batch_size, opt.resSize)  # 64 2048
input_att = torch.FloatTensor(opt.batch_size, opt.attSize)  # 64 1024
input_attn = torch.FloatTensor(opt.batch_size, opt.attSize)  # 64 1024
noise = torch.FloatTensor(opt.batch_size, opt.nz)
one = torch.tensor(1.)
mone = one * -1
input_label = torch.LongTensor(opt.batch_size)

classifier_module = importlib.import_module(opt.classifier_module_name)
ClassifierClass = getattr(classifier_module, 'CLASSIFIER')

if opt.cuda:
    netD.to(opt.device)
    netG.to(opt.device)
    # netM.to(opt.device)
    input_res = input_res.to(opt.device)
    input_attn = input_attn.to(opt.device)
    noise, input_att = noise.to(opt.device), input_att.to(opt.device)
    one = one.to(opt.device)
    mone = mone.to(opt.device)
    input_label = input_label.to(opt.device)


def sample(batch_size=opt.batch_size):
    batch_feature, batch_label, batch_att = data.next_batch_trainval(batch_size)
    input_res.copy_(batch_feature)
    input_att.copy_(batch_att)
    input_label.copy_(batch_label)


optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
schedulerG = optim.lr_scheduler.CosineAnnealingLR(optimizerG, float(opt.nepoch),
                                                  eta_min=0.1 * opt.lr * opt.slow)
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
schedulerD = optim.lr_scheduler.CosineAnnealingLR(optimizerD, float(opt.nepoch),
                                                  eta_min=0.1 * opt.lr * opt.slow)

fake_feature = None
fake_label = None
gzsl_unseen = []
gzsl_seen = []
gzsl_H = []
zsl_unseen = []

time_str = time.strftime("%m-%d-%H-%M", time.localtime())

#############################
#generator training phase
#############################
for epoch in range(opt.nepoch):
    schedulerG.step()
    schedulerD.step()
    for i in range(0, data.ntrain, opt.batch_size):
        # (1) Update netD
        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG_search update
        for p in netG.parameters():  # reset requires_grad
            p.requires_grad = False  # they are set to Ture below in netG_search update

        iter_d = 0
        while True:
            netD.zero_grad()
            sample(opt.batch_size)
            criticD_real = netD(input_res, input_att).mean()

            noise.normal_(0, 1)  # noise for attribute augmentation
            as_fake = opt.att_std * noise + input_att  # augmented attribute
            noise.normal_(0, 1)  # noise for generation
            fake = netG(noise, as_fake).detach()
            criticD_fake = netD(fake, input_att).mean()

            gradient_penalty = util.calc_gradient_penalty(opt, netD, input_res, fake.data,
                                                          input_att)  # gradient penalty

            W_D = (criticD_real - criticD_fake).item()
            D_cost = criticD_fake - criticD_real + gradient_penalty  # cost of the discriminator
            D_cost.backward()
            optimizerD.step()
            D_cost = D_cost.item()

            if opt.dynamic_D:
                if (W_D >= opt.w_up and D_cost < 0) or iter_d >= 20:
                    break
            elif iter_d >= opt.critic_iter:
                break

            iter_d += 1

        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = False
        for p in netG.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False above in netD update
        netG.zero_grad()
        fake = netG(noise, as_fake)
        G_cost = -netD(fake, input_att).mean()
        G_cost.backward()
        optimizerG.step()
        G_cost = G_cost.item()  # cost of the generator

    print('[%d/%d]iter_d:%d Loss_D: %.4f Loss_G: %.4f W_d: %.4f'
          % (epoch, opt.nepoch, iter_d, D_cost, G_cost, W_D))
    #############################
    # classifier training phase
    #############################
    netG.eval()
    syn_feature, syn_label = util.generate_syn_feature(opt, netG, data.unseenclasses0, data.attribute,
                                                       opt.syn_num)  # generate pseudo unseen samples
    if (opt.gzsl):
        train_X = torch.cat((data.trainval_feature, syn_feature), dim=0)
        train_Y = torch.cat((data.trainval_label, syn_label), dim=0)

        cls_gzsl = ClassifierClass(opt.netM_layer_sizes, opt.lambda_1, train_X, train_Y, data, opt.lr_classifier,
                                   opt.beta1, opt.nepoch_classifier, opt.batch_size,
                                   opt.temperature)
        gzsl_unseen.append(cls_gzsl.acc_unseen)
        gzsl_seen.append(cls_gzsl.acc_seen)
        gzsl_H.append(cls_gzsl.H)
        print('unseen=%.4f, seen=%.4f, h=%.4f' % (cls_gzsl.acc_unseen, cls_gzsl.acc_seen, cls_gzsl.H))

    cls_zsl = classifier1.CLASSIFIER(opt, syn_feature, util.map_label(syn_label, data.unseenclasses0), data,
                                     opt.classifier_lr, 0.5, 50, opt.syn_num, False, True)
    zsl_unseen.append(cls_zsl.acc)
    print('unseen class accuracy= ', cls_zsl.acc)

    if (opt.gzsl):
        if best_gzsl_acc < cls_gzsl.H:
            best_gzsl_acc = cls_gzsl.H
            best_epoch = epoch
            best_cls_gzsl = cls_gzsl
            best_cls_zsl = cls_zsl
            fake_feature = syn_feature
            fake_label = syn_label
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(netG.state_dict(), model_path)
    else:
        if best_zsl_acc < cls_zsl.acc:
            best_acc = cls_gzsl.acc
            best_epoch = epoch
            best_cls = cls_zsl
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(netG.state_dict(), model_path)

    if epoch - best_epoch >= 100:
        break
if opt.original:
    file_path = opt.geo_dir + "original_output.txt".format(opt.dataset)
else:
    file_path = opt.geo_dir + "output.txt".format(opt.dataset)

time_end = time.strftime("%m-%d-%H-%M", time.localtime())

with open(file_path, 'a') as file:
    file.write("____________________retrain_____________________\n")
    file.write(time_str + "->" + time_end + "\n")
    file.write("manualSeed={}\n".format(opt.manualSeed))
    if (opt.gzsl):
        file.write("best_epoch={} acc_unseen={} acc_seen={} H={}\n".format(best_epoch, best_cls_gzsl.acc_unseen,
                                                                           best_cls_gzsl.acc_seen, best_cls_gzsl.H))
    file.write("unseen class accuracy={}\n".format(best_cls_zsl.acc))
    for key, value in vars(opt).items():
        file.write(f'{key}: {value}\t')
    file.write('\n____________________________________________\n')

if (opt.gzsl):
    print('unseen=%.4f, seen=%.4f, h=%.4f' % (best_cls_gzsl.acc_unseen, best_cls_gzsl.acc_seen, best_cls_gzsl.H))
print('unseen class accuracy= ', best_cls_zsl.acc)

if opt.dynamic_D:
    opt.geo_dir = opt.geo_dir + "yes/"
else:
    opt.geo_dir = opt.geo_dir + "no/"

print(opt.geo_dir)

os.makedirs(opt.geo_dir, exist_ok=True)

if opt.save_data:
    if opt.original:
        feature_path = os.path.join(opt.geo_dir, "{}_original_features_{}.txt".format(opt.dataset, opt.manualSeed))
        label_path = os.path.join(opt.geo_dir, "{}_original_labels.txt_{}".format(opt.dataset, opt.manualSeed))
        np.savetxt(feature_path, fake_feature, fmt='%.6f')
        np.savetxt(label_path, fake_label, fmt='%d')

        np.savez(opt.geo_dir + "/{}_original_fakedata_{}".format(opt.dataset, opt.manualSeed),
                 fake_feature=fake_feature, fake_label=fake_label)
        np.savez(opt.geo_dir + "/{}_original_retraindata_{}".format(opt.dataset, opt.manualSeed),
                 Epoch=range(1, len(zsl_unseen) + 1), zsl_unseen=zsl_unseen,
                 gzsl_unseen=gzsl_unseen, gzsl_seen=gzsl_seen, gzsl_H=gzsl_H)
    else:
        feature_path = os.path.join(opt.geo_dir, "{}_our_features_{}.txt".format(opt.dataset, opt.manualSeed))
        label_path = os.path.join(opt.geo_dir, "{}_our_labels_{}.txt".format(opt.dataset, opt.manualSeed))
        np.savetxt(feature_path, fake_feature, fmt='%.6f')
        np.savetxt(label_path, fake_label, fmt='%d')

        np.savez(opt.geo_dir + "/{}_retrain_fakedata_{}".format(opt.dataset, opt.manualSeed),
                 fake_feature=fake_feature, fake_label=fake_label)
        np.savez(opt.geo_dir + "/{}_retraindata_{}".format(opt.dataset, opt.manualSeed),
                 Epoch=range(1, len(zsl_unseen) + 1), zsl_unseen=zsl_unseen,
                 gzsl_unseen=gzsl_unseen, gzsl_seen=gzsl_seen, gzsl_H=gzsl_H)
