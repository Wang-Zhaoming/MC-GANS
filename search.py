from __future__ import print_function

import argparse
import importlib
import os
import time

# import pandas as pd

import numpy as np
import torch
import random

import torch.optim as optim
from torch import nn
from torch.autograd import Variable
from tqdm import tqdm

import classifier0
import classifier1
import model_retrain
import util
import classifier
import model
from config import parser, str2bool
from search_algs import GanAlgorithm, search_evol_arch1, pop_evolution, get_pop_similarity, get_similarity


parser = argparse.ArgumentParser(parents=[parser])
parser.add_argument('--preprocessing', action='store_true', default=True, help='enbale MinMaxScaler on visual features')
parser.add_argument('--nepoch', type=int, default=500, help='number of epochs to train for GANs')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate to train GANs ')
parser.add_argument('--warmup_nepoch', type=int, default=5, help='number of epochs to warm up')
parser.add_argument('--critic_iter', type=int, default=8, help='critic iteration, following WGAN-GP')
parser.add_argument('--num_individual', type=int, default=50, help='# num of population')
parser.add_argument('--num_initial_input', type=int, default=3, help='# num of population')
parser.add_argument('--num_train', type=int, default=1, help='# num of train_times')
parser.add_argument('--cls_weight', type=float, default=0.2, help='0.2weight of the classification loss')
parser.add_argument('--regular_weight', type=float, default=0.0, help='weight regular')
parser.add_argument('--pretrain_classifier', default='', help="path to pretrain classifier (to continue training)")
parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
parser.add_argument('--pop_size', type=int, default=5, help='learning rate to train softmax classifier')

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

# load data
data = util.DATA_LOADER(opt)
print("# of training samples: ", data.ntrain)

if opt.classifier_module_name == 'classifier':
    opt.syn_num = int(data.test_unseen_label.shape[0]/data.unseenclasses0.shape[0])
else:
    opt.syn_num = int(data.test_unseen_label.shape[0] / data.unseenclasses0.shape[0])*10
print("syn_num=%d" % opt.syn_num)

if opt.fix_alg is None:
    netG_search = model.MLP_search(opt, 'g')
    netD_search = model.MLP_search(opt, 'd')
    G_alg = GanAlgorithm(opt)
    D_alg = GanAlgorithm(opt)
    genotypes_G = np.stack([G_alg.search() for i in range(opt.num_individual)], axis=0)
    genotypes_D = np.stack([D_alg.search() for i in range(opt.num_individual)], axis=0)
    G_init = G_alg.genotype_init
    D_init = D_alg.genotype_init
elif opt.fix_alg == "G":
    genotypes_G = []
    loaded_data = np.load(opt.geo_dir + "/{}_G.npz".format(opt.dataset))
    genotype_G = loaded_data['best_genotype']
    genotypes_G.append(genotype_G)
    netG_search = model_retrain.NetworkRetrain(opt, 'g', genotype_G)

    netD_search = model.MLP_search(opt, 'd')
    D_alg = GanAlgorithm(opt)
    genotypes_D = np.stack([D_alg.search() for i in range(opt.num_individual)], axis=0)
    D_init = D_alg.genotype_init
    G_init = None
elif opt.fix_alg == "D":
    genotypes_D = []
    D_alg = GanAlgorithm(opt)
    genotype_D = D_alg.sample()
    genotypes_D.append(genotype_D)
    netD_search = model.Discriminator(opt.resSize, opt.attSize, opt.netD_layer_sizes)
    netG_search = model.MLP_search(opt, 'g')
    G_alg = GanAlgorithm(opt)
    genotypes_G = np.stack([G_alg.search() for i in range(opt.num_individual)], axis=0)
    G_init = G_alg.genotype_init
    D_init = None

best_epoch = 0
best_acc = 0

best_Ggenotype = None
best_Dgenotype = None

input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
input_att = torch.FloatTensor(opt.batch_size, opt.attSize)
input_attn = torch.FloatTensor(opt.batch_size, opt.attSize)
noise = torch.FloatTensor(opt.batch_size, opt.nz)
input_label = torch.LongTensor(opt.batch_size)

classifier_module = importlib.import_module(opt.classifier_module_name)
ClassifierClass = getattr(classifier_module, 'CLASSIFIER')
cls_criterion = nn.CrossEntropyLoss()

if opt.cuda:
    netG_search.to(opt.device)
    netD_search.to(opt.device)
    input_res = input_res.to(opt.device)
    input_attn = input_attn.to(opt.device)
    noise, input_att = noise.to(opt.device), input_att.to(opt.device)
    input_label = input_label.to(opt.device)
    cls_criterion.to(opt.device)

def sample(batch_size=opt.batch_size):
    if opt.test_data:
        batch_feature, batch_label, batch_att = data.next_batch_trainval(batch_size)
    else:
        batch_feature, batch_label, batch_att = data.next_batch_train(batch_size)
    input_res.copy_(batch_feature).to(opt.device)
    input_att.copy_(batch_att).to(opt.device)

def search_evol_Garch(gen_net, genotypes, dis_net, genotype_D, G_alg):
    num = genotypes.shape[0]
    d_values, cls_values, acc_values, para_values = np.zeros(num), np.zeros(num), np.zeros(num), np.zeros(
        num)
    for idx, genotype_G in enumerate(genotypes):
        d_value, cls_value, para_value, acc_value = validateG(gen_net, dis_net, genotype_G, genotype_D)
        d_values[idx] = d_value + 0 * cls_value + opt.regular_weight * para_value + 0 * acc_value
        cls_values[idx] = cls_value
        acc_values[idx] = acc_value
        para_values[idx] = para_value
    keep = np.argmin(d_values)
    print("best_Loss_D:{} best_cls:{} best_acc:{}".format(d_values[keep], cls_values[keep], acc_values[keep]))
    genotypes = evol_arch(genotypes, d_values, G_alg)
    return genotypes


def validateG(gen_net, dis_net, genotype_G, genotype_D):
    if opt.test_data:
        all_class = data.allclasses
        unseen=data.unseenclasses0
        _, input_label_val, input_att_val = data.get_test()
    else:
        all_class =data.seenclasses0
        unseen = data.unseenclasses
        _, input_label_val, input_att_val = data.get_val()


    input_labelv_val = Variable(util.map_label(input_label_val, all_class)).to(opt.device)
    input_attv_val = Variable(input_att_val).to(opt.device)
    noise = torch.FloatTensor(input_labelv_val.size(0), opt.nz).to(opt.device)
    noise.normal_(0, 1)

    with torch.no_grad():
        fake = gen_net(noise, input_attv_val, genotype_G).detach()
        criticD_fake = dis_net(fake, input_attv_val, genotype_D)

    criticD_fake_all = criticD_fake.mean().item()

    clsloss_all = 0
    acc = 0

    para_size = (genotype_G.shape[0] - genotype_G[:, -1].sum()) / genotype_G.shape[0]  #连接边比例
    return -criticD_fake_all, clsloss_all, para_size, -acc


def search_evol_Darch(gen_net, genotype_G, dis_net, genotypes, D_alg):
    num = genotypes.shape[0]
    fit_values, para_values, Wd_values = np.zeros(num), np.zeros(num), np.zeros(num)

    if opt.test_data:
        data_len = len(data.test_seen_label)
    else:
        data_len = len(data.val_seen_label)
    noise_val = torch.FloatTensor(data_len, opt.nz).to(opt.device)  # wzm
    noise_val.normal_(0, 1)

    for idx, genotype_D in enumerate(genotypes):
        Wd_value, para_value = validateD(gen_net, dis_net, genotype_G, genotype_D, noise_val)
        fit_values[idx] = Wd_value + opt.regular_weight*para_value
        Wd_values[idx] = Wd_value
        para_values[idx] = para_value
    keep = np.argmin(fit_values)
    print("best Wasserstein_dist:{}".format(fit_values[keep]))
    genotypes = evol_arch(genotypes, fit_values, D_alg)
    return genotypes


def validateD(gen_net, dis_net, genotype_G, genotype_D, noise_valv):
    if opt.test_data:
        input_res_val, _, input_att_val = data.get_seen_test()
    else:
        input_res_val, _, input_att_val = data.get_seen_val()

    input_resv_val = Variable(input_res_val).to(opt.device)
    input_attv_val = Variable(input_att_val).to(opt.device)
    with torch.no_grad():
        fake = gen_net(noise_valv, input_attv_val, genotype_G)
        criticD_real = dis_net(input_resv_val, input_attv_val, genotype_D)
        criticD_fake = dis_net(fake.detach(), input_attv_val, genotype_D)
    eval_loss_D = -criticD_real + criticD_fake
    para_size = (genotype_D.shape[0]-genotype_D[:,-1].sum())/genotype_D.shape[0]

    return eval_loss_D.mean().item(), para_size


def evol_arch(genotypes, values, gan_alg):
    keep_N = opt.num_individual
    if opt.pop_evolution:
        evol_genotypes = pop_evolution(genotypes, values, opt.pop_size)
    else:
        keep = np.argsort(values)[0:keep_N]
        evol_genotypes = genotypes[keep]

    pop_similarity.append(get_pop_similarity(evol_genotypes))
    global bef_gene
    if bef_gene is not None:
        gene_similarity.append(get_similarity(bef_gene, evol_genotypes[0]))
    bef_gene = evol_genotypes[0]
    print(evol_genotypes.shape)
    gan_alg.update(evol_genotypes)
    return evol_genotypes


optimizerG = optim.Adam(netG_search.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
schedulerG = optim.lr_scheduler.CosineAnnealingLR(optimizerG, float(opt.nepoch),
                                                   eta_min=1e-1 * opt.lr * opt.slow)
optimizerD = optim.Adam(netD_search.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
schedulerD = optim.lr_scheduler.CosineAnnealingLR(optimizerD, float(opt.nepoch),
                                                   eta_min=1e-1 * opt.lr * opt.slow)
if opt.fix_alg == 'G':
    searchG = False
else:
    searchG = True

search_iter = 0

gzsl_unseen = []
gzsl_seen = []
gzsl_H = []
zsl_unseen = []
fake_feature = None
fake_label = None

pop_similarity = []
gene_similarity = []
gene_similarity.append(0.0)
bef_gene = None

time_str = time.strftime("%m-%d-%H-%M", time.localtime())

for epoch in range(opt.nepoch):
    schedulerG.step()
    schedulerD.step()
    if epoch >= opt.warmup_nepoch:
        if searchG:
            genotypes_G = search_evol_arch1(opt, genotypes_G, G_alg)
        else:
            genotypes_D = search_evol_arch1(opt, genotypes_D, D_alg)
    # ==================================train GAN=================================== #
    for i in range(0, data.ntrain, opt.batch_size):
        # (1) Update netD
        for p in netG_search.parameters():
            p.requires_grad = False
        for p in netD_search.parameters():  # reset requires_grad
            p.requires_grad = True
        iter_d = 0
        while True:
            if epoch < opt.warmup_nepoch:
                g_G = G_init
                g_D = D_init
            elif searchG:
                g_D = genotypes_D[0]
                g_G = genotypes_G[random.randint(0, genotypes_G.shape[0] - 1), :, :]
            else:
                g_G = genotypes_G[0]
                g_D = genotypes_D[random.randint(0, genotypes_D.shape[0] - 1), :, :]

            netD_search.zero_grad()
            sample(opt.batch_size)
            criticD_real = netD_search(input_res, input_att, g_D).mean()

            noise.normal_(0, 1)
            as_fake = opt.att_std * noise + input_att
            noise.normal_(0, 1)
            fake = netG_search(noise, as_fake, g_G).detach()
            criticD_fake = netD_search(fake, input_att, g_D).mean()

            gradient_penalty = util.calc_gradient_penalty(opt, netD_search, input_res, fake.data,
                                                          input_att, g_D)  # gradient penalty

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

        # (2) Update netG_search
        for p in netD_search.parameters():  # reset requires_grad
            p.requires_grad = False
        for p in netG_search.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False above in netD update
        netG_search.zero_grad()
        fake = netG_search(noise, as_fake, g_G)
        G_cost = -netD_search(fake, input_att, g_D).mean()
        G_cost.backward()
        optimizerG.step()
        G_cost = G_cost.item()  # cost of the generator

    torch.cuda.empty_cache()
    print('[%d/%d] iter_d:%d Loss_D: %.4f Loss_G: %.4f W_d: %.4f'
          % (epoch, opt.nepoch, iter_d, D_cost, G_cost, W_D))
    #############################
    # classifier training phase
    #############################
    netG_search.eval()
    netD_search.eval()
    if epoch == opt.warmup_nepoch - 1:
        if opt.fix_alg == 'D' or opt.fix_alg is None:
            genotypes_G = search_evol_Garch(netG_search, genotypes_G, netD_search, genotypes_D[0], G_alg)
        if opt.fix_alg == 'G' or opt.fix_alg is None:
            genotypes_D = search_evol_Darch(netG_search, genotypes_G[0], netD_search, genotypes_D, D_alg)
        g_G = genotypes_G[0]
    elif epoch >= opt.warmup_nepoch:
        if searchG:
            genotypes_G = search_evol_Garch(netG_search, genotypes_G, netD_search, genotypes_D[0], G_alg)
        else:
            genotypes_D = search_evol_Darch(netG_search, genotypes_G[0], netD_search, genotypes_D, D_alg)

        g_G = genotypes_G[0]
    else:
        g_G = G_init

    if opt.test_data:
        unseen = data.unseenclasses0
    else:
        unseen = data.unseenclasses

    syn_feature, syn_label = util.generate_syn_feature(opt, netG_search, unseen, data.attribute,
                                                       opt.syn_num, g_G)  # generate pseudo unseen samples
    cls_zsl = classifier1.CLASSIFIER(opt, syn_feature, util.map_label(syn_label, unseen), data, opt.lr_classifier,
                                   opt.beta1, opt.nepoch_classifier, opt.batch_size, False, opt.test_data)
    acc = cls_zsl.acc
    zsl_unseen.append(acc)
    print('unseen class accuracy=%.4f '% (cls_zsl.acc))
    if opt.gzsl:
        if opt.test_data:
            train_X = torch.cat((data.trainval_feature, syn_feature), dim=0)
            train_Y = torch.cat((data.trainval_label, syn_label), dim=0)
            all_class = data.allclasses
        else:
            train_X = torch.cat((data.train_feature, syn_feature), dim=0)
            train_Y = torch.cat((data.train_label, syn_label), dim=0)
            all_class = data.seenclasses0

        cls_gzsl = ClassifierClass(opt.netM_layer_sizes, opt.lambda_1, train_X,
                                   util.map_label(train_Y, all_class), data, opt.lr_classifier,
                                   opt.beta1, opt.nepoch_classifier, opt.batch_size,
                                   opt.temperature, True, opt.test_data)
        acc = cls_gzsl.H
        gzsl_unseen.append(cls_gzsl.acc_unseen)
        gzsl_seen.append(cls_gzsl.acc_seen)
        gzsl_H.append(cls_gzsl.H)
        print('unseen=%.4f, seen=%.4f, h=%.4f' % (cls_gzsl.acc_unseen, cls_gzsl.acc_seen, cls_gzsl.H))
        if opt.classifier_module_name == "classifier1":
            if cls_gzsl.acc_unseen-cls_gzsl.acc_seen >= 0.1:
                opt.syn_num = int(opt.syn_num * 0.9)
                print("syn_num=%d" % opt.syn_num)
    if epoch == opt.warmup_nepoch - 1:
        best_Dgenotype = genotypes_D[0]
        print('Best')
        print('Dgenerator', genotypes_D[0])
    elif epoch >= opt.warmup_nepoch:
        search_iter += 1
        if best_acc < acc:
            best_epoch = epoch
            best_acc = acc
            best_cls_zsl = cls_zsl
            fake_feature = syn_feature
            fake_label = syn_label

            if (opt.gzsl):
                best_gzsl_acc = cls_gzsl.H
                best_cls_gzsl = cls_gzsl

            if searchG:
                best_Ggenotype = genotypes_G[0]
                print('Best')
                print('Ggenerator', genotypes_G[0])
            else:
                best_Dgenotype = genotypes_D[0]
                print('Best')
                print('Dgenerator', genotypes_D[0])
        else:
            if searchG:
                if (not G_alg.judge_repeat(best_Ggenotype)) and opt.retain_genotype:
                    G_alg.genotypes[G_alg.encode(best_Ggenotype)] = best_Ggenotype
                    best_genotype_3d = best_Ggenotype[np.newaxis, :, :]
                    genotypes_G = np.concatenate((genotypes_G, best_genotype_3d), axis=0)
            else:
                if (not D_alg.judge_repeat(best_Dgenotype)) and opt.retain_genotype:
                    D_alg.genotypes[D_alg.encode(best_Dgenotype)] = best_Dgenotype
                    best_genotype_3d = best_Dgenotype[np.newaxis, :, :]
                    genotypes_D = np.concatenate((genotypes_D, best_genotype_3d), axis=0)
            if epoch - best_epoch > 10 and search_iter > 10:
                if opt.fix_alg is None:
                    searchG = not searchG
                search_iter = 0
    if opt.fix_alg is not None:
        break_num = 20
    else:
        break_num = 40
    if epoch - best_epoch >= break_num:
        break

    netG_search.train()
    netD_search.train()
    torch.cuda.empty_cache()

os.makedirs(opt.geo_dir, exist_ok=True)

if (opt.fix_alg is None) or opt.fix_alg == "D":
    np.savez(opt.geo_dir + "/{}_G".format(opt.dataset), best_genotype=best_Ggenotype)
if (opt.fix_alg is None) or opt.fix_alg == "G":
    np.savez(opt.geo_dir + "/{}_D".format(opt.dataset), best_genotype=best_Dgenotype)

time_end = time.strftime("%m-%d-%H-%M", time.localtime())

file_path = opt.geo_dir + "alloutput.txt".format(opt.dataset)
with open(file_path, 'a') as file:
    file.write("____________________search_____________________\n")
    file.write(time_str+"->"+time_end+"\n")
    file.write("manualSeed={}\n".format(opt.manualSeed))
    if (opt.gzsl):
        file.write("best_epoch={} acc_unseen={} acc_seen={} H={}\n".format(best_epoch, best_cls_gzsl.acc_unseen, best_cls_gzsl.acc_seen, best_cls_gzsl.H))
    file.write("unseen class accuracy={}\n".format(best_cls_zsl.acc))
    for key, value in vars(opt).items():
        file.write(f'{key}: {value}\t')
    file.write("\n")

if (opt.gzsl):
    print('unseen=%.4f, seen=%.4f, h=%.4f' % (best_cls_gzsl.acc_unseen, best_cls_gzsl.acc_seen, best_cls_gzsl.H))
print('unseen class accuracy= ', best_cls_zsl.acc)

if opt.save_data:
    np.savez(opt.geo_dir + "/{}_similarity_{}".format(opt.dataset, opt.manualSeed),
             pop_similarity=pop_similarity, gene_similarity=gene_similarity)
    np.savez(opt.geo_dir + "/{}_fakedata_{}".format(opt.dataset, opt.manualSeed),
             fake_feature=fake_feature, fake_label=fake_label)
    np.savez(opt.geo_dir + "/{}_traindata_{}".format(opt.dataset, opt.manualSeed),
             Epoch=range(1, len(zsl_unseen) + 1), zsl_unseen=zsl_unseen,
             gzsl_unseen=gzsl_unseen, gzsl_seen=gzsl_seen, gzsl_H=gzsl_H)

