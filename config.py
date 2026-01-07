import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(add_help=False)

parser.add_argument('--dataset', default='AWA2', help='dateset')
parser.add_argument('--syn_num', type=int, default=100, help='number samples to generate per class')
parser.add_argument('--batch_size', type=int, default=256, help='512 256 input batch size')
parser.add_argument('--nepoch_classifier', type=int, default=80, help='number of epochs to train for the mapping net')
parser.add_argument('--att_std', type=float, default=0.08, help='std of the attribute augmentation noise')
parser.add_argument('--lambda_1', type=float, default=0.005)
parser.add_argument('--temperature', type=float, default=0.04)
parser.add_argument('--test_data', type=str2bool, default=True, help='learning rate to train softmax classifier')

parser.add_argument('--manualSeed', type=int, default=None, help='manual seed, default=1429')
parser.add_argument('--save_data', type=str2bool, default=True, help='manual seed, default=1429')
parser.add_argument('--fix_alg',  default=None, help='Wether fixed an alg, only search other alg  G/D/None')
parser.add_argument('--w_up', type=float, default=2.0, help='')
parser.add_argument('--dynamic_D', type=str2bool, default=True, help='Whether to dynamically adjust the number of discriminator training iterations')
parser.add_argument('--new_genotype', type=str2bool, default=True, help='Whether new genes are introduced during evolution')
parser.add_argument('--node_crossover', type=str2bool, default=True, help='Whether to use random point crossover')
parser.add_argument('--pop_evolution', type=str2bool, default=True, help='Whether to use population evolution')
parser.add_argument('--retain_genotype', type=str2bool, default=True, help='Whether to retain the best genotype')
parser.add_argument('--SelfAttention', type=str2bool, default=True, help='Whether to use SelfAttention')

parser.add_argument('--outf', default='./output/checkpoint/', help='folder to output data and model checkpoints')
parser.add_argument('--geo_dir', default='./output/genotypes/', help='folder to save the searched architectures')
parser.add_argument('--dataroot', default='./data', help='path to dataset')
parser.add_argument('--image_embedding', default='res101')
parser.add_argument('--class_embedding', default='att', help='att or sent')
parser.add_argument('--resSize', type=int, default=2048, help='size of visual features')
parser.add_argument('--attSize', type=int, default=85, help='size of attribute features')
parser.add_argument('--nz', type=int, default=85, help='size of the latent z vector')

parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--lambda_0', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
parser.add_argument('--slow', type=float, default=0.1, help='beta1 for adam. default=0.5')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
parser.add_argument('--lr_classifier', type=float, default=1e-4, help='learning rate to train mapping net')
parser.add_argument('--netG_search', default='', help="path to netG_search (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--netD_layer_sizes', type=list, default=[4096])
parser.add_argument('--netG_layer_sizes', type=list, default=[4096, 2048, 2048])
parser.add_argument('--netM_layer_sizes', type=list, default=[1024, 2048])


parser.add_argument('--num_nodes', type=int, default=5, help='# num of population')
parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--device', default='cuda:0')
parser.add_argument('--gzsl', type=str2bool,  default=True, help='enable generalized zero-shot learning')
parser.add_argument('--classifier_module_name', default='classifier', help='classifier')
