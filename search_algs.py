from __future__ import absolute_import, division, print_function
import numpy as np
import random

from tqdm import tqdm

from operations import *


class GanAlgorithm():
    def __init__(self, opt):
        self.genotypes = {}
        self.num_initial_input = opt.num_initial_input
        self.num_node = opt.num_nodes
        self.operation_name_list = []
        # Generate all the mixed layer
        for i in range(self.num_node):
            # All previous outputs and additional inputs
            for j in range(i + self.num_initial_input):
                if j < self.num_initial_input:  # Input layer
                    self.operation_name_list.append(list(operation_dict_all.keys()))
        # Alpha list for each operation
        self.num_eg = int((self.num_node + 2 * self.num_initial_input - 1) * self.num_node / 2)
        self.num_op = len(self.operation_name_list[0])
        genotype_init = []
        for i in range(self.num_eg):
            genotype_init.append(np.ones(self.num_op))
        self.genotype_init = np.stack(genotype_init)

    def encode(self, genotype):
        lists = [0 for i in range(self.num_eg)]
        for i in range(len(lists)):
            lists[i] = str(genotype[i])
        return tuple(lists)

    def clean(self, genotype):
        node = [0 for i in range(self.num_node - 1)]
        node_ = [0 for i in range(self.num_node - 1)]
        input_node = [0 for i in range(self.num_initial_input)]

        for k in range(3):
            offset = 0
            for i in range(self.num_node):
                for j in range(self.num_initial_input):
                    input_node[j] += genotype[offset + j, -1]
                offset += self.num_initial_input + i
            for i in range(self.num_initial_input):
                input_node[i] = input_node[i] // self.num_node
            if (sum(input_node) >= self.num_initial_input - 1) and (input_node[0] != 1):
                return None
            offset = 0
            for i in range(self.num_node):
                if i < self.num_node - 1:
                    node_[i] = (genotype[offset:offset + self.num_initial_input + i, -1] - 1).sum()
                for j in range(self.num_initial_input + i):
                    if j - 3 >= 0:
                        node[j - 3] += genotype[offset, -1] - 1
                    offset += 1
            offset = 0
            offset_ = self.num_initial_input
            for i in range(self.num_node - 1):
                if node_[i] == 0:
                    offset_2 = offset_
                    if node[i] != 0:
                        for j in range(self.num_node - 1 - i):
                            offset_2 += self.num_initial_input + i
                            genotype[offset_2, :] = np.zeros(self.num_op)
                            genotype[offset_2, -1] = 1
                            offset_2 += j + 1
                        node[i] = 0
                offset_ += self.num_initial_input + i + 1
                if node[i] == 0 and node_[i] != 0:
                    for j in range(self.num_initial_input + i):
                        genotype[offset, :] = np.zeros(self.num_op)
                        genotype[offset, -1] = 1
                        offset += 1
                    node_[i] = 0
                else:
                    offset += self.num_initial_input + i
        if genotype[-self.num_node - self.num_initial_input + 1:, -1].sum() == (
                self.num_node + self.num_initial_input - 1):
            return None
        return genotype

    def search(self):
        new_genotype = self.sample()
        t = self.encode(new_genotype)
        while (t in self.genotypes):
            new_genotype = self.sample()
            new_genotype = self.clean(new_genotype)
            if new_genotype is None:
                continue
            t = self.encode(new_genotype)
        self.genotypes[t] = new_genotype
        return new_genotype

    def sample(self):
        genotype = np.zeros((self.num_eg, self.num_op), dtype=int)
        for i in range(self.num_eg):
            if np.random.rand() < 0.1:
                x1 = self.num_op - 1
            else:
                x1 = random.randint(0, self.num_op - 2)
            x2 = np.zeros(self.num_op)
            x2[x1] = 1
            genotype[i, :] = x2
        return genotype

    def judge_repeat(self, new_genotype):
        if new_genotype is None:
            return True
        t = self.encode(new_genotype)
        return t in self.genotypes

    def update(self, genotypes):
        self.genotypes = {}
        for i in range(genotypes.shape[0]):
            t = self.encode(genotypes[i])
            self.genotypes[t] = genotypes[i]


def search_evol_arch1(opt, genotypes, gan_alg):
    if genotypes.shape[0] < 2 * opt.num_individual:
        offsprings = gen_offspring(opt, genotypes, gan_alg)
        genotypes_new = np.concatenate((genotypes, offsprings), axis=0)
    else:
        genotypes_new = genotypes
    return genotypes_new


def gen_offspring(opt, alphas, gan_alg):
    num_offspring = 2 * opt.num_individual - alphas.shape[0]
    offsprings = []
    while len(offsprings) < num_offspring:
        rand = np.random.rand()
        if rand < 0.2 and opt.new_genotype:
            alphas_c = gan_alg.search()
        elif rand < 0.6:
            alphas_c = mutation(alphas[np.random.randint(0, alphas.shape[0])], gan_alg)
        else:
            a, b = np.random.randint(
                0, alphas.shape[0]), np.random.randint(0, alphas.shape[0])
            if a == b:
                continue
            alphas_c = crossover(opt, alphas[a], alphas[b], gan_alg)

        if not gan_alg.judge_repeat(alphas_c):
            offsprings.append(alphas_c)
            gan_alg.genotypes[gan_alg.encode(alphas_c)] = alphas_c
    offsprings = np.stack(offsprings, axis=0)
    return offsprings


def crossover(opt, alphas_a, alphas_b, gan_alg):
    """Crossover for two individuals."""
    # alpha a
    new_alphas = alphas_a.copy()
    if opt.node_crossover:
        node = random.randint(1, opt.num_nodes - 1)
        operation = node * (2 * opt.num_initial_input + node - 1) // 2
    else:
        operation = random.randint(0, alphas_a.shape[0] - 1)
    for j in range(operation, alphas_a.shape[0]):
        new_alphas[j, :] = alphas_b[j, :]
    new_alphas = gan_alg.clean(new_alphas)
    return new_alphas


def mutation(alphas_a, gan_alg):
    """Mutation for An individual."""
    new_alphas = alphas_a.copy()
    if np.random.rand() < 0.3:
        i = 1
    elif np.random.rand() > 0.3:
        i = 2
    else:
        i = 4
    for j in range(i):
        operation = random.randint(0, alphas_a.shape[0] - 1)
        if np.random.rand() < 0.2:
            x1 = alphas_a.shape[1] - 1
        else:
            x1 = random.randint(0, alphas_a.shape[1] - 2)
        x2 = np.zeros(alphas_a.shape[1])
        x2[x1] = 1
        new_alphas[operation, :] = x2
    new_alphas = gan_alg.clean(new_alphas)
    return new_alphas


def pop_evolution(genotypes, value, pop_size=3):
    best_value = value[np.argmin(value)]
    if best_value > 0:
        threshold = best_value + best_value * 0.5
    else:
        threshold = best_value - best_value * 0.5
    mask = value <= threshold
    gene_value = value[mask]
    optimize_gene = genotypes[mask]

    keep = np.argsort(gene_value)
    evolution_gene = []
    pop_num = 0
    max_pop_num = int(50 / pop_size)

    while len(keep) > 0 and len(evolution_gene) < 50:
        optimize_gene = optimize_gene[keep]
        gene_value = gene_value[keep]
        evolution_gene.append(optimize_gene[0])
        leader = optimize_gene[0]
        pop_num += 1
        keep = []
        num = 1
        for i in range(len(optimize_gene)-1):
            similarity_value = get_similarity(leader, optimize_gene[i+1])
            if similarity_value < 0.7:
                keep.append(i+1)
            elif num <= pop_size:
                evolution_gene.append(optimize_gene[i+1])
                num += 1

    print('种群个数：', pop_num)
    return np.array(evolution_gene)


def get_similarity(alphas_a, alphas_b):
    gen_similarity = np.sum(alphas_a * alphas_b)
    similarity_value = (float(gen_similarity-np.sum((alphas_a * alphas_b)[-1])) /
                        (alphas_a.shape[0]-np.sum((alphas_a * alphas_b)[-1])))
    return similarity_value

def get_pop_similarity(genotypes):
    if len(genotypes) < 2:
        return 0
    leader = genotypes[0]
    pop_similarity = 0
    for i in range(len(genotypes) - 1):
        similarity_value = get_similarity(leader, genotypes[i+1])
        pop_similarity += similarity_value

    pop_similarity /= len(genotypes)-1
    return pop_similarity