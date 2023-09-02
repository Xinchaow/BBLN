import gzip
import os
import json
import pickle
from collections import defaultdict

import numpy as np
import copy

import pandas as pd
import torch
import random

from tqdm import tqdm

from utils import UnionFindSet, get_bfs_sub_graph, get_dfs_sub_graph
from torch_geometric.data import Data, Dataset, InMemoryDataset, DataLoader


device = 'cuda:0'

with open('protein_GO_terms.json', 'r') as file:
    tfprotdict = json.load(file)

with open('protein_GO_emb.json', 'r') as file:
    protein_go_emb = json.load(file)


def embed_normal(seq, dim, max_len):
    padd_mask_pytorch = torch.ones(max_len, dtype=torch.bool)
    if len(seq) > max_len:
        padd_mask_pytorch[0: max_len] = False
        return seq[:max_len], padd_mask_pytorch
    elif len(seq) < max_len:
        less_len = max_len - len(seq)
        padd_mask_pytorch[0: len(seq)] = False
        return np.concatenate((seq, np.zeros((less_len, dim)))), padd_mask_pytorch
    padd_mask_pytorch[0: max_len] = False
    return seq, padd_mask_pytorch


class DATA:
    def __init__(self, ppi_path, exclude_protein_path=None, max_len=2000, skip_head=True, p1_index=0, p2_index=1,
                 label_index=2, graph_undirection=True, bigger_ppi_path=None):
        self.ppi_list = []  # 数据集中每个ppi对应的两个蛋白质,形如:[[蛋白质A,蛋白质B],[蛋白质A，蛋白质C],[蛋白质B,蛋白质C],...]
        self.ppi_dict = {}  # 数据集中每个ppi对应编号的字典,key为蛋白质名_蛋白质名,value为对应编号(1,2,3...)
        self.ppi_label_list = []  # 数据集中每个ppi对应的标签文件,其中之一的元素例如[1,0,1,1,1,0,0]
        self.protein_dict = {}  # 数据集中的每个蛋白质的氨基酸序列,其中每个氨基酸以向量形式表达,key为蛋白质名,value为蛋白质对应的氨基酸序列
        self.protein_name = {}  # 数据集中的蛋白质名称,key为蛋白质名,value为对应编号(1,2,3...)
        self.ppi_path = ppi_path
        self.bigger_ppi_path = bigger_ppi_path
        self.max_len = max_len

        name = 0
        ppi_name = 0
        # maxlen = 0
        self.node_num = 0
        self.edge_num = 0

        if exclude_protein_path != None:
            with open(exclude_protein_path, 'r') as f:
                ex_protein = json.load(f)
                f.close()
            ex_protein = {p: i for i, p in enumerate(ex_protein)}
        else:
            ex_protein = {}

        class_map = {'reaction': 0, 'binding': 1, 'ptmod': 2, 'activation': 3, 'inhibition': 4, 'catalysis': 5,
                     'expression': 6}

        for line in tqdm(open(ppi_path)):

            if skip_head:
                skip_head = False
                continue
            line = line.strip().split('\t')

            if line[p1_index] in ex_protein.keys() or line[p2_index] in ex_protein.keys():
                continue

            if line[p1_index] not in tfprotdict.keys() or line[p2_index] not in tfprotdict.keys():
                continue

            # get node and node name
            if line[p1_index] not in self.protein_name.keys():
                self.protein_name[line[p1_index]] = name
                name += 1

            if line[p2_index] not in self.protein_name.keys():
                self.protein_name[line[p2_index]] = name
                name += 1

            # get edge and its label
            temp_data = ""
            if line[p1_index] < line[p2_index]:
                temp_data = line[p1_index] + "__" + line[p2_index]
            else:
                temp_data = line[p2_index] + "__" + line[p1_index]

            if temp_data not in self.ppi_dict.keys():
                self.ppi_dict[temp_data] = ppi_name
                temp_label = [0, 0, 0, 0, 0, 0, 0]
                temp_label[class_map[line[label_index]]] = 1
                self.ppi_label_list.append(temp_label)
                ppi_name += 1
            else:
                index = self.ppi_dict[temp_data]
                temp_label = self.ppi_label_list[index]
                temp_label[class_map[line[label_index]]] = 1
                self.ppi_label_list[index] = temp_label

        if bigger_ppi_path != None:
            skip_head = True
            for line in tqdm(open(bigger_ppi_path)):
                if skip_head:
                    skip_head = False
                    continue
                line = line.strip().split('\t')

                if line[p1_index] not in tfprotdict.keys() or line[p2_index] not in tfprotdict.keys():
                    continue

                if line[p1_index] not in self.protein_name.keys():
                    self.protein_name[line[p1_index]] = name
                    name += 1

                if line[p2_index] not in self.protein_name.keys():
                    self.protein_name[line[p2_index]] = name
                    name += 1

                temp_data = ""
                if line[p1_index] < line[p2_index]:
                    temp_data = line[p1_index] + "__" + line[p2_index]
                else:
                    temp_data = line[p2_index] + "__" + line[p1_index]

                if temp_data not in self.ppi_dict.keys():
                    self.ppi_dict[temp_data] = ppi_name
                    temp_label = [0, 0, 0, 0, 0, 0, 0]
                    temp_label[class_map[line[label_index]]] = 1
                    self.ppi_label_list.append(temp_label)
                    ppi_name += 1
                else:
                    index = self.ppi_dict[temp_data]
                    temp_label = self.ppi_label_list[index]
                    temp_label[class_map[line[label_index]]] = 1
                    self.ppi_label_list[index] = temp_label

        # 获取ppi_list
        i = 0
        for ppi in tqdm(self.ppi_dict.keys()):
            name = self.ppi_dict[ppi]
            assert name == i
            i += 1
            temp = ppi.strip().split('__')
            self.ppi_list.append(temp)

        ppi_num = len(self.ppi_list)
        self.origin_ppi_list = copy.deepcopy(self.ppi_list)
        assert len(self.ppi_list) == len(self.ppi_label_list)
        for i in tqdm(range(ppi_num)):
            seq1_name = self.ppi_list[i][0]
            seq2_name = self.ppi_list[i][1]
            # print(len(self.protein_name))
            # 把ppi_list里的蛋白质名换成name对应的编号
            self.ppi_list[i][0] = self.protein_name[seq1_name]
            self.ppi_list[i][1] = self.protein_name[seq2_name]

        # 无向图
        if graph_undirection:
            for i in tqdm(range(ppi_num)):
                temp_ppi = self.ppi_list[i][::-1]
                temp_ppi_label = self.ppi_label_list[i]
                # if temp_ppi not in self.ppi_list:
                self.ppi_list.append(temp_ppi)
                self.ppi_label_list.append(temp_ppi_label)

        self.node_num = len(self.protein_name)
        self.edge_num = len(self.ppi_list)

        self.prot_go_emb = dict()
        for p_name in self.protein_name.keys():
            if p_name not in protein_go_emb.keys():
                self.prot_go_emb[p_name] = [[0 for i in range(64)]]
            else:
                self.prot_go_emb[p_name] = protein_go_emb[p_name]

        maxlen = 0
        for p_name in self.prot_go_emb.keys():
            maxlen = max(maxlen, len(self.prot_go_emb[p_name]))

        self.prot_go_mask = dict()
        # padding
        for p_name in self.prot_go_emb.keys():
            emb = np.array(self.prot_go_emb[p_name])
            emb, emb_mask = embed_normal(emb, 64, 50)
            self.prot_go_emb[p_name] = emb
            self.prot_go_mask[p_name] = emb_mask

    # 获取蛋白质序列
    def get_protein_aac(self, pseq_path):
        # aac: amino acid sequences
        self.pseq_path = pseq_path
        self.pseq_dict = {}  # 各个蛋白质的氨基酸序列字典: "蛋白质名":"ANDSSA..."
        self.protein_len = []

        for line in tqdm(open(self.pseq_path)):
            line = line.strip().split('\t')
            if line[0] not in self.pseq_dict.keys():
                self.pseq_dict[line[0]] = line[1]
                self.protein_len.append(len(line[1]))

        print("protein num: {}".format(len(self.pseq_dict)))
        print("protein average length: {}".format(np.average(self.protein_len)))
        print("protein max & min length: {}, {}".format(np.max(self.protein_len), np.min(self.protein_len)))

    def embed_normal(self, seq, dim):
        if len(seq) > self.max_len:
            return seq[:self.max_len]
        elif len(seq) < self.max_len:
            less_len = self.max_len - len(seq)
            return np.concatenate((seq, np.zeros((less_len, dim))))
        return seq

    def vectorize(self, vec_path):
        self.acid2vec = {}  # 每个氨基酸对应的向量字典: "A":"[-1.545, 0.8, ... , 1, 0, 0,..., 0]"
        self.dim = None
        for line in open(vec_path):
            line = line.strip().split('\t')
            temp = np.array([float(x) for x in line[1].split()])
            self.acid2vec[line[0]] = temp
            if self.dim is None:
                # self.dim = len(temp) + 1 # 加上一维asa
                self.dim = len(temp)
        print("acid vector dimension: {}".format(self.dim))

        self.pvec_dict = {} # 每个蛋白质的氨基酸以向量形式存放的字典: "蛋白质名":"[[氨基酸1的向量形式],[氨基酸2的向量形式],...]"

        for p_name in tqdm(self.pseq_dict.keys()):
            temp_seq = self.pseq_dict[p_name]
            temp_vec = []
            for acid in temp_seq:
                temp_vec.append(self.acid2vec[acid])

            temp_vec = np.array(temp_vec)

            temp_vec = self.embed_normal(temp_vec, self.dim)

            self.pvec_dict[p_name] = temp_vec

    def get_feature_origin(self, pseq_path, vec_path):
        self.get_protein_aac(pseq_path)  # 给self.pseq_dict赋值,该字典中保存了所有的蛋白质对应的氨基酸序列

        self.vectorize(vec_path)  # 对self.pvec_dict赋值,该字典保存了所有蛋白质对应的氨基酸序列(氨基酸以向量形式)

        self.protein_dict = {}  # 对数据集中的蛋白质对应的氨基酸序列进行向量化,保存在self.protein_dict中
        for name in tqdm(self.protein_name.keys()):
            self.protein_dict[name] = self.pvec_dict[name]

    def get_connected_num(self):
        self.ufs = UnionFindSet(self.node_num)
        ppi_ndary = np.array(self.ppi_list)
        for edge in ppi_ndary:
            start, end = edge[0], edge[1]
            self.ufs.union(start, end)

    def generate_data(self):
        self.get_connected_num()

        print("Connected domain num: {}".format(self.ufs.count))

        ppi_list = np.array(self.ppi_list)
        ppi_label_list = np.array(self.ppi_label_list)

        self.edge_index = torch.tensor(ppi_list, dtype=torch.long)
        self.edge_attr = torch.tensor(ppi_label_list, dtype=torch.long)
        self.x = []
        self.x_GO = []
        self.x_mask = []
        i = 0
        for name in self.protein_name:
            assert self.protein_name[name] == i
            i += 1
            self.x.append(self.protein_dict[name])
            self.x_GO.append(self.prot_go_emb[name])
            self.x_mask.append(self.prot_go_mask[name])

        self.x = np.array(self.x)
        self.x = torch.tensor(self.x, dtype=torch.float)
        self.x_GO = np.array(self.x_GO)
        self.x_GO = torch.tensor(self.x_GO, dtype=torch.float, device=device)
        mask_list = [aa.tolist() for aa in self.x_mask]
        self.x_mask = torch.tensor(mask_list, dtype=torch.bool, device=device)

        self.data = Data(x=self.x, edge_index=self.edge_index.T, edge_attr_1=self.edge_attr)

    def split_dataset(self, train_valid_index_path, test_size=0.2, random_new=False, mode='random'):
        if random_new:
            if mode == 'random':
                ppi_num = int(self.edge_num // 2)
                random_list = [i for i in range(ppi_num)]
                random.shuffle(random_list)

                self.ppi_split_dict = {}
                self.ppi_split_dict['train_index'] = random_list[: int(ppi_num * (1 - test_size))]
                self.ppi_split_dict['valid_index'] = random_list[int(ppi_num * (1 - test_size)):]

                jsobj = json.dumps(self.ppi_split_dict)
                with open(train_valid_index_path, 'w') as f:
                    f.write(jsobj)
                    f.close()

            elif mode == 'bfs' or mode == 'dfs':
                print("use {} methed split train and valid dataset".format(mode))
                node_to_edge_index = {}
                edge_num = int(self.edge_num // 2)
                for i in range(edge_num):
                    edge = self.ppi_list[i]
                    if edge[0] not in node_to_edge_index.keys():
                        node_to_edge_index[edge[0]] = []
                    node_to_edge_index[edge[0]].append(i)

                    if edge[1] not in node_to_edge_index.keys():
                        node_to_edge_index[edge[1]] = []
                    node_to_edge_index[edge[1]].append(i)

                node_num = len(node_to_edge_index)

                sub_graph_size = int(edge_num * test_size)
                if mode == 'bfs':
                    selected_edge_index = get_bfs_sub_graph(self.ppi_list, node_num, node_to_edge_index, sub_graph_size)
                elif mode == 'dfs':
                    selected_edge_index = get_dfs_sub_graph(self.ppi_list, node_num, node_to_edge_index, sub_graph_size)

                all_edge_index = [i for i in range(edge_num)]

                unselected_edge_index = list(set(all_edge_index).difference(set(selected_edge_index)))

                self.ppi_split_dict = {}
                self.ppi_split_dict['train_index'] = unselected_edge_index
                self.ppi_split_dict['valid_index'] = selected_edge_index
                # self.ppi_split_dict['train_index'] = selected_edge_index
                # self.ppi_split_dict['valid_index'] = unselected_edge_index

                assert len(unselected_edge_index) + len(selected_edge_index) == edge_num

                jsobj = json.dumps(self.ppi_split_dict)
                with open(train_valid_index_path, 'w') as f:
                    f.write(jsobj)
                    f.close()

            else:
                print("your mode is {}, you should use bfs, dfs or random".format(mode))
                return
        else:
            with open(train_valid_index_path, 'r') as f:
                self.ppi_split_dict = json.load(f)
                f.close()
