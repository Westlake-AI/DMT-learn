import os
import time
import math
import json
import random
import numpy as np
import argparse
import torch
import torch.nn as nn

from tqdm import tqdm

from gnn_data import GNN_DATA
from gnn_model import GIN_Net2, GIN_Net3
from utils import Metrictor_PPI, print_file
import dmt_loss as dmtloss

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

parser = argparse.ArgumentParser(description='Test Model')
parser.add_argument('--description', default='KeAP20_27k_DMT', type=str,
                    help='train description')
parser.add_argument('--ppi_path', default="./data/protein.actions.SHS27k.STRING.txt", type=str,
                    help="ppi path")
parser.add_argument('--pseq_path', default="data/protein.SHS27k.sequences.dictionary.tsv", type=str,
                    help="protein sequence path")
parser.add_argument('--vec_path', default="data/PPI_embeddings/protein_embedding_KeAP20_shs27k.npy", type=str,
                    help='protein sequence vector path')

parser.add_argument('--index_path', default='data/new_train_valid_index_json/SHS27k.dfs.fold1.json', type=str,
                    help='cnn_rnn and gnn unified train and valid ppi index')
parser.add_argument('--gnn_model', default='output/ppi/gnn_KeAP20_string_dfs_0_True_0.01_2024-11-09_15:54:54/gnn_model_valid_best.ckpt', type=str,
                    help="gnn trained model")
parser.add_argument('--test_all', default='False', type=boolean_string,
                    help="test all or test separately")


##hyper-parameters for DMT
parser.add_argument('--use_dmt', default=True, type=bool)
parser.add_argument('--v_input', default=100, type=float)
parser.add_argument('--v_latent', default=0.01, type=float)
parser.add_argument('--sigmaP', default=1.0, type=float)
parser.add_argument('--augNearRate', default=100000, type=float)
parser.add_argument('--balance', default=1e-3, type=float)
parser.add_argument('--seed', default=0, type=int)


def Similarity(dist, rho, sigma_array, gamma, v=100):

    # if torch.is_tensor(rho):
    dist_rho = (dist - rho) / sigma_array
    # else:
    #     dist_rho = dist
    
    dist_rho[dist_rho < 0] = 0
    # Pij = torch.pow(
    #     gamma * torch.pow(
    #         (1 + dist_rho / v),
    #         -1 * (v + 1) / 2
    #         ) * torch.sqrt(torch.tensor(2 * 3.14)),
    #         2
    #     )
    Pij = gamma*gamma * torch.pow(
            (1 + dist_rho / v),
            -1 * (v + 1)
            ) * 2 * 3.14
    # print(Pij, Pij2)
    # input()
    P = Pij + Pij.t() - torch.mul(Pij, Pij.t())

    return P

def test(model, graph, test_mask, device):
    valid_pre_result_list = []
    valid_label_list = []

    model.eval()

    batch_size = 256

    valid_steps = math.ceil(len(test_mask) / batch_size)

    for step in tqdm(range(valid_steps)):
        if step == valid_steps-1:
            valid_edge_id = test_mask[step*batch_size:]
        else:
            valid_edge_id = test_mask[step*batch_size : step*batch_size + batch_size]

        output, latent = model(graph.x, graph.edge_index, valid_edge_id)
        label = graph.edge_attr_1[valid_edge_id]
        label = label.type(torch.FloatTensor).to(device)

        m = nn.Sigmoid()
        pre_result = (m(output) > 0.5).type(torch.FloatTensor).to(device)

        valid_pre_result_list.append(pre_result.cpu().data)
        valid_label_list.append(label.cpu().data)

    valid_pre_result_list = torch.cat(valid_pre_result_list, dim=0)
    valid_label_list = torch.cat(valid_label_list, dim=0)

    metrics = Metrictor_PPI(valid_pre_result_list, valid_label_list)

    metrics.show_result()

    print("Recall: {}, Precision: {}, F1: {}".format(metrics.Recall, metrics.Precision, metrics.F1))

def main():

    args = parser.parse_args()

    ppi_data = GNN_DATA(ppi_path=args.ppi_path)

    ppi_data.get_feature_origin(pseq_path=args.pseq_path, vec_path=args.vec_path)

    ppi_data.generate_data()

    graph = ppi_data.data
    temp = graph.edge_index.transpose(0, 1).numpy()
    ppi_list = []

    for edge in temp:
        ppi_list.append(list(edge))

    truth_edge_num = len(ppi_list) // 2
    # fake_edge_num = len(ppi_data.fake_edge) // 2
    fake_edge_num = 0
    
    with open(args.index_path, 'r') as f:
        index_dict = json.load(f)
        f.close()
    graph.train_mask = index_dict['train_index']

    graph.val_mask = index_dict['valid_index']

    print("train gnn, train_num: {}, valid_num: {}".format(len(graph.train_mask), len(graph.val_mask)))

    node_vision_dict = {}
    for index in graph.train_mask:
        ppi = ppi_list[index]
        if ppi[0] not in node_vision_dict.keys():
            node_vision_dict[ppi[0]] = 1
        if ppi[1] not in node_vision_dict.keys():
            node_vision_dict[ppi[1]] = 1

    for index in graph.val_mask:
        ppi = ppi_list[index]
        if ppi[0] not in node_vision_dict.keys():
            node_vision_dict[ppi[0]] = 0
        if ppi[1] not in node_vision_dict.keys():
            node_vision_dict[ppi[1]] = 0
    
    vision_num = 0
    unvision_num = 0
    for node in node_vision_dict:
        if node_vision_dict[node] == 1:
            vision_num += 1
        elif node_vision_dict[node] == 0:
            unvision_num += 1
    print("vision node num: {}, unvision node num: {}".format(vision_num, unvision_num))

    test1_mask = []
    test2_mask = []
    test3_mask = []

    for index in graph.val_mask:
        ppi = ppi_list[index]
        temp = node_vision_dict[ppi[0]] + node_vision_dict[ppi[1]]
        if temp == 2:
            test1_mask.append(index)
        elif temp == 1:
            test2_mask.append(index)
        elif temp == 0:
            test3_mask.append(index)
    print("test1 edge num: {}, test2 edge num: {}, test3 edge num: {}".format(len(test1_mask), len(test2_mask), len(test3_mask)))

    graph.test1_mask = test1_mask
    graph.test2_mask = test2_mask
    graph.test3_mask = test3_mask

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # model = GIN_Net2(in_len=2000, in_feature=13, gin_in_feature=256, num_layers=1, pool_size=3, cnn_hidden=1).to(device)
    model = GIN_Net3(embed_size=1024, gin_in_feature=256, num_layers=1, pool_size=3, cnn_hidden=1).to(device)

    model.load_state_dict(torch.load(args.gnn_model)['state_dict'])

    graph.to(device)

    if args.test_all:
        print("---------------- valid-test-all result --------------------")
        test(model, graph, graph.val_mask, device)
    else:
        print("---------------- valid-test1 result --------------------")
        if len(graph.test1_mask) > 0:
            test(model, graph, graph.test1_mask, device)
        print("---------------- valid-test2 result --------------------")
        if len(graph.test2_mask) > 0:
            test(model, graph, graph.test2_mask, device)
        print("---------------- valid-test3 result --------------------")
        if len(graph.test3_mask) > 0:
            test(model, graph, graph.test3_mask, device)

if __name__ == "__main__":
    main()
