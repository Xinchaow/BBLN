import os
import time
import math
import random
import numpy as np
import argparse
import torch
import torch.nn as nn

from data import DATA
from model import BBLN
from utils import Metrictor_PPI, print_file
import torch.nn.functional as F

from tensorboardX import SummaryWriter

np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def sim(h1, h2):
    z1 = F.normalize(h1, dim=-1, p=2)
    z2 = F.normalize(h2, dim=-1, p=2)
    return torch.mm(z1, z2.t())

def contrastive_loss_cross_view(h1, h2):
    f = lambda x: torch.exp(x)
    cross_sim = f(sim(h1, h2))
    return -torch.log(cross_sim.diag() / cross_sim.sum(dim=-1))


parser = argparse.ArgumentParser(description='Train Model')
parser.add_argument('--description', default='random', type=str,
                    help='train description')
parser.add_argument('--ppi_path', default='data/protein.actions.SHS27k.STRING.txt', type=str,
                    help="ppi path")
parser.add_argument('--pseq_path', default='data/protein.SHS27k.sequences.dictionary.tsv', type=str,
                    help="protein sequence path")
parser.add_argument('--vec_path', default='data/vec5_CTC.txt', type=str,
                    help='protein sequence vector path')
parser.add_argument('--split_new', default=False, type=boolean_string,
                    help='split new index file or not')
parser.add_argument('--split_mode', default='random', type=str,
                    help='split method, random, bfs or dfs')
parser.add_argument('--train_valid_index_path', default='train_valid_index_json/train_valid_index_json.json', type=str,
                    help='cnn_rnn and gnn unified train and valid ppi index')
parser.add_argument('--use_lr_scheduler', default=True, type=boolean_string,
                    help="train use learning rate scheduler or not")
parser.add_argument('--save_path', default='save_model', type=str,
                    help='model save path')
parser.add_argument('--graph_only_train', default=False, type=boolean_string,
                    help='train ppi graph conctruct by train or all(train with test)')
parser.add_argument('--batch_size', default=128, type=int,
                    help="gnn train batch size, edge batch size")
parser.add_argument('--epochs', default=2000, type=int,
                    help='train epoch number')


def train(model, graph, x_go, go_mask, ppi_list, loss_fn, optimizer, device,
          result_file_path, summary_writer, save_path,
          batch_size=512, epochs=1000, scheduler=None):
    global_step = 0
    global_best_valid_f1 = 0.0
    global_best_valid_f1_epoch = 0

    for epoch in range(epochs):

        recall_sum = 0.0
        precision_sum = 0.0
        f1_sum = 0.0
        loss_sum = 0.0

        steps = math.ceil(len(graph.train_mask) / batch_size)

        model.train()

        random.shuffle(graph.train_mask)

        for step in range(steps):
            if step == steps - 1:
                train_edge_id = graph.train_mask[step * batch_size:]
            else:
                train_edge_id = graph.train_mask[step * batch_size: step * batch_size + batch_size]

            output, x_seq_cl, x_go_cl = model(graph.x, x_go, go_mask, graph.edge_index, train_edge_id)

            label = graph.edge_attr_1[train_edge_id]

            label = label.type(torch.FloatTensor).to(device)

            cmc_loss = 0.5 * contrastive_loss_cross_view(x_seq_cl, x_go_cl) + 0.5 * contrastive_loss_cross_view(x_go_cl, x_seq_cl)
            cmc_loss = cmc_loss.mean()

            cls_loss = loss_fn(output, label)

            loss = cls_loss + cmc_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            m = nn.Sigmoid()
            pre_result = (m(output) > 0.5).type(torch.FloatTensor).to(device)

            metrics = Metrictor_PPI(pre_result.cpu().data, label.cpu().data)

            metrics.show_result()

            recall_sum += metrics.Recall
            precision_sum += metrics.Precision
            f1_sum += metrics.F1
            loss_sum += loss.item()

            summary_writer.add_scalar('train/loss', loss.item(), global_step)
            summary_writer.add_scalar('train/precision', metrics.Precision, global_step)
            summary_writer.add_scalar('train/recall', metrics.Recall, global_step)
            summary_writer.add_scalar('train/F1', metrics.F1, global_step)

            global_step += 1
            print_file("epoch: {}, step: {}, Train: label_loss: {}, precision: {}, recall: {}, f1: {}"
                       .format(epoch, step, loss.item(), metrics.Precision, metrics.Recall, metrics.F1))

        torch.save({'epoch': epoch,
                    'state_dict': model.state_dict()},
                   os.path.join(save_path, 'gnn_model_train.ckpt'))

        valid_pre_result_list = []
        valid_label_list = []
        valid_loss_sum = 0.0

        model.eval()

        valid_steps = math.ceil(len(graph.val_mask) / batch_size)

        with torch.no_grad():
            for step in range(valid_steps):
                if step == valid_steps - 1:
                    valid_edge_id = graph.val_mask[step * batch_size:]
                else:
                    valid_edge_id = graph.val_mask[step * batch_size: step * batch_size + batch_size]

                output, _, _ = model(graph.x, x_go, go_mask, graph.edge_index, valid_edge_id)
                label = graph.edge_attr_1[valid_edge_id]
                label = label.type(torch.FloatTensor).to(device)

                loss = loss_fn(output, label)

                valid_loss_sum += loss.item()

                m = nn.Sigmoid()
                pre_result = (m(output) > 0.5).type(torch.FloatTensor).to(device)

                valid_pre_result_list.append(pre_result.cpu().data)
                valid_label_list.append(label.cpu().data)

        valid_pre_result_list = torch.cat(valid_pre_result_list, dim=0)
        valid_label_list = torch.cat(valid_label_list, dim=0)

        metrics = Metrictor_PPI(valid_pre_result_list, valid_label_list)

        metrics.show_result()

        recall = recall_sum / steps
        precision = precision_sum / steps
        f1 = f1_sum / steps
        loss = loss_sum / steps

        valid_loss = valid_loss_sum / valid_steps

        if scheduler != None:
            scheduler.step(loss)
            print_file("epoch: {}, now learning rate: {}".format(epoch, scheduler.optimizer.param_groups[0]['lr']),
                       save_file_path=result_file_path)

        if global_best_valid_f1 < metrics.F1:
            global_best_valid_f1 = metrics.F1
            global_best_valid_f1_epoch = epoch

            torch.save({'epoch': epoch,
                        'state_dict': model.state_dict()},
                       os.path.join(save_path, 'gnn_model_valid_best.ckpt'))

        summary_writer.add_scalar('valid/precision', metrics.Precision, global_step)
        summary_writer.add_scalar('valid/recall', metrics.Recall, global_step)
        summary_writer.add_scalar('valid/F1', metrics.F1, global_step)
        summary_writer.add_scalar('valid/loss', valid_loss, global_step)

        print_file(
            "epoch: {}, Training_avg: label_loss: {}, recall: {}, precision: {}, F1: {}, Validation_avg: loss: {}, recall: {}, precision: {}, F1: {}, Best valid_f1: {}, in {} epoch"
            .format(epoch, loss, recall, precision, f1, valid_loss, metrics.Recall, metrics.Precision, metrics.F1,
                    global_best_valid_f1, global_best_valid_f1_epoch), save_file_path=result_file_path)


def main():
    args = parser.parse_args()

    ppi_data = DATA(ppi_path=args.ppi_path)

    print("use_get_feature_origin")
    ppi_data.get_feature_origin(pseq_path=args.pseq_path, vec_path=args.vec_path)

    ppi_data.generate_data()

    print("----------------------- start split train and valid index -------------------")
    print("whether to split new train and valid index file, {}".format(args.split_new))
    if args.split_new:
        print("use {} method to split".format(args.split_mode))
    ppi_data.split_dataset(args.train_valid_index_path, random_new=args.split_new, mode=args.split_mode)
    print("----------------------- Done split train and valid index -------------------")

    graph = ppi_data.data
    x_go = ppi_data.x_go
    go_mask = ppi_data.go_mask

    ppi_list = ppi_data.ppi_list

    graph.train_mask = ppi_data.ppi_split_dict['train_index']
    graph.val_mask = ppi_data.ppi_split_dict['valid_index']

    print("train gnn, train_num: {}, valid_num: {}".format(len(graph.train_mask), len(graph.val_mask)))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    graph.to(device)
    x_go = x_go.to(device)
    go_mask = go_mask.to(device)

    model = BBLN(seq_in_len=2000, seq_in_feature=13, gin_in_feature=512, num_layers=1).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    scheduler = None
    if args.use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=25,
                                                               verbose=True)

    loss_fn = nn.BCEWithLogitsLoss().to(device)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    time_stamp = time.strftime("%Y_%m_%d_%H_%M_%S")
    save_path = os.path.join(args.save_path, "gnn_{}_{}".format(args.description, time_stamp))
    result_file_path = os.path.join(save_path, "valid_results.txt")
    config_path = os.path.join(save_path, "config.txt")
    os.mkdir(save_path)

    with open(config_path, 'w') as f:
        args_dict = args.__dict__
        for key in args_dict:
            f.write("{} = {}".format(key, args_dict[key]))
            f.write('\n')
        f.write('\n')
        f.write("train gnn, train_num: {}, valid_num: {}".format(len(graph.train_mask), len(graph.val_mask)))

    summary_writer = SummaryWriter(save_path)

    train(model, graph, x_go, go_mask, ppi_list, loss_fn, optimizer, device,
          result_file_path, summary_writer, save_path,
          batch_size=args.batch_size, epochs=args.epochs, scheduler=scheduler)

    summary_writer.close()


if __name__ == "__main__":
    main()
