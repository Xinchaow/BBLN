import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GATConv
from torch_geometric.nn import GCNConv, JumpingKnowledge, global_mean_pool, SAGEConv


class ASE(torch.nn.Module):

    def __init__(self, in_feature, in_len):
        super(ASE, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=in_feature, out_channels=1, kernel_size=3, padding=0)
        self.bn1 = nn.BatchNorm1d(1)
        self.biGRU = nn.GRU(1, 1, bidirectional=True, batch_first=True, num_layers=1)
        self.maxpool1d = nn.MaxPool1d(3, stride=3)
        self.global_avgpool1d = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(math.floor(in_len / 3), 512)  # 原版

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1d(x)
        x = self.bn1(x)
        x = self.maxpool1d(x)
        x = x.transpose(1, 2)
        x, _ = self.biGRU(x)
        x = self.global_avgpool1d(x)
        x = x.squeeze()
        x = self.fc1(x)
        return x


class GSE(torch.nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        super(GSE, self).__init__()

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout,
                                                   dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.linear = nn.Linear(d_model, 512)

    def forward(self, emb_proteinA, protA_mask):
        memory = self.transformer_encoder(emb_proteinA, src_key_padding_mask=protA_mask)

        output = memory.permute(1, 0, 2)
        output_c = torch.linalg.norm(output, dim=2)
        output_c = F.softmax(output_c, dim=1).unsqueeze(1)
        output = torch.bmm(output_c, output)

        return self.linear(output).squeeze(1)


class BBLN(torch.nn.Module):
    def __init__(self, seq_in_len=2000, seq_in_feature=13, gin_in_feature=512, num_layers=1,
                 hidden=512, use_jk=False, train_eps=True,
                 feature_fusion=None, class_num=7):
        super(BBLN, self).__init__()
        self.use_jk = use_jk
        self.train_eps = train_eps
        self.feature_fusion = feature_fusion
        
        self.GSE = GSE(64, 2, 2, 64, 0.2)
        self.ASE = ASE(seq_in_feature, seq_in_len)

        self.gin_conv1 = GINConv(
            nn.Sequential(
                nn.Linear(gin_in_feature, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.BatchNorm1d(hidden),
            ), train_eps=self.train_eps
        )

        self.gin_conv1_GO = GINConv(
            nn.Sequential(
                nn.Linear(gin_in_feature, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.BatchNorm1d(hidden),
            ), train_eps=self.train_eps
        )

        if self.use_jk:
            mode = 'cat'
            self.jump = JumpingKnowledge(mode)
            self.lin1 = nn.Linear(num_layers * hidden, hidden)
            self.lin1_go = nn.Linear(num_layers * hidden, hidden)
        else:
            self.lin1 = nn.Linear(hidden, hidden)
            self.lin1_go = nn.Linear(hidden, hidden)
        self.lin2 = nn.Linear(hidden, hidden)
        self.lin2_go = nn.Linear(hidden, hidden)

        self.mlp_cl = nn.Sequential(nn.Linear(512, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(), nn.BatchNorm1d(hidden))
        
        self.fca = nn.Linear(512*2, 512)
        self.fc3 = nn.Linear(512, class_num)

        self.eps = nn.Parameter(torch.Tensor([1.0]))
        self.eps1 = nn.Parameter(torch.Tensor([1.0]))
        self.bn_fuse = nn.BatchNorm1d(512)


    def forward(self, x_seq, x_GO, x_mask, edge_index, train_edge_id, p=0.5):
        x_seq = self.ASE(x_seq)

        x_GO = x_GO.permute(1, 0, 2)
        x_GO = self.GSE(x_GO, x_mask)

        x_seq_cl = x_seq
        x_GO_cl = x_GO
        
        x_seq_gin = x_seq
        x_GO_gin = x_GO

        x_seq_gin = self.gin_conv1(x_seq_gin, edge_index)
        x_GO_gin = self.gin_conv1_GO(x_GO_gin, edge_index)

        x_seq_gin = F.relu(self.lin1(x_seq_gin))
        x_seq_gin = F.dropout(x_seq_gin, p=p, training=self.training)
        x_seq_gin = self.lin2(x_seq_gin)
        x_GO_gin = F.relu(self.lin1_go(x_GO_gin))
        x_GO_gin = F.dropout(x_GO_gin, p=p, training=self.training)
        x_GO_gin = self.lin2_go(x_GO_gin)

        node_id = edge_index[:, train_edge_id]
        
        x1_seq_gin = x_seq_gin[node_id[0]]
        x2_seq_gin = x_seq_gin[node_id[1]]
        x1_GO_gin = x_GO_gin[node_id[0]]
        x2_GO_gin = x_GO_gin[node_id[1]]

        x1_seq_cl = x_seq_cl[node_id[0]]
        x2_seq_cl = x_seq_cl[node_id[1]]
        x1_GO_cl = x_GO_cl[node_id[0]]
        x2_GO_cl = x_GO_cl[node_id[1]]

        x_seq_ppi = torch.mul(x1_seq_gin, x2_seq_gin)
        x_GO_ppi = torch.mul(x1_GO_gin, x2_GO_gin)
        
        x_seq_cl_ppi = torch.mul(x1_seq_cl, x2_seq_cl)
        x_GO_cl_ppi = torch.mul(x1_GO_cl, x2_GO_cl)

        x_seq_cl_ppi = self.mlp_cl(x_seq_cl_ppi)
        x_GO_cl_ppi = self.mlp_cl(x_GO_cl_ppi)
        
        x_cl_modal1 = x_seq_cl_ppi.clone().detach()
        x_cl_modal2 = x_GO_cl_ppi.clone().detach()

        x1 = torch.cat([x_seq_ppi, x_GO_ppi], dim=1)
        x2 = torch.cat([x_cl_modal1, x_cl_modal2], dim=1)

        x = self.eps*(x1) + self.eps1*(x2)
 
        x = self.fca(x)
        x = self.fc3(x)

        return x, x_seq_cl_ppi, x_GO_cl_ppi


