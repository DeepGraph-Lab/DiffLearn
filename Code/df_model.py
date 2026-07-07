import numpy as np
import pandas as pd
import torch_geometric.utils
from torch.nn.parameter import Parameter
import torch
import torch.nn.functional as F
from torch_geometric.utils import dropout_adj, negative_sampling, remove_self_loops, add_self_loops
import torch.nn as nn
import math
from torch_geometric.typing import Adj, OptTensor, PairTensor
import Da_Mv_model



class DiffLearn(torch.nn.Module):
    def __init__(self, data, new_edge, input_dim: int, hidden_dim: int, output_dim: int, drop_p: float = 0.5, drop_edge_p: float = 0.5,):

        super(DiffLearn, self).__init__()
        input_dim = int(input_dim)
        hidden_dim = int(hidden_dim)
        output_dim = int(output_dim)
        self.drop_p = drop_p
        self.drop_edge_p = drop_edge_p
        self.new_edge = new_edge

        self.add_self_loops = False

        self.DF_net1 = Da_Mv_model.net(input_dim, hidden_dim, model_name='GAT')
        self.DF_net2 = Da_Mv_model.net(3 * hidden_dim, output_dim, model_name='GAT')
        self.DF_net3 = Da_Mv_model.net(output_dim, 1, model_name='GCN')


    def forward(self, data ):

        pre = self.df_forward(data, self.drop_edge_p, self.drop_p )

        return pre

    def df_forward(self, data, p_edge, p_features):

        edge_index,_ = dropout_adj(data.edge_list,p=p_edge,
                                        force_undirected=True,
                                        num_nodes=data.x.shape[0],
                                        training=self.training)

        x = F.dropout(data.x[:, :], p=p_features, training=self.training)

        x1, x2, x3= self.DF_net1(x, edge_index, self.new_edge, 1)

        x_1 = torch.cat((x1,x2,x3), 1)
        x4 = F.dropout(x_1, p=p_features, training=self.training)

        x5, x6, x7 = self.DF_net2(x4, edge_index, self.new_edge, 2)
        x_2 = x5 + x6 + x7
        x8 = F.dropout(x_2, p=p_features, training=self.training)

        x9, x10, x11 = self.DF_net3(x8, edge_index, self.new_edge, 3)
        pre = x9 + x10 + x11

        return pre