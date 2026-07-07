import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv

class net(nn.Module):

    def __init__(self, in_dim, out_dim, out_num=3, model_name='GAT', add_self_loops=False):
        super().__init__()
        self.out_num = out_num
        self.add_self_loops = add_self_loops
        self.model_name = model_name

        # 创建网络池
        if model_name == 'GAT':
            self.conv1 = torch.nn.Linear(in_dim, out_dim)
            self.conv2 = GATConv(in_dim, out_dim, heads=3, concat=False, negative_slope=0.01, dropout=0.2,
                                 add_self_loops=self.add_self_loops)
            self.conv3 = GATConv(in_dim, out_dim, heads=3, concat=False, negative_slope=0.01, dropout=0.2,
                                 add_self_loops=self.add_self_loops)

            self.conv4 = nn.Linear(out_dim, out_dim)
            self.conv5 = nn.Linear(out_dim, out_dim)
            self.conv6 = nn.Linear(out_dim, out_dim)

        elif model_name == 'GCN':
            self.conv1 = torch.nn.Linear(in_dim, out_dim)
            self.conv2 = GCNConv(in_dim, out_dim, add_self_loops=self.add_self_loops)
            self.conv3 = GCNConv(in_dim, out_dim, add_self_loops=self.add_self_loops)

    def find_the_max_and_min(self, x1, x2):
        stacked = torch.stack([x1, x2])
        max_tensor = torch.max(stacked, dim=0)[0]
        min_tensor = torch.min(stacked, dim=0)[0]
        return max_tensor, min_tensor

    def forward(self, x, edge_index1, edge_index2, l_num):
        if l_num < 3:
            if self.model_name == 'GAT':

                x1 = torch.tanh(self.conv1(x))
                x2 = torch.tanh(self.conv2(x, edge_index1))
                x3 = torch.tanh(self.conv3(x, edge_index2))

                max_Difference_x1, min_Difference_x1 = self.find_the_max_and_min(x2 - x1, x3 - x1)
                max_Difference_x2, min_Difference_x2 = self.find_the_max_and_min(x1 - x2, x3 - x2)
                max_Difference_x3, min_Difference_x3 = self.find_the_max_and_min(x2 - x3, x1 - x3)

                max_Difference_x1 = F.normalize(max_Difference_x1, p=2, dim=1)
                max_Difference_x2 = F.normalize(max_Difference_x2, p=2, dim=1)
                max_Difference_x3 = F.normalize(max_Difference_x3, p=2, dim=1)
                min_Difference_x1 = F.normalize(min_Difference_x1, p=2, dim=1)
                min_Difference_x2 = F.normalize(min_Difference_x2, p=2, dim=1)
                min_Difference_x3 = F.normalize(min_Difference_x3, p=2, dim=1)

                min_Difference_x1[min_Difference_x1 > 0.1] = 0
                min_Difference_x2[min_Difference_x2 > 0.1] = 0
                min_Difference_x3[min_Difference_x3 > 0.1] = 0

                max_Difference_x1[max_Difference_x1 < -0.1] = 0
                max_Difference_x2[max_Difference_x2 < -0.1] = 0
                max_Difference_x3[max_Difference_x3 < -0.1] = 0

                Difference_x1 = max_Difference_x1 + min_Difference_x1
                Difference_x2 = max_Difference_x2 + min_Difference_x2
                Difference_x3 = max_Difference_x3 + min_Difference_x3

                cha_yi_x1 = torch.tanh((self.conv4(Difference_x1)))
                cha_yi_x2 = torch.tanh((self.conv5(Difference_x2)))
                cha_yi_x3 = torch.tanh((self.conv6(Difference_x3)))

                x1_out = x1 + cha_yi_x1
                x2_out = x2 + cha_yi_x2
                x3_out = x3 + cha_yi_x3
                return x1_out, x2_out, x3_out
            else:
                x1 = torch.relu(self.conv1(x))
                x2 = torch.relu(self.conv2(x, edge_index1))
                x3 = torch.relu(self.conv3(x, edge_index2))

                cha_yi2 = x2 - x1
                cha_yi3 = x3 - x1
                cha_yi2_3 = x2 - x3
                cha_yi3_2 = x3 - x2

                cha_yi_x1 = torch.sin(cha_yi2_3 ** 3 + cha_yi3_2 ** 3)
                cha_yi_x2 = torch.sin(cha_yi2 ** 3 + cha_yi3_2 ** 3)
                cha_yi_x3 = torch.sin(cha_yi3 ** 3 + cha_yi2_3 ** 3)

                x1_out = x1 + cha_yi_x1
                x2_out = cha_yi_x2 + x2
                x3_out = cha_yi_x3 + x3
                return x1_out, x2_out, x3_out
        if l_num == 3:
            if self.model_name == 'GAT':

                x1 = self.conv1(x)
                x2 = self.conv2(x, edge_index1)
                x3 = self.conv3(x, edge_index2)

                return x1, x2, x3
            else:
                x1 = self.conv1(x)
                x2 = self.conv2(x, edge_index1)
                x3 = self.conv3(x, edge_index2)

                x1_out = x1
                x2_out = x2
                x3_out = x3
                return x1_out, x2_out, x3_out
