import numpy as np
import pandas as pd
import random
import torch
import torch.nn.functional as F
import time
from sklearn import metrics
import pickle
import os
from utils import parse_args, create_train_dir, create_model_dir, write_hyper_params
from torch_geometric.utils import dropout_adj, dropout_adj, negative_sampling, remove_self_loops, add_self_loops
import df_model

def train(data, train_mask, p_train):

    model.train()
    optimizer.zero_grad()

    pre = model(data)
    output = pre[train_mask]
    Y = data.y[train_mask]
    loss = F.binary_cross_entropy_with_logits(output, Y)

    print(f'train loss {loss}')
    loss.backward()
    optimizer.step()



def test(data, test_mask):
    model.eval()
    with torch.no_grad():
        pre = model(data)
        output = pre[test_mask]
        Y = data.y[test_mask]
        auc, aupr, = quality_index(output, Y)

    return auc, aupr, pre

def quality_index(output, Y):

    pred = torch.sigmoid(output).cpu().detach().numpy()
    Yn = Y.cpu().numpy()

    precision, recall, _thresholds = metrics.precision_recall_curve(Yn, pred)
    aupr = metrics.auc(recall, precision)

    auc = metrics.roc_auc_score(Yn, pred)

    return auc, aupr


args = parse_args()
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
seed = args.seed
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)


EPOCH = args.epochs
cv_num = args.cv_runs
num = args.num

AUC = np.zeros(shape=(num, cv_num))
AUPR = np.zeros(shape=(num, cv_num))

cancer = args.cancer

data = torch.load('./data/' + cancer + '_pan_cancer.pkl')


BioGRID_edge = []

with open(f'net_data/{cancer}_BioGRID_edge.txt', 'r') as file:
    for line in file:
        # 分割每行的两个节点ID
        parts = line.strip().split()
        if len(parts) >= 2:
            src = int(parts[0])
            dst = int(parts[1])
            BioGRID_edge.append([src, dst])

BioGRID_edge = torch.tensor(BioGRID_edge, dtype=torch.long).t().contiguous()


BioGRID_edge = BioGRID_edge.to(device)
data = data.to(device)
k_sets = data.k_sets

time_start = time.time()
for i in range(num):
    root_dir = './result/'
    num_path = create_train_dir(root_dir, i)

    MAX_epoch1 = []
    MAX_epoch2 = []
    MAX_AUC_tr = []
    MAX_AUPR_tr = []

    for cv_run in range(cv_num):
        stop_symbol_in = False
        stop_symbol = False
        print(cv_run)
        epoch_list = []
        epoch_list_auc = []
        pre_all = []
        p_train, p_test, tr_mask, te_mask = k_sets[i][cv_run]

        cv_path = create_model_dir(num_path, cv_run)
        model = df_model.DiffLearn(data, new_edge=BioGRID_edge , input_dim=data.x.shape[1], hidden_dim=args.hidden_dims[0], output_dim=args.hidden_dims[1],
                    drop_p=args.dropout, drop_edge_p=args.dropout_edge).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

        AUC_tr = []
        AUPR_tr = []

        Metrics_Log_AUC = []
        Metrics_Log_AUPR = []

        MAX_auc = []
        MAX_aupr = []

        for epoch in range(0, EPOCH):
            print(epoch)
            train(data, tr_mask, p_train)
            auc, aupr, max_pred = test(data, te_mask)
            print(f'auc:{auc},aupr:{aupr}')
            AUC_tr.append((auc, epoch))
            AUPR_tr.append((aupr, epoch))
            max_pred_num, _ = max(AUPR_tr, key=lambda x: x[0])

            if not (aupr < max_pred_num):
                pred = torch.sigmoid(max_pred).cpu().detach().numpy()
                pd.DataFrame(pred).to_csv(f'net_data/{cancer}/max_pred/predict' + '_' + str(cv_run) + '.txt')
                print(f'Usage of Files for Cancer Driver Gene Prediction auc:{auc},aupr:{aupr},epoch:{epoch}')


        max_AUC_value = max(AUC_tr, key=lambda x: x[0])
        max_auc, max_epoch1 = max_AUC_value
        max_AUPR_value = max(AUPR_tr, key=lambda x: x[0])
        max_aupr, max_epoch2 = max_AUPR_value
        MAX_AUC_tr.append(max_auc)
        MAX_AUPR_tr.append(max_aupr)
        MAX_epoch1.append(max_epoch1)
        MAX_epoch2.append(max_epoch2)
        print(max_AUC_value)
        print(max_AUPR_value)

        torch.save(model, cv_path + 'model' + '_' + str(cv_run) + '.pkl')
        auc, aupr, x = test(data, te_mask)
        print(f' test auc {auc}, test aupr {aupr}')
        AUC[i][cv_run] = max_auc
        AUPR[i][cv_run] = max_aupr

        pred = torch.sigmoid(x).cpu().detach().numpy()
        pd.DataFrame(pred).to_csv(cv_path + 'predict' + '_' + str(cv_run) + '.txt')
    print(MAX_AUC_tr)
    print(MAX_epoch1)
    print(MAX_AUPR_tr)
    print(MAX_epoch2)
    print('Max mean AUC:', np.mean(MAX_AUC_tr))
    print('Max mean AUPR:', np.mean(MAX_AUPR_tr))
    print('mean auc epoch:', np.mean(MAX_epoch1))
    print('mean aupr epoch:', np.mean(MAX_epoch2))


print(time.time() - time_start)
print(AUC.mean())
print(AUC.mean(1).std())
print(AUPR.mean())
print(AUPR.mean(1).std())

with open('result/auc.pkl', 'wb') as fo:
    pickle.dump(AUC, fo)
with open('result/au_pr.pkl', 'wb') as fo:
    pickle.dump(AUPR, fo)

write_hyper_params(vars(args), os.path.join('./result/', 'hyper_params.txt'))
