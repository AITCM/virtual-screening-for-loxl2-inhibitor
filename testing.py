import warnings
warnings.filterwarnings(action='ignore')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import read_csv, convert_to_graph
import sys
from model import GCN, args


if( len(sys.argv) == 6 ):
    # Note that sys.argv[0] is gcn_logP.py
    num_layer = int(sys.argv[1])
    hidden_dim1 = int(sys.argv[2])
    hidden_dim2 = int(sys.argv[3])
    init_lr = float(sys.argv[4])
    using_sc = sys.argv[5]

path = r'data\tcm-screen.csv'
SMILES, label, CID = read_csv(path, mode='test')

num_features=58

total_time = 0.0
start_id=0
step=range(50)

model=GCN(args,input_dim=num_features)
# model.load_state_dict(torch.load(path_model))

path_model = r'saved_model.pth'
load_pred_model = True
pred_dict = torch.load(path_model,map_location='cpu')
pre_param_keys = [k for k,v in pred_dict.items()]
model_param_keys = [k for k,v in model.named_parameters()]
frozen_filter = pre_param_keys
if load_pred_model:
    model.load_state_dict(pred_dict, strict=False)

if args.cuda:
    model.cuda()

optimizer = torch.optim.Adam( model.parameters(), lr=args.lr)
loss_func = torch.nn.MSELoss()
model.train()

def adjust_lr(epoch):
    lrate = args.lr * (0.95 ** np.sum(epoch >= np.array(step)))
    for params_group in optimizer.param_groups:
        params_group['lr'] = lrate
    return lrate

def Eval():
    f = open(r'results/output.txt', 'w+')
    model.eval()
    X_eval, A_eval = convert_to_graph(SMILES)
    Y_eval = label
    X_eval, A_eval = torch.tensor(X_eval).float(), torch.tensor(A_eval).float()
    Y_eval = torch.tensor(Y_eval).float()
    if args.cuda:
        X_eval, A_eval, Y_eval = X_eval.cuda(), A_eval.cuda(), Y_eval.cuda()
    pred_eval = model((X_eval, A_eval))
    pred_eval = F.sigmoid(pred_eval)
    pred_eval = pred_eval.cpu()
    pred_eval = pred_eval.detach().numpy()
    for score, cid in zip(pred_eval, CID):
        print(score, cid)
        # if p<0.5:
        f.write(str(score)+'$')
        f.write(str(cid))
        f.write('\n')

if __name__ == '__main__':
    Eval()
