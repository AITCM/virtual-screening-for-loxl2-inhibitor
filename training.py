# -*- coding: utf-8 -*-
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from sklearn.metrics import roc_auc_score
from utils import convert_to_graph, read_csv, mini_batch
import sys
from sklearn.model_selection import train_test_split
import pandas as pd
import copy
from model import GCN, args


# Default option
if( len(sys.argv) == 6 ):
    # Note that sys.argv[0] is gcn_logP.py
    num_layer = int(sys.argv[1])
    hidden_dim1 = int(sys.argv[2])
    hidden_dim2 = int(sys.argv[3])
    init_lr = float(sys.argv[4])
    using_sc = sys.argv[5]

# dataset
path = r'data/loxl_data.csv'

SMILES, label, split = read_csv(path, mode='train')

smi_train, smi_validation,label_train,label_validation = train_test_split(SMILES, label, test_size=0.4, stratify=split)
print('Train:',len(label_train))
print('Test:',len(label_validation))

#
best_auc = 0
avg_loss = 0
num_atoms = 58  # Max atom size
num_features = 58
total_iter = 0
start_id = 0
start_id_A = 0
step = range(30)
save_model = True


model = GCN(args, input_dim=num_features)

if args.cuda:
    model.cuda()

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
loss_func = torch.nn.BCELoss()
model.train()

def adjust_lr(epoch):
    lrate = args.lr * (0.95 ** np.sum(epoch >= np.array(step)))
    for params_group in optimizer.param_groups:
        params_group['lr'] = lrate
    return lrate

def shuffled(smiles,label):
    shuffled_ix = np.random.permutation(np.arange(len(label)))
    smi_ret = np.array(smiles)[shuffled_ix]
    label_ret = np.array(label)[shuffled_ix]
    return smi_ret, label_ret

for epoch in range(1, args.epoch + 1):

    adjust_lr(epoch)
    train_set_results = []

    for i in range(len(label_train) // args.batch_size):
        smi_batch, label_batch, start_id=mini_batch(smi_train,label_train, start_id, batchsize=args.batch_size)
        X_batch, A_batch = convert_to_graph(smi_batch)
        X_batch, A_batch=torch.tensor(X_batch).float(), torch.tensor(A_batch).float()
        Y_batch = torch.tensor(label_batch).float().unsqueeze(1)
        # print(X_batch.shape,Y_batch.shape)

        if args.cuda:
            X_batch, A_batch ,Y_batch=X_batch.cuda(),A_batch.cuda(),Y_batch.cuda()

        optimizer.zero_grad()
        pred_train = model((X_batch, A_batch))
        pred_train, Y_batch = pred_train.cpu(), Y_batch.cpu()
        loss = loss_func(pred_train, Y_batch)

        pred_train, Y_batch = pred_train.detach().numpy(), Y_batch.detach().numpy()
        auc = roc_auc_score(Y_batch, pred_train)
        sys.stdout.write(
            '\rBatch[{}] - loss: {:.4f}  auc: {:.3f} batch{}'.format(total_iter, loss.item(),
                                                                            auc, args.batch_size))
        loss.backward()

        nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=5.0)
        optimizer.step()
        total_iter += 1

        if total_iter % args.test_interval == 0:
            model.eval()
            print("!!evaluation")
            X_eval, A_eval = convert_to_graph(smi_validation)
            Y_eval = label_validation
            X_eval, A_eval = torch.tensor(X_eval).float(), torch.tensor(A_eval).float()
            Y_eval = torch.tensor(Y_eval).float()
            if args.cuda:
                X_eval, A_eval, Y_eval = X_eval.cuda(), A_eval.cuda(), Y_eval.cuda()
            pred_eval = model((X_eval, A_eval))
            # print('results:',pred_eval)
            pred_eval, Y_eval = pred_eval.cpu(), Y_eval.cpu()
            Y_eval = Y_eval.unsqueeze(1)
            # Y_eval_one_hot = torch.zeros(len(Y_eval.numpy()), 2).scatter_(1, Y_eval.long(), 1)

            loss_eval = loss_func(pred_eval, Y_eval)
            eval_loss = loss_eval.item()
            pred_eval, Y_eval = pred_eval.detach().numpy(), Y_eval.detach().numpy()

            eval_auc = roc_auc_score(Y_eval, pred_eval)

            print('\nEvaluation - loss: {:.6f} auc: {:.4f} '.format(eval_loss, eval_auc))

            if eval_auc > best_auc:
                best_auc = eval_auc
                best_dict = copy.deepcopy(model.state_dict())
                params = [param.cpu().detach().numpy() for name, param in model.fc_h2.named_parameters()]
                pred_results = np.array(pred_eval).reshape(-1, 1)
                Y_results = np.array(Y_eval).reshape(-1, 1)
                # results = np.hstack((pred_results, Y_results))
                # df = pd.DataFrame(results, columns=['pred', 'true'])
                if save_model:
                    torch.save(model.state_dict(), 'saved_model.pth')

            print('best- Evaluation_ acc: {:.4f} \n'.format(best_auc))
            model.train()

