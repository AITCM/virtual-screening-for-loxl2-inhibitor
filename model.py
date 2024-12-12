# -*- coding: utf-8 -*-
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.switch_backend('agg')

class Skip_Connection(nn.Module):
    def __init__(self,input_dim,output_dim,act):
        super(Skip_Connection, self).__init__()

        self.act=act()
        self.input_dim=input_dim
        self.output_dim=output_dim
        if input_dim!=output_dim:
            self.fc_1=nn.Linear(input_dim, output_dim)

    def forward(self,input): # input=[X,new_X]
        x,new_X=input
        if self.input_dim!=self.output_dim:
            out=self.fc_1(x)
            x=self.act(out+new_X)
        else:
            x = self.act(x + new_X)
        return x

class Gated_Skip_Connection(nn.Module):
    def __init__(self,input_dim,output_dim,act):
        super(Gated_Skip_Connection, self).__init__()

        self.act = act()
        self.input_dim = input_dim
        self.output_dim = output_dim
        if input_dim != output_dim:
            self.fc_1 = nn.Linear(input_dim, output_dim)
        self.gated_X1=nn.Linear(self.output_dim, self.output_dim)
        self.gated_X2=nn.Linear(self.output_dim, self.output_dim)

    def forward(self,input):
        x,x_new=input
        if self.input_dim != self.output_dim:
            x = self.fc_1(x)
        gate_coefficient = torch.sigmoid(self.gated_X1(x)+self.gated_X2(x_new))
        x=x_new.mul(gate_coefficient)+x.mul((1.0-gate_coefficient))
        return x

class Graph_Conv(nn.Module):
    def __init__(self,input_dim,hidden_dim,act,using_sc):
        super(Graph_Conv, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.fc_hidden=nn.Linear(self.input_dim, self.hidden_dim)
        self.act=act()
        self.using_sc=using_sc

        if (using_sc == 'sc'):
            self.skip_connection=Skip_Connection(self.input_dim,self.hidden_dim,act)
        if (using_sc == 'gsc'):
            self.skip_connection = Gated_Skip_Connection(self.input_dim, self.hidden_dim, act)
        # if (using_sc=="no"):
        #     output_X = act(output_X)
        # self.bn = nn.BatchNorm1d(50)

    def forward(self,input):

        x, A =input
        # print('init_x:',x.shape)
        # print('inti_A:',A.shape)
        x_new = torch.bmm(A, x)
        x_new = self.fc_hidden(x_new)  # [Batch,N,H]

        # print('init_new_x:',x_new.shape)
        # print("x:", x.shape, "A:", A.shape)
        if self.using_sc == "no":
            x = self.act(x_new)
        else:
            x = self.skip_connection((x, x_new))
        # print('out_x:',x.shape)

        return (x, A)

class Readout(nn.Module):
    def __init__(self,input_dim,hidden_dim,act):
        super(Readout, self).__init__()
        self.act = act()
        self.fc_hidden=nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(args.dp_rate) #TODO: Dropout
        self.bn = nn.BatchNorm1d(hidden_dim)
    def forward(self,x):
        x=self.dropout(self.fc_hidden(x)) #TODO: Dropout
        x=torch.sum(x,1)
        x=self.dropout(self.act(self.bn(x)))#TODO: Dropout
        return x


class GCN(nn.Module):
    def __init__(self, args,input_dim):
        super(GCN, self).__init__()
        self.args=args
        self.graph_pre=Graph_Conv(input_dim,args.hidden_size,torch.nn.ReLU,args.using_sc)
        layer_conv=[Graph_Conv(args.hidden_size,args.hidden_size,torch.nn.ReLU,args.using_sc) for i in range(args.num_layer)]
        self.layers=torch.nn.Sequential(*layer_conv)
        self.readout=Readout(args.hidden_size,args.hidden_size2,torch.nn.ReLU)
        self.fc_h1=nn.Linear(args.hidden_size2, args.hidden_size2)
        self.fc_h2=nn.Linear(args.hidden_size2, args.hidden_size2)
        self.fc_pred=nn.Linear(args.hidden_size2, 1)
        self.batch_norm_1 = nn.BatchNorm1d(256)  #TODO: BatchNorm1

        self.dropout = nn.Dropout(args.dp_rate)#TODO: Dropout

    def forward(self,input):
        x,A=input
        x,A = self.graph_pre((x,A))
        x,A=self.layers((x,A))
        # print(x.shape)
        x=self.readout(x)
        x=self.dropout(self.fc_h1(x)) #TODO: Dropout 去掉了
        x=self.fc_h2(self.batch_norm_1(x)) #TODO: BatchNorm1
        Y_pred=self.fc_pred(x).float()
        Y_pred = nn.functional.sigmoid(Y_pred)
        return Y_pred




import argparse
parser = argparse.ArgumentParser(description='GCN for cancer inhibitor identification')
parser.add_argument('-lr', type=float, default=0.001, help='learning rate')
parser.add_argument('-dp_rate', type=float, default=0.3, help='dropout rate')
parser.add_argument('-batch_size', type=int, default=128)
parser.add_argument('-epoch', type=int, default=20)
parser.add_argument('-hidden_size', type=int, default=64, help='first layer')
parser.add_argument('-hidden_size2', type=int, default=256, help='secend layer')
parser.add_argument('-num_layer', type=int, default=6 , help='lstm stack layer number')
parser.add_argument('-cuda', type=bool, default=False)
parser.add_argument('-using_sc', type=str, default="sc")
parser.add_argument('-log-interval', type=int, default=1,)
parser.add_argument('-test-interval', type=int, default=64)
parser.add_argument('-early-stopping', type=int, default=1000)
parser.add_argument('-save-dir', type=str, default='model_dir')
args = parser.parse_args()

model_name = 'gcn_logP_' + str(args.num_layer) + '_' + str(args.hidden_size) + '_' + str(args.hidden_size2) + '_' + str(args.lr) + '_' + args.using_sc