from __future__ import print_function
import time
import csv
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
# import tensorflow as tf
import warnings
import cmath
import argparse
import glob
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='./model/complex-1.25-t-v4.model',required=False)
parser.add_argument('--data', default='./cxdatabase/withoutX',required=False)
parser.add_argument('--output_dir', default='./',required=False)
parser.add_argument('--test_num', default=-1,required=False,help='Enter -1 to test on whole testset,or enter the real number')
args = parser.parse_args()


h_sp = 0
h_sn = 0
h_mcc = 0
h_acc = 0
warnings.filterwarnings('ignore')

learning_rate = 1e-4
Max_length = 100

test_num = args.test_num
# sample_num = 10
# test_num = 15
#sample_num=1260
#test_num=1285


epoch_num =1
import torch.utils.data as data
import torch

batchsize = 1
lamda = 0.1
import datetime

print(datetime.datetime.now())
np.set_printoptions(threshold=np.inf)
# 读入pssm-csv

# pssm测试集
print('Loading pssm test data...')
# cwd = os.getcwd()
read_path = args.data+'/pssm(peptide)'
# os.chdir(read_path)
csv_name_list=glob.glob(read_path+'/*')
# csv_name_list=glob.glob('/Users/bytedance/Desktop/mvirt/cxdatabase/withoutX/pssm(peptide)/*')
# csv_name_list = os.listdir()
csv_name_list.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
protein = pd.read_csv(csv_name_list[0])
if test_num==-1:
    test_num=len(csv_name_list)
else:
    pass
for i in tqdm(range(test_num)):
    protein = pd.read_csv(csv_name_list[i])
    seqt = pd.read_csv(csv_name_list[i],usecols=[1]).transpose()
    seqt = seqt.values.tolist()
    seqt=sum(seqt,[])
    # print(len(seq))
    x = protein[['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']]
    y = protein['HEC']
    y = list(y)
    for k in range(0, len(y)):
        y[k] = ord(y[k]) - 65  # 从0开始
        if y[k] == 7:
            y[k] = 1;
        elif y[k] == 4:
            y[k] = 2;
        elif y[k] == 2:
            y[k] = 3;
    for j in range(0, len(seqt)):
        seqt[j] = ord(seqt[j]) - 64  # 从0开始
        if seqt[j] == 25:
            seqt[j] = seqt[j] - 5
        elif 22 <= seqt[j] <= 23:
            seqt[j] = seqt[j] - 4

        elif 16 <= seqt[j] <= 20:
            seqt[j] = seqt[j] - 3

        elif 11 <= seqt[j] <= 14:
            seqt[j] = seqt[j] - 2

        elif 3 <= seqt[j] <= 9:
            seqt[j] = seqt[j] - 1
        elif seqt[j] == 1:
            seqt[j] = seqt[j]
    classes = 3
    reallength = [len(y)]
    # print('seq',seq)

    if (len(x) > Max_length):
        x = x[:Max_length]
    x = np.array(x)
    x = torch.from_numpy(x)
    # x = torch.unsqueeze(x,0)
    # x = torch.unsqueeze(x,0).float()
    # m = nn.ReflectionPad2d((0, 0, 0, 1))
    # x = m(x)
    # x = x.squeeze()
    # print("pssm x", i)
    seqt=np.array(seqt)
    if (len(seqt) < Max_length + 1):
        seqt = np.pad(seqt, (0, Max_length - len(seqt)), 'constant')
    if (len(x) < Max_length + 1):
        x = np.pad(x, ((0, Max_length - len(x)), (0, 0)), 'constant')
    if (len(y) > Max_length):
        y = y[:Max_length]
    if (len(seqt) > Max_length):
        seqt =seqt[:Max_length]
    y = np.array(y)  # y是数组类型的标签
    y = torch.from_numpy(y);
    y = torch.unsqueeze(y, dim=1)
    y = y.type(torch.LongTensor)
    y = torch.zeros(Max_length, 4).scatter_(1, y, 1)
    # # print(y.shape)
    # y = torch.unsqueeze(y,0)
    # y = torch.unsqueeze(y,0).float()
    # y = m(y)
    # y = y.squeeze()
    # # print("pssm y", i)
    # if (len(y) < Max_length+1):
    #     y = np.pad(y, ((0, Max_length +1- len(y)), (0, 0)), 'constant')
    # print(y.shape)
    if (i == 0):
        x1t = x
        y1t = y
        reallength1t = reallength
        seq1t=seqt
    if (i != 0):
        x1t = np.concatenate((x1t, x), axis=0)
        y1t = np.concatenate((y1t, y), axis=0)
        reallength1t = np.concatenate((reallength1t, reallength), axis=0)
        seq1t=np.concatenate((seq1t,seqt),axis=0)
    # print('pssm:%d' % i)
    # print('length:%d' % (len(x1)))
# print(x1t.shape)
x1t = torch.from_numpy(x1t);
x1t = x1t.view(test_num , Max_length, 20)
# print("x1t",x1t.shape)
y1t = np.delete(y1t, 0, axis=1)
y1t = torch.from_numpy(y1t);
y1t = y1t.view(test_num , Max_length, 3)
seqt=torch.from_numpy(seq1t)
seqt = seqt.view(test_num , Max_length)
# print("y1t",y1t.shape)
# print("reallengtht",reallength1t)
print('Loading pssm test data over')

# 读入hmm-csv
# hmm训练集

# hmm测试集
print('Loading hmm test data...')
# cwd = os.getcwd()
read_path = args.data+'/hmm(peptide)'
# os.chdir(read_path)
csv_name_list=glob.glob((read_path+'/*'))
# csv_name_list = os.listdir()
csv_name_list.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
protein = pd.read_csv(csv_name_list[0])
for i in tqdm(range(0, test_num)):
    protein = pd.read_csv(csv_name_list[i])
    x = protein[['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']]
    if len(x) > Max_length:
        x = x[:Max_length]
    x = np.array(x)
    x = torch.from_numpy(x)
    # x = torch.unsqueeze(x, 0)
    # x = torch.unsqueeze(x, 0).float()
    # m = nn.ReflectionPad2d((0, 0, 0, 1))
    # x = m(x)
    # x = x.squeeze()
    # print("pssm x", i)
    if (len(x) < Max_length + 1):
        x = np.pad(x, ((0, Max_length - len(x)), (0, 0)), 'constant')
    if (i == 0):
        x2t = x
    if (i != 0):
        x2t = np.concatenate((x2t, x), axis=0)
        # print('hmm : %d' % i)
# print(x2t.shape)
x2t = torch.from_numpy(x2t);
x2t = x2t.view(test_num , Max_length, 20)
# print("x2t", x2t.shape)
print('Loading hmm test data over')

# cwd = os.getcwd()
read_path = args.model
# os.chdir(read_path)


def cosine(x, y):
    y=torch.transpose(y,1,2)
    value = torch.matmul(x, y)
    n = torch.norm(input=value, p=float('inf'))
    return value / n


class bigru_attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim):
        super(bigru_attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.bigru = nn.LSTM(input_dim, hidden_dim, num_layers=layer_dim, bidirectional=True, batch_first=True)
        self.weight_W = nn.Parameter(torch.Tensor(self.hidden_dim * 2, self.hidden_dim * 2))
        self.weight_proj = nn.Parameter(torch.Tensor(self.hidden_dim * 2, 1))
        self.fc = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2)
        nn.init.uniform_(self.weight_W, -0.1, 0.1)
        nn.init.uniform_(self.weight_proj, -0.1, 0.1)

    def forward(self, x, trainstate=True):
        gru_out, _ = self.bigru(x)
        u = torch.tanh(torch.matmul(gru_out, self.weight_W))
        att = torch.matmul(u, self.weight_proj)
        att_score = F.softmax(att, dim=1)

        scored_x = gru_out * att_score

        y = self.fc(scored_x)
        return y


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.restriction = torch.zeros([batchsize, Max_length, 3])
        self.embedding = nn.Embedding(25, 20)
        self.fnn1_1 = nn.Linear(in_features=240, out_features=512)

        self.padding1d = nn.ReflectionPad1d((1, 1))
        self.conv1d = nn.Conv1d(in_channels=Max_length, out_channels=Max_length, kernel_size=3, stride=1)
        self.pooling1d = nn.MaxPool1d(kernel_size=3, stride=1)

        # self.unsqueeze=torch.unsqueeze(self,dim=1)
        self.padding2d = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv2d_1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.conv2d_2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.conv2d_3 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.padding2d1_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.pooling2d = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        # self.squeeze=torch.squeeze()
        # self.attgru1_1=bigru_attention(input_dim=128,hidden_dim=128,layer_dim=1)
        # self.attgru2_1 = bigru_attention(input_dim=256, hidden_dim=256, layer_dim=1)
        # self.attgru1_2 = bigru_attention(input_dim=128, hidden_dim=128, layer_dim=1)
        # self.attgru2_2 = bigru_attention(input_dim=256, hidden_dim=256, layer_dim=1)
        # self.attgru1_3 = bigru_attention(input_dim=128, hidden_dim=128, layer_dim=1)
        # self.attgru2_3 = bigru_attention(input_dim=256, hidden_dim=256, layer_dim=1)

        self.bgru1 = nn.LSTM(input_size=512, hidden_size=512, num_layers=1, bidirectional=True, batch_first=True)


        self.replayer1 = nn.Linear(in_features=1024, out_features=3)
        # self.replayer2=nn.Linear(in_features=128,out_features=3)
        self.fnn4 = nn.Linear(in_features=1024, out_features=512)

        self.bgru2 = nn.LSTM(input_size=512, hidden_size=512, num_layers=1, bidirectional=True, batch_first=True)


        self.fnn3_1 = nn.Linear(in_features=1024, out_features=512)
        self.fnn3_2 = nn.Linear(in_features=512, out_features=128)
        self.fnn3_3 = nn.Linear(in_features=128, out_features=3)

        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=2)
        self.batchnorm = nn.BatchNorm2d(num_features=1)

        # self.transformer_layer=nn.TransformerEncoderLayer()
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=6)

    def forward(self, pssm, hmm, seq, trainstate=True):
        seq = self.embedding(seq.long())
        ps = cosine(pssm, seq)
        # hs= cosine(pssm, torch.transpose(seq, 1, 2))
        # print('ph',ph.shape)
        hs = cosine(hmm, seq)

        x = torch.cat((pssm, hmm, ps, hs), 2)

        x = self.fnn1_1(x)
        x = F.dropout(input=x, training=trainstate)
        x = self.relu(x)

        x = torch.unsqueeze(input=x, dim=0)
        x = self.batchnorm(x)

        x = self.conv2d_1(x)
        x = self.batchnorm(x)
        x = self.relu(x)

        x = self.conv2d_2(x)
        x = self.batchnorm(x)
        x = self.relu(x)

        x = self.conv2d_3(x)
        x = self.relu(x)

        x = self.pooling2d(x)
        x = self.relu(x)
        x = self.batchnorm(x)
        x = torch.squeeze(x, 1)

        # x3 = x[:, 60:100, :]
        # x1 = x[:, 0:50, :]
        # x2 = x[:, 50:100, :]
        # x1 = self.attgru1_1(x1)
        # x2 = self.attgru1_2(x2)
        # x3 = self.attgru1_3(x3)
        # x = torch.cat([x1, x2,x3], dim=1)
        x, _ = self.bgru1(x)

        # x = torch.cat([x1, x2], dim=1)
        x = torch.unsqueeze(input=x, dim=0)
        x = self.batchnorm(x)
        x = torch.squeeze(input=x, dim=0)
        x = self.relu(x)

        r = self.replayer1(x)
        # r=self.relu(r)
        # r=self.replayer2(r)

        r = self.softmax(r)
        self.restriction = r
        x = self.fnn4(x)

        # x1 = x[:, 0:50, :]
        # x2 = x[:, 50:100, :]
        x, _ = self.bgru2(x)
        # x2, _ = self.bgru2_2(x2)
        # x = torch.cat([x1, x2], dim=1)

        x = torch.unsqueeze(input=x, dim=0)
        x = self.batchnorm(x)
        x = torch.squeeze(input=x, dim=0)
        x = self.relu(x)

        # x1 = x[:, 0:30, :]
        # x2 = x[:, 30:60, :]
        # x3 = x[:, 60:100, :]
        # x1 = self.attgru2_1(x1)
        # x2 = self.attgru2_2(x2)
        # x3 = self.attgru2_3(x3)
        # x = torch.cat([x1, x2, x3], dim=1)

        x = self.fnn3_1(x)

        x = self.transformer(x)

        x = self.relu(x)
        x = self.fnn3_2(x)
        x = self.fnn3_3(x)

        x = self.softmax(x)
        # x = self.final(x)
        # x = self.softmax(x)
        return x

    def get_rep_layer(self):
        return self.restriction


class restricted_loss(nn.Module):
    def __init__(self):
        super().__init__()
        # self.crossentropy=nn.CrossEntropyLoss()

    def forward(self, output, tar, rep, who):
        # loss1=self.crossentropy(torch.sigmoid(output),torch.sigmoid(target).long())
        # loss2= self.crossentropy(torch.sigmoid(output), torch.sigmoid(target).long())
        # print('output1', output)
        w = torch.tensor([1., 1.25, 1.]).to(device)
        tar = tar * w
        # output=output*w
        # print('output2',output)
        loss1 = F.mse_loss(output.float(), tar.float(), reduction='mean')
        loss2 = F.mse_loss(rep.float(), tar.float(), reduction='mean')
        # loss2 = self.crossentropy(output, target.long())

        # loss1=F.mse_loss(input=output.float(),target=target.float(),reduction='mean')
        # loss2=F.mse_loss(input=rep.float(),target=target.float(),reduction='mean')
        loss = loss1 + lamda * loss2
        # print('-----------------------loss','for',who,'------------------------------------------')
        # print(datetime.datetime.now())
        # print('loss1:',loss1.cpu())
        # print('rep loss:',loss2.cpu())
        # print('-----------------------loss','for',who,'------------------------------------------')
        return loss


bgru1 = 0.0

def accuracy(x_1,x_2,x_3,i,j,reallength1,label,epoch,who):
    with torch.no_grad():
        x1=x_1.detach().numpy()
        x2=x_2.detach().numpy()
        x3=x_3.detach().numpy()


        # total number of real H,E,C
        H_count=0.0
        E_count=0.0
        C_count=0.0
        # predict right of H,E,C
        H=0
        E=0
        C=0
        #
        def likely(x1,x2,x3):
            HEC=1
            temp=0
            if(x1>x2):
                temp=x1
                HEC=1
            elif(x1<x2):
                temp=x2
                HEC=2
            if(temp>x3):
                temp=temp
                HEC=HEC
            elif(temp<x3):
                temp=x3
                HEC=3
            return temp,HEC




        H1 = 0
        E1 = 0
        C1 = 0
        H2 = 0
        E2 = 0
        C2 = 0
        H3 = 0
        E3 = 0
        C3 = 0





        H1 = E1 = C1 = 0
        for k in range(0, reallength1[i * 1 + j]):
            if (label[j][k][0] == 1):
                H1 = H1 + 1
            if (label[j][k][1] == 1):
                E1 = E1 + 1
            if (label[j][k][2] == 1):
                C1 = C1 + 1
        H2 = E2 = C2 = 0
        for k in range(0, reallength1[i * 1 + j]):
            _, kind = likely(x1[k], x2[k], x3[k])
            if (kind == 1):
                H2 = H2 + 1
            if (kind == 2):
                E2 = E2 + 1
            if (kind == 3):
                C2 = C2 + 1

        def sov(x_1, x_2, x_3, i, j, reallength1, label):
            with torch.no_grad():
                x1 = x_1.detach().numpy()
                x2 = x_2.detach().numpy()
                x3 = x_3.detach().numpy()
                # total number of real H,E,C
                H_count = 0.0
                E_count = 0.0
                C_count = 0.0

                H=0
                E=0
                C=0
                H1 = 0
                E1 = 0
                C1 = 0
                H2 = 0
                E2 = 0
                C2 = 0


                flag=0
                sovh=0
                sove=0
                sovc=0

                for k in range(0, reallength1[i * 1 + j]):
                    if (label[j][k][0] == 1):
                        H_count = H_count + 1
                    if (label[j][k][1] == 1):
                        E_count = E_count + 1
                    if (label[j][k][2] == 1):
                        C_count = C_count + 1
                for k in range(0, reallength1[i * 1 + j]):
                    _, kind = likely(x1[k], x2[k], x3[k])
                    if (label[j][k][0] == 1 and kind == 1):
                        H = H + 1
                    if (label[j][k][1] == 1 and kind == 2):
                        E = E + 1
                    if (label[j][k][2] == 1 and kind == 3):
                        C = C + 1
                H1 = E1 = C1 = 0
                for k in range(0, reallength1[i * 1 + j]):
                    if (label[j][k][0] == 1):
                        H1 = H1 + 1
                    if (label[j][k][1] == 1):
                        E1 = E1 + 1
                    if (label[j][k][2] == 1):
                        C1 = C1 + 1
                H2 = E2 = C2 = 0
                for k in range(0, reallength1[i * 1 + j]):
                    _, kind = likely(x1[k], x2[k], x3[k])
                    if (kind == 1):
                        H2 = H2 + 1
                    if (kind == 2):
                        E2 = E2 + 1
                    if (kind == 3):
                        C2 = C2 + 1
                length = float(H_count + E_count + C_count)
                mini = H + E + C
                maxh = H1 + H2 - H
                maxe = E1 + E2 - E
                maxc = C1 + C2 - C
                max = maxh + maxe + maxc
                dec = min(maxc - C, C, 0.5 * C_count, 0.5 * C)
                deh = min(maxh - H, H, 0.5 * H_count, 0.5 * H)
                dee = min(maxe - E, E, 0.5 * E_count, 0.5 * E)
                de = min(max - mini, mini, 0.5 * (C_count + H_count + E_count), 0.5 * mini)
                if (maxe != 0):
                    sove = ((E + dee) / maxe)
                else:
                    flag = 1
                if (maxh != 0):
                    sovh = ((H + deh) / maxh)
                else:
                    flag = 1
                if (maxc != 0):
                    sovc = ((C + dec) / maxc)
                else:
                    flag = 1

                sov = (mini + de) / max
            return sovh, sove, sovc, sov,flag



        for k in range(0,reallength1[i*batchsize+j]):
            if(label[j][k][0]==1):

                H_count=H_count+1
            if(label[j][k][1]==1):
                E_count=E_count+1
            if(label[j][k][2]==1):
                C_count=C_count+1
        for k in range(0,reallength1[i*batchsize+j]):
            # print("x1[k]",x1[k])
            _,kind=likely(x1[k],x2[k],x3[k])
            if (label[j][k][0] == 1 and kind==1):
                H = H + 1
            if (label[j][k][1] == 1 and kind==2):
                E = E + 1
            if (label[j][k][2] == 1 and kind==3):
                C = C + 1

        TP = 0
        FP = 0
        TN = 0
        FN = 0
        H_TP = 0
        H_FP = 0
        H_TN = 0
        H_FN = 0
        E_TP = 0
        E_FP = 0
        E_TN = 0
        E_FN = 0
        C_TP = 0
        C_FP = 0
        C_TN = 0
        C_FN = 0
        H_H=0
        H_E=0
        H_C=0
        E_H = 0
        E_E = 0
        E_C = 0
        C_H = 0
        C_E = 0
        C_C = 0
        for k in range(0, reallength1[i * batchsize + j]):
            # print("x1[k]",x1[k])
            _, kind = likely(x1[k], x2[k], x3[k])
            if (label[j][k][0] == 1 and kind == 1):
                H_H=H_H+1
                H_TP = H_TP + 1
            if (label[j][k][0] != 1 and kind == 1):
                H_FP = H_FP + 1

            if (label[j][k][0] != 1 and kind != 1):
                H_TN = H_TN + 1
            if (label[j][k][0] == 1 and kind != 1):
                H_FN = H_FN + 1
                if (kind == 2):
                    H_E = H_E + 1
                if (kind == 3):
                    H_C = H_C + 1


            if (label[j][k][1] == 1 and kind == 2):
                E_TP = E_TP + 1
                E_E=E_E+1
            if (label[j][k][1] != 1 and kind == 2):
                E_FP = E_FP + 1
            if (label[j][k][1] != 1 and kind != 2):
                E_TN = E_TN + 1
            if (label[j][k][1] == 1 and kind != 2):
                E_FN = E_FN + 1
                if (kind == 1):
                    E_H = E_H + 1
                if (kind == 3):
                    E_C= E_C + 1

            if (label[j][k][2] == 1 and kind == 3):
                C_TP = C_TP + 1
                C_C=C_C+1
            if (label[j][k][2] != 1 and kind == 3):
                C_FP = C_FP + 1
            if (label[j][k][2] != 1 and kind != 3):
                C_TN = C_TN + 1
            if (label[j][k][2] == 1 and kind != 3):
                C_FN = C_FN + 1
                if (kind == 1):
                    C_H = C_H + 1
                if (kind == 2):
                    C_E = C_E + 1
#---------------------------------------------------------------------------------------
        sov_h, sov_e, sov_c, sov_overall, flag = sov(x_1=x_1, x_2=x_2, x_3=x_3, i=i, j=j, label=label,
                                                     reallength1=reallength1)

        TP=(((H_count/(H_count+E_count+C_count))*H_TP)+((E_count/(H_count+E_count+C_count))*E_TP)+((C_count/(H_count+E_count+C_count))*C_TP))
        TN = (((H_count/(H_count+E_count+C_count))*H_TN)+((E_count/(H_count+E_count+C_count))*E_TN)+((C_count/(H_count+E_count+C_count))*C_TN))
        FP =(((H_count/(H_count+E_count+C_count))*H_FP)+((E_count/(H_count+E_count+C_count))*E_FP)+((C_count/(H_count+E_count+C_count))*C_FP))
        FN = (((H_count/(H_count+E_count+C_count))*H_FN)+((E_count/(H_count+E_count+C_count))*E_FN)+((C_count/(H_count+E_count+C_count))*C_FN))
        # sn = TP / (TP + FN)
        # sp = TN / (TN + FN)
        # mcc = ((TP * TN) - (FP * FN)) / (
        #     cmath.sqrt((TP + FN) * (TP + FP) * (TN + FN) * (TN + FP)))







        return TP,TN,FP,FN,H_TP,H_TN,H_FP,H_FN,E_TP,E_TN,E_FP,E_FN,C_TP,C_TN,C_FP,C_FN,H1,E1,C1,H2,E2,C2,H,E,C,H_count,E_count,C_count,H_FN,H_TP,H_FP,H_TN,E_FN,E_TP,E_FP,E_TN,C_FN,C_TP,C_FP,C_TN,H_H,H_E,H_C,E_H,E_E,E_C,C_H,C_E,C_C,sov_h,sov_e,sov_c,sov_overall,flag



# cwd = os.getcwd()
read_path = args.model
# 修改当前工作目录
# os.chdir(read_path)
############################################################################################################################################################

model = Net()


if torch.cuda.is_available():
    device = torch.device("cuda:0" )
    model.load_state_dict(torch.load(read_path,map_location=device))
else:
    device=torch.device("cpu")
    model.load_state_dict(torch.load(read_path, map_location=torch.device('cpu')))
model.to(device)


criterion = restricted_loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)





class Test_Dataset(data.Dataset):
    def __init__(self, pssm, hmm, label,seqt):
        self.pssm = pssm
        self.hmm = hmm
        self.label = label

        self.seqt =seqt
    def __getitem__(self, index):  # 返回的是tensor
        pssm, hmm, label,seqt= self.pssm[index], self.hmm[index], self.label[index],self.seqt[index]
        return pssm, hmm, label,seqt

    def __len__(self):
        return test_num



# class MyTestDataset(data.Dataset):
#     def __init__(self, pssm,hmm,label):
#         self.pssm = pssm
#         self.hmm=hmm
#         self.label = label
#
#     def __getitem__(self, index):#返回的是tensor
#         test_pssm,test_hmm,test_label = self.pssm[index], self.hmm[index],self.label[index]
#         return test_pssm,test_hmm,test_label
#
#     def __len__(self):
#         return (test_num-sample_num)

test_dataset = Test_Dataset(x1t, x2t, y1t,seqt)
# test_dataset=MyDataset(x1,x2,reallength1t)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batchsize, shuffle=False, drop_last=True)

acc_dict = []
def save_sequence(y,index,length):
    re=''
    for i in range(length[index]):
        if y[i]==0:
            re+='H'
        elif y[i]==1:
            re+='E'
        else:
            re+='C'
    return re
def save_model(high_acc, epoch, bgru, h_acc, h_sn, h_sp, h_mcc):
    highest = high_acc

    BGRU = bgru
    pred = []
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    htp = 0
    htn = 0
    hfp = 0
    hfn = 0
    etp = 0
    etn = 0
    efp = 0
    efn = 0
    ctp = 0
    ctn = 0
    cfp = 0
    cfn = 0
    H_H = 0
    H_E = 0
    H_C = 0
    E_H = 0
    E_E = 0
    E_C = 0
    C_H = 0
    C_E = 0
    C_C = 0
    SOV_H = 0
    SOV_E = 0
    SOV_C = 0
    SOV = 0
    total = 0
    h = 0
    e = 0
    c = 0
    t_H=0
    t_E=0
    t_C=0
    t_H1=0
    t_E1=0
    t_C1=0
    t_H2=0
    t_E2=0
    t_C2=0
    H=0
    E=0
    C=0
    H1=0
    E1=0
    C1=0
    H2=0
    E2=0
    C2 = 0
    sn = 0
    sp = 0
    mcc = 0
    acc = 0
    precision = 0
    sov_count = test_num
    tH = 0
    tE = 0
    tC = 0
    tH_count = 0
    tE_count = 0
    tC_count = 0
    H = 0
    E = 0
    C = 0
    H_count = 0
    E_count = 0
    C_count = 0
    HEC_acc = 0
    H_TP=0
    E_TP = 0
    C_TP = 0
    H_FP = 0
    E_FP = 0
    C_FP = 0
    H_TN = 0
    E_TN = 0
    C_TN = 0
    H_FN = 0
    E_FN = 0
    C_FN = 0

    tH_TP = 0
    tE_TP = 0
    tC_TP = 0
    tH_FP = 0
    tE_FP = 0
    tC_FP = 0
    tH_TN = 0
    tE_TN = 0
    tC_TN = 0
    tH_FN = 0
    tE_FN = 0
    tC_FN = 0
    sequences=[]
    for i, (test_pssm, test_hmm, test_label, label_for_loss) in enumerate(tqdm(test_loader)):
        # print("test_pssm",test_pssm)
        test_pssm = torch.tensor(test_pssm, dtype=torch.float32).to(device)
        test_hmm = torch.tensor(test_hmm, dtype=torch.float32).to(device)
        test_label = torch.tensor(test_label, dtype=torch.float32).to(device)
        label_for_loss = torch.tensor(label_for_loss, dtype=torch.float32).to(device)
        model.eval()
        y_test = model(test_pssm, test_hmm, label_for_loss, trainstate=False)
        y_test1=y_test.squeeze()
        y_test1=torch.argmax(y_test1,dim=1)
        y_test1=np.array(y_test1.cpu())
        sequence=save_sequence(y_test1,i,reallength1t)

        sequences.append(sequence)
        # pred=np.array(pred)
        #
        # pred=np.append(pred,y_test1)


        # rep_test = model.get_rep_layer()
        # criterion(output=y_test, tar=test_label, rep=rep_test, who='test')

        t_TP, t_TN, t_FP, t_FN, tH_TP, tH_TN, tH_FP, tH_FN, tE_TP, tE_TN, tE_FP, tE_FN, tC_TP, tC_TN, tC_FP, tC_FN,t_H1,t_E1,t_C1,t_H2,t_E2,t_C2,t_H,t_E,t_C,tH_count,tE_count,tC_count,tH_FN,tH_TP,tH_FP,tH_TN,tE_FN,tE_TP,tE_FP,tE_TN,tC_FN,tC_TP,tC_FP,tC_TN,h_h, h_e, h_c, e_h, e_e, e_c, c_h, c_e, c_c,sov_h, sov_e, sov_c, sov_o, flag = accuracy(
            x_1=y_test[0, :, 0].cpu(), x_2=y_test[0, :, 1].cpu(), x_3=y_test[0, :, 2].cpu(), label=test_label.cpu(),
            i=i, j=0, reallength1=reallength1t, epoch=epoch, who='test')
        H=H+t_H
        E=E+t_E
        C=C+t_C
        H1 = H1 + t_H1
        E1 = E1 + t_E1
        C1 = C1 + t_C1
        H2 = H2 + t_H2
        E2 = E2 + t_E2
        C2 = C2 + t_C2
        H_count=H_count+tH_count
        E_count = E_count + tE_count
        C_count = C_count + tC_count
        H_TP= H_TP+tH_TP
        H_FP = H_FP + tH_FP
        H_TN = H_TN + tH_TN
        H_FN = H_FN + tH_FN

        E_TP = E_TP + tE_TP
        E_FP = E_FP + tE_FP
        E_TN = E_TN + tE_TN
        E_FN = E_FN + tE_FN

        C_TP = C_TP + tC_TP
        C_FP = C_FP + tC_FP
        C_TN = C_TN + tC_TN
        C_FN = C_FN + tC_FN




        hfn = hfn + tH_FN
        efn = efn + tE_FN
        cfn = cfn + tC_FN

        htp = htp + tH_TP
        etp = etp + tE_TP
        ctp = ctp + tC_TP

        htn = htn + tH_TN
        etn = etn + tE_TN
        ctn = ctn + tC_TN

        hfp = hfp + tH_FP
        efp = efp + tE_FP
        cfp = cfp + tC_FP

        H_H = H_H + h_h
        H_E = H_E + h_e
        H_C = H_C + h_c
        E_H = E_H + e_h
        E_E = E_E + e_e
        E_C = E_C + e_c
        C_H = C_H + c_h
        C_E = C_E + c_e
        C_C = C_C + c_c

        if (flag == 0):
            SOV_H = SOV_H + sov_h
            SOV_E = SOV_E + sov_e
            SOV_C = SOV_C + sov_c
            SOV = SOV + sov_o
        else:
            sov_count = sov_count - 1


        total = (TP + FN) * (TP + FP) * (TN + FN) * (TN + FP)
        h = (htp + hfn) * (htp + hfp) * (htn + hfn) * (htn + hfp)
        e = (etp + efn) * (etp + efp) * (etn + efn) * (etn + efp)
        c = (ctp + cfn) * (ctp + cfp) * (ctn + cfn) * (ctn + cfp)




    with open(args.output_dir+'predicted_seq.txt','w') as f:

        for i in range(len(sequences)):
            f.write(csv_name_list[i].split('/')[-1]+':')
            f.write(sequences[i]+'\n')



    HEC_acc = float(H + E + C) / float(H_count + E_count + C_count)
    print('HEC_acc:', HEC_acc)
  

    print('H_H', H_H)
    print('H_E', H_E)
    print('H_C', H_C)
    print('E_H', E_H)
    print('E_E', E_E)
    print('E_C', E_C)
    print('C_H', C_H)
    print('C_E', C_E)
    print('C_C', C_C)

    print('SOV_H', SOV_H / sov_count)
    print('SOV_E', SOV_E / sov_count)
    print('SOV_C', SOV_C / sov_count)
    print('SOV', SOV / sov_count)
    with open(args.output_dir+'evaluate.txt','w') as f:
        f.write(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        f.write('\n')
        f.write('HEC_acc:'+str(HEC_acc)+'\n')
        f.write('H_H:'+str(H_H)+'\n')
        f.write('H_E:'+str(H_E)+'\n')
        f.write('H_C:'+str(H_C)+'\n')
        f.write('E_H:'+str(E_H)+'\n')
        f.write('E_E:'+str(E_E)+'\n')
        f.write('E_C:'+str(E_C)+'\n')
        f.write('C_H:'+str(C_H)+'\n')
        f.write('C_E:'+str(C_E)+'\n')
        f.write('C_C:'+str(C_C)+'\n')

        f.write('SOV_H:'+ str(SOV_H / sov_count)+'\n')
        f.write('SOV_E:'+ str(SOV_E / sov_count)+'\n')
        f.write('SOV_C:'+ str(SOV_C / sov_count)+'\n')
        f.write('SOV:'  + str(SOV / sov_count)  +'\n')






    return highest, BGRU, h_sn, h_sp, h_mcc, h_acc


high_acc = 0


def get_parameters(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('Total', total_num, 'Trainable', trainable_num)


# get_parameters(model)
for epoch in range(0, epoch_num):


    with torch.no_grad():

        save_model(high_acc=high_acc, epoch=epoch, bgru=bgru1,
                                                                    h_sn=h_sn, h_sp=h_sp, h_mcc=h_mcc, h_acc=h_acc)

        
        # with torch.no_grad:
        #     for j in range(batchsize):
        #         accuracy(y_pred[j, :, 0].cpu(), y_pred[j, :, 1].cpu(), y_pred[j, :, 2].cpu(), i, j,
        #                  reallength1=reallength1, epoch=epoch)

print('Training Finished!')
print('sn:', h_sn)
print('sp:', h_sp)
print('mcc:', h_mcc)
print('acc:', h_acc)
with open(args.output_dir + 'evaluate.txt', 'a') as f:
    f.write('sn:'+str(h_sn)+'\n')
    f.write('sp:'+str(h_sp)+'\n')
    f.write('mcc:'+ str(h_mcc)+'\n')
    f.write('acc:'+str(h_acc)+'\n')

get_parameters(model)
