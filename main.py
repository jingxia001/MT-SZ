"""
Code Reference:

PyTorch Geometric Team
Graph Classification Colab Notebook (PyTorch Geometric)
https://colab.research.google.com/drive/1I8a0DfQ3fI7Njc62__mVXUlcAleUclnb?usp=sharing 

"""

import torch
import numpy as np
import os
import scipy
import pandas as pd
import csv
import time
from sklearn.model_selection import KFold
from sklearn import preprocessing
import math
from utils import get_data
from utils import create_dataset
from torch_geometric.data import DataLoader
from model import  MT_GAT_topk_shareEn_multiple8_joint_last
import torch.nn
from torch.utils.data.dataloader import default_collate

#os.environ["CUDA_VISIBLE_DEVICES"]="3"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def RMSELoss(yhat,y):
    return torch.sqrt(torch.mean((yhat-y)**2))

def MAELoss(yhat,y):
    return torch.mean(torch.abs(yhat-y))
def L2Loss(model, alpha):
    l2_loss = torch.tensor(0.0, requires_grad=True)
    for name, parma in model.named_parameters():
        if 'bias' not in name:
            l2_loss = l2_loss + (0.5*alpha * torch.sum(torch.pow(parma, 2)))
    return l2_loss

def L1Loss(model, alpha):
    l1_loss = torch.tensor(0.0, requires_grad=True)
    for name, parma in model.named_parameters():
        if 'bias' not in name:
            l1_loss = l1_loss + (0.5*alpha * torch.sum(torch.abs(parma)))
    return l1_loss

def write_attention(filename, train_attention, train_node_index, fold, task):
    #train_attention = np.float32(train_attention.cpu().detach().numpy())
    filename1 = filename + '/train_attention' + str(fold) + task + ".npy" 
    np.save(filename1, train_attention)
    mean_train = np.mean(train_attention, axis=0)
    filename2 = filename + '/train_mean_attention' + str(fold) + task + ".npy"
    np.save(filename2, mean_train) 

    filename1 = filename + '/train_node_index' + str(fold) + task + ".npy" 
    np.save(filename1, train_node_index)
    mean_train = np.mean(train_node_index, axis=0)
    filename2 = filename + '/train_mean_node_index' + str(fold) + task + ".npy"
    np.save(filename2, mean_train) 

def write_attention2(filename, train_attention, fold, task):
    filename1 = filename + '/train_attention' + str(fold) + task + ".npy" 
    np.save(filename1, train_attention)
    mean_train = np.mean(train_attention, axis=0)
    filename2 = filename + '/train_mean_attention' + str(fold) + task + ".npy"
    np.save(filename2, mean_train) 

def write_attention3(filename, train_index_szf, train_weight_szf, fold, task):

    filename1 = filename + '/train_attention' + str(fold) + task + ".npy" 
    np.save(filename1, train_weight_szf)
    mean_train = np.mean(train_weight_szf, axis=0)
    filename2 = filename + '/train_mean_attention' + str(fold) + task + ".npy"
    np.save(filename2, mean_train) 

    filename1 = filename + '/train_node_index' + str(fold) + task + ".npy" 
    np.save(filename1, train_index_szf)
    mean_train = np.mean(train_index_szf, axis=0)
    filename2 = filename + '/train_mean_node_index' + str(fold) + task + ".npy"
    np.save(filename2, mean_train) 


def train(model_to_train, train_dataset_loader, model_optimizer,test_score, test_score2, test_score3, test_score4, test_score5, test_score6,test_score7, test_score8, device):
    model_to_train.train()

    for data in train_dataset_loader:  # Iterate in batches over the training dataset.
        data.x, data.edge_index, data.edge_weight, data.batch, data.y = data.x.to(device), data.edge_index.to(device), data.edge_weight.to(device),data.batch.to(device),data.y.to(device) 
        out, index, att, out2, index2, att2, out3, index3, att3, out4, index4, att4, out5, index5, att5, out6, index6, att6, \
        out7, index7, att7,out8, index8, att8, att9  = model_to_train(data.x.float(), data.edge_index, data.edge_weight, data.batch, device)  # Perform a single forward pass.
        #, x111, x222, edge_sz, weight_sz, edge_cog, weight_cog
        test_tem = test_score[data.y]
        test_tem2 = test_score2[data.y]
        test_tem3 = test_score3[data.y]
        test_tem4 = test_score4[data.y]
        test_tem5 = test_score5[data.y]
        test_tem6 = test_score6[data.y]
        test_tem7 = test_score7[data.y]
        test_tem8 = test_score8[data.y]
       
        loss = RMSELoss(out, torch.reshape(test_tem.float(),out.shape)) + RMSELoss(out2, torch.reshape(test_tem2.float(),out2.shape)) + \
        RMSELoss(out3, torch.reshape(test_tem3.float(),out3.shape)) + RMSELoss(out4, torch.reshape(test_tem4.float(),out4.shape)) + \
        2*(RMSELoss(out5, torch.reshape(test_tem5.float(),out5.shape)) + RMSELoss(out6, torch.reshape(test_tem6.float(),out6.shape)) + \
        RMSELoss(out7, torch.reshape(test_tem7.float(),out7.shape)) + RMSELoss(out8, torch.reshape(test_tem8.float(),out8.shape))) + \
        L2Loss(model_to_train, 0.001) 
        
        if loss == 'nan':
            break
        loss.backward()  # Deriving gradients.
        model_optimizer.step()  # Updating parameters based on gradients.
        model_optimizer.zero_grad()  # Clearing gradients.

def test(model, loader, test_score, test_score2, test_score3, test_score4, test_score5, test_score6, test_score7, test_score8, device):
   
    out_sum = torch.tensor(()).to(device)
    true_sum = torch.tensor(()).to(device)
    attention = torch.tensor(()).to(device)
    node_index = torch.tensor(()).to(device)

    out_sum2 = torch.tensor(()).to(device)
    true_sum2 = torch.tensor(()).to(device)
    attention2 = torch.tensor(()).to(device)
    node_index2 = torch.tensor(()).to(device)

    out_sum3 = torch.tensor(()).to(device)
    true_sum3 = torch.tensor(()).to(device)
    attention3 = torch.tensor(()).to(device)
    node_index3 = torch.tensor(()).to(device)

    out_sum4 = torch.tensor(()).to(device)
    true_sum4 = torch.tensor(()).to(device)
    attention4 = torch.tensor(()).to(device)
    node_index4 = torch.tensor(()).to(device)

    out_sum5 = torch.tensor(()).to(device)
    true_sum5 = torch.tensor(()).to(device)
    attention5 = torch.tensor(()).to(device)
    node_index5 = torch.tensor(()).to(device)

    out_sum6 = torch.tensor(()).to(device)
    true_sum6 = torch.tensor(()).to(device)
    attention6 = torch.tensor(()).to(device)
    node_index6 = torch.tensor(()).to(device)

    out_sum7 = torch.tensor(()).to(device)
    true_sum7 = torch.tensor(()).to(device)
    attention7 = torch.tensor(()).to(device)
    node_index7 = torch.tensor(()).to(device)

    out_sum8 = torch.tensor(()).to(device)
    true_sum8 = torch.tensor(()).to(device)
    attention8 = torch.tensor(()).to(device)
    node_index8 = torch.tensor(()).to(device)

    attention9 = torch.tensor(()).to(device)

    attention_index_sz = torch.tensor(()).to(device)
    attention_weight_sz = torch.tensor(()).to(device)
    attention_index_cog = torch.tensor(()).to(device)
    attention_weight_cog = torch.tensor(()).to(device)

    model.eval()
    with torch.no_grad():
    
        for data in loader:  # Iterate in batches over the training/test dataset.
            data.x, data.edge_index, data.edge_weight,data.batch, data.y = data.x.to(device), data.edge_index.to(device), data.edge_weight.to(device),data.batch.to(device),data.y.to(device) 
            test_tem = test_score[data.y]
            test_tem2 = test_score2[data.y]
            test_tem3 = test_score3[data.y]
            test_tem4 = test_score4[data.y]
            test_tem5 = test_score5[data.y]
            test_tem6 = test_score6[data.y]
            test_tem7 = test_score7[data.y]
            test_tem8 = test_score8[data.y]
            out, index, att, out2, index2, att2, out3, index3, att3, out4, index4, att4, out5, index5, att5, out6, index6, att6, \
            out7, index7, att7,out8, index8, att8, att9 = model(data.x.float(), data.edge_index, data.edge_weight, data.batch, device) 
            # , x111, x222, edge_sz, weight_sz, edge_cog, weight_cog
            # print(out_sum.shape)
            out_sum = torch.cat((out_sum, out), dim=0)       
            true_sum = torch.cat((true_sum, torch.reshape(test_tem.float(), out.shape)), dim=0)
            node_index = torch.cat((node_index, index), dim=0)
            attention = torch.cat((attention, att), dim=0)

            out_sum2 = torch.cat((out_sum2, out2), dim=0)       
            true_sum2 = torch.cat((true_sum2, torch.reshape(test_tem2.float(), out2.shape)), dim=0)
            node_index2 = torch.cat((node_index2, index2), dim=0)
            attention2 = torch.cat((attention2, att2), dim=0)

            out_sum3 = torch.cat((out_sum3, out3), dim=0)       
            true_sum3 = torch.cat((true_sum3, torch.reshape(test_tem3.float(), out3.shape)), dim=0)
            node_index3 = torch.cat((node_index3, index3), dim=0)
            attention3 = torch.cat((attention3, att3), dim=0)

            out_sum4 = torch.cat((out_sum4, out4), dim=0)       
            true_sum4 = torch.cat((true_sum4, torch.reshape(test_tem4.float(), out4.shape)), dim=0)
            node_index4 = torch.cat((node_index4, index4), dim=0)
            attention4 = torch.cat((attention4, att4), dim=0)

            out_sum5 = torch.cat((out_sum5, out5), dim=0)       
            true_sum5 = torch.cat((true_sum5, torch.reshape(test_tem5.float(), out5.shape)), dim=0)
            node_index5 = torch.cat((node_index5, index5), dim=0)
            attention5 = torch.cat((attention5, att5), dim=0)

            out_sum6 = torch.cat((out_sum6, out6), dim=0)       
            true_sum6 = torch.cat((true_sum6, torch.reshape(test_tem6.float(), out6.shape)), dim=0)
            node_index6 = torch.cat((node_index6, index6), dim=0)
            attention6 = torch.cat((attention6, att6), dim=0)

            out_sum7 = torch.cat((out_sum7, out7), dim=0)       
            true_sum7 = torch.cat((true_sum7, torch.reshape(test_tem7.float(), out7.shape)), dim=0)
            node_index7 = torch.cat((node_index7, index7), dim=0)
            attention7 = torch.cat((attention7, att7), dim=0)

            out_sum8 = torch.cat((out_sum8, out8), dim=0)       
            true_sum8 = torch.cat((true_sum8, torch.reshape(test_tem8.float(), out8.shape)), dim=0)
            node_index8 = torch.cat((node_index8, index8), dim=0)
            attention8 = torch.cat((attention8, att8), dim=0)

            attention9 = torch.cat((attention9, att9), dim=0)
            # attention_index_sz = torch.cat((attention_index_sz, edge_sz), dim=1)
            # attention_weight_sz = torch.cat((attention_weight_sz, weight_sz), dim=0)
            # attention_index_cog = torch.cat((attention_index_cog, edge_cog), dim=1)
            # attention_weight_cog = torch.cat((attention_weight_cog, weight_cog), dim=0)

        rmse = RMSELoss(out_sum, torch.reshape(true_sum.float(),out_sum.shape))
        mae = MAELoss(out_sum, torch.reshape(true_sum.float(),out_sum.shape))
        rmse2 = RMSELoss(out_sum2, torch.reshape(true_sum2.float(),out_sum2.shape))
        mae2 = MAELoss(out_sum2, torch.reshape(true_sum2.float(),out_sum2.shape))
        rmse3 = RMSELoss(out_sum3, torch.reshape(true_sum3.float(),out_sum3.shape))
        mae3 = MAELoss(out_sum3, torch.reshape(true_sum3.float(),out_sum3.shape))
        rmse4 = RMSELoss(out_sum4, torch.reshape(true_sum4.float(),out_sum4.shape))
        mae4 = MAELoss(out_sum4, torch.reshape(true_sum4.float(),out_sum4.shape))
        rmse5 = RMSELoss(out_sum5, torch.reshape(true_sum5.float(),out_sum5.shape))
        mae5 = MAELoss(out_sum5, torch.reshape(true_sum5.float(),out_sum5.shape))
        rmse6 = RMSELoss(out_sum6, torch.reshape(true_sum6.float(),out_sum6.shape))
        mae6 = MAELoss(out_sum6, torch.reshape(true_sum6.float(),out_sum6.shape))
        rmse7 = RMSELoss(out_sum7, torch.reshape(true_sum7.float(),out_sum7.shape))
        mae7 = MAELoss(out_sum7, torch.reshape(true_sum7.float(),out_sum7.shape))
        rmse8 = RMSELoss(out_sum8, torch.reshape(true_sum8.float(),out_sum8.shape))
        mae8 = MAELoss(out_sum8, torch.reshape(true_sum8.float(),out_sum8.shape))

        truev = torch.reshape(true_sum,(len(out_sum),1))
        out = np.squeeze(np.asarray(out_sum.cpu().detach().numpy()))
        truev = np.squeeze(np.asarray(truev.cpu().detach().numpy()))       
        corr = scipy.stats.pearsonr(out, truev)
        
        truev2 = torch.reshape(true_sum2,(len(out_sum2),1))
        out2 = np.squeeze(np.asarray(out_sum2.cpu().detach().numpy()))
        truev2 = np.squeeze(np.asarray(truev2.cpu().detach().numpy()))       
        corr2 = scipy.stats.pearsonr(out2, truev2)

        truev3 = torch.reshape(true_sum3,(len(out_sum3),1))
        out3 = np.squeeze(np.asarray(out_sum3.cpu().detach().numpy()))
        truev3 = np.squeeze(np.asarray(truev3.cpu().detach().numpy()))       
        corr3 = scipy.stats.pearsonr(out3, truev3)

        truev4 = torch.reshape(true_sum4,(len(out_sum4),1))
        out4 = np.squeeze(np.asarray(out_sum4.cpu().detach().numpy()))
        truev4 = np.squeeze(np.asarray(truev4.cpu().detach().numpy()))       
        corr4 = scipy.stats.pearsonr(out4, truev4)

        truev5 = torch.reshape(true_sum5,(len(out_sum5),1))
        out5 = np.squeeze(np.asarray(out_sum5.cpu().detach().numpy()))
        truev5 = np.squeeze(np.asarray(truev5.cpu().detach().numpy()))       
        corr5 = scipy.stats.pearsonr(out5, truev5)

        truev6 = torch.reshape(true_sum6,(len(out_sum6),1))
        out6 = np.squeeze(np.asarray(out_sum6.cpu().detach().numpy()))
        truev6 = np.squeeze(np.asarray(truev6.cpu().detach().numpy()))       
        corr6 = scipy.stats.pearsonr(out6, truev6)

        truev7 = torch.reshape(true_sum7,(len(out_sum7),1))
        out7 = np.squeeze(np.asarray(out_sum7.cpu().detach().numpy()))
        truev7 = np.squeeze(np.asarray(truev7.cpu().detach().numpy()))       
        corr7 = scipy.stats.pearsonr(out7, truev7)

        truev8 = torch.reshape(true_sum8,(len(out_sum8),1))
        out8 = np.squeeze(np.asarray(out_sum8.cpu().detach().numpy()))
        truev8 = np.squeeze(np.asarray(truev8.cpu().detach().numpy()))       
        corr8 = scipy.stats.pearsonr(out8, truev8)
        

        attention = np.squeeze(np.asarray(attention.cpu().detach().numpy()))
        node_index = np.squeeze(np.asarray(node_index.cpu().detach().numpy()))

        attention2 = np.squeeze(np.asarray(attention2.cpu().detach().numpy()))
        node_index2 = np.squeeze(np.asarray(node_index2.cpu().detach().numpy()))

        attention3 = np.squeeze(np.asarray(attention3.cpu().detach().numpy()))
        node_index3 = np.squeeze(np.asarray(node_index3.cpu().detach().numpy()))

        attention4 = np.squeeze(np.asarray(attention4.cpu().detach().numpy()))
        node_index4 = np.squeeze(np.asarray(node_index4.cpu().detach().numpy()))

        attention5 = np.squeeze(np.asarray(attention5.cpu().detach().numpy()))
        node_index5 = np.squeeze(np.asarray(node_index5.cpu().detach().numpy()))

        attention6 = np.squeeze(np.asarray(attention6.cpu().detach().numpy()))
        node_index6 = np.squeeze(np.asarray(node_index6.cpu().detach().numpy()))

        attention7 = np.squeeze(np.asarray(attention7.cpu().detach().numpy()))
        node_index7 = np.squeeze(np.asarray(node_index7.cpu().detach().numpy()))

        attention8 = np.squeeze(np.asarray(attention8.cpu().detach().numpy()))
        node_index8 = np.squeeze(np.asarray(node_index8.cpu().detach().numpy()))

        attention9 = np.squeeze(np.asarray(attention9.cpu().detach().numpy()))
        attention_index_sz = np.squeeze(np.asarray(attention_index_sz.cpu().detach().numpy()))
        attention_weight_sz = np.squeeze(np.asarray(attention_weight_sz.cpu().detach().numpy()))
        attention_index_cog = np.squeeze(np.asarray(attention_index_cog.cpu().detach().numpy()))
        attention_weight_cog = np.squeeze(np.asarray(attention_weight_cog.cpu().detach().numpy()))

    return corr[0], rmse, mae, out, truev, node_index, attention, corr2[0], rmse2, mae2, out2, truev2, node_index2, attention2, \
    corr3[0], rmse3, mae3, out3, truev3, node_index3, attention3, corr4[0], rmse4, mae4, out4, truev4, node_index4, attention4, \
    corr5[0], rmse5, mae5, out5, truev5, node_index5, attention5, corr6[0], rmse6, mae6, out6, truev6, node_index6, attention6, \
    corr7[0], rmse7, mae7, out7, truev7, node_index7, attention7, corr8[0], rmse8, mae8, out8, truev8, node_index8, attention8, \
    attention9#, attention_index_sz, attention_weight_sz, attention_index_cog, attention_weight_cog
           
def transform_inver(target_scaler, test_out, test_true,train_out, train_true):

    test_outt = target_scaler.inverse_transform(test_out.reshape(-1,1))[:,0]    
    test_truet = target_scaler.inverse_transform(test_true.reshape(-1,1))[:,0]
    train_outt = target_scaler.inverse_transform(train_out.reshape(-1,1))[:,0]    
    train_truet = target_scaler.inverse_transform(train_true.reshape(-1,1))[:,0]
    return test_outt, test_truet, train_outt, train_truet

def transform(train_scoreo, test_scoreo):
    target_scaler = preprocessing.StandardScaler().fit(train_scoreo.reshape(-1,1))
    train_score = target_scaler.transform(train_scoreo.reshape(-1,1))[:,0] 
    test_score = target_scaler.transform(test_scoreo.reshape(-1,1))[:,0]
    return target_scaler, train_score, test_score
def generate_train(all_score, X_train, X_test):
    
    train_scoreo = all_score[X_train]
    test_scoreo = all_score[X_test]
    return train_scoreo, test_scoreo

def compute_RSME(test_outt, test_truet, train_outt,train_truet):
    rmse_test = RMSELoss(torch.tensor(test_outt), torch.tensor(test_truet))
    MAE_test = MAELoss(torch.tensor(test_outt),torch.tensor(test_truet))
    rmse_train = RMSELoss(torch.tensor(train_outt), torch.tensor(train_truet))
    MAE_train = MAELoss(torch.tensor(train_outt),torch.tensor(train_truet))
    return rmse_test, MAE_test, rmse_train, MAE_train

def format_trans(train_corr, test_corr, test_rmse, test_mae):
    train_corr = torch.tensor(np.float32(train_corr))
    test_corr = torch.tensor(np.float32(test_corr))
    test_rmse = torch.tensor(np.float32(test_rmse.cpu().detach().numpy()))
    test_mae = torch.tensor(np.float32(test_mae.cpu().detach().numpy()))
    return train_corr, test_corr, test_rmse, test_mae

def thebest(test_corr, rmse_test, MAE_test, test_out, train_out, train_true, test_true, test_outt, train_outt, train_truet, test_truet, train_attention, train_node_index):
    test_corrf = test_corr
    test_rmsef = rmse_test
    test_maef = MAE_test

    test_outf = test_out
    train_outf = train_out
    train_truef = train_true
    test_truef = test_true

    test_outtf = test_outt
    train_outtf = train_outt
    train_truetf = train_truet
    test_truetf = test_truet

    train_attentionf = train_attention
    train_node_indexf = train_node_index
    return test_corrf, test_rmsef, test_maef, test_outf, train_outf, train_truef, test_truef, test_outtf, train_outtf, train_truetf, test_truetf, train_attentionf, train_node_indexf 

def app(test_corr_s, test_rmse_s, test_mae_s, test_corr, test_rmse, test_mae):
   
    test_corr_s.append(test_corr)
    test_rmse_s.append(test_rmse)
    test_mae_s.append(test_mae)
    return test_corr_s, test_rmse_s, test_mae_s

def compute_corr(test_corr_s, test_rmse_s, test_mae_s):
    corr = torch.tensor(np.mean(np.array(test_corr_s,dtype=np.float32)))
    rmse = torch.tensor(np.mean(np.array(test_rmse_s,dtype=np.float32)))
    mae = torch.tensor(np.mean(np.array(test_mae_s,dtype=np.float32)))
    return corr, rmse, mae  

hidden_channels=64
hidden_channels2=64
hc = 96 
hc2= 96
hc3 = 96
hc4 = 96
hc5 = 96
hc6 = 96
hc7 = 96
hc8 = 96
ratio=0.9            #256
epoch_num = 60
decay_rate = 0.005
decay_step = 20
lr = 0.0001
num_folds = 5
batch_size = 20
runnum = 'e10'

print("\n---------Starting to load Data---------\n")
task='positive'
task2='negative'
task3='total'
task4='general'
task5= 'PS'
task6= 'BACS'
task7= 'WM'
task8= 'VL'
connpath = '/data/jing001/Multi-task-SZ/COBRE/X.npy'  ##need to modify
scorepath = '/data/jing001/Multi-task-SZ/COBRE/Y_' + task +'.npy'  ##need to modify
scorepath2 = '/data/jing001/Multi-task-SZ/COBRE/Y_' + task2 +'.npy'  ##need to modify
scorepath3 = '/data/jing001/Multi-task-SZ/COBRE/Y_' + task3 +'.npy'  ##need to modify
scorepath4 = '/data/jing001/Multi-task-SZ/COBRE/Y_' + task4 +'.npy'  ##need to modify
scorepath5 = '/data/jing001/Multi-task-SZ/COBRE/Y_' + task5 +'.npy'  ##need to modify
scorepath6 = '/data/jing001/Multi-task-SZ/COBRE/Y_' + task6 +'.npy'  ##need to modify
scorepath7 = '/data/jing001/Multi-task-SZ/COBRE/Y_' + task7 +'.npy'  ##need to modify
scorepath8 = '/data/jing001/Multi-task-SZ/COBRE/Y_' + task8 +'.npy'  ##need to modify
timefile = '/data/jing001/Multi-task-SZ/MT-SZ/results_FC/COBRE-MT-GAT_topk_shareEn_joint_corr_' + task +str(hidden_channels) + task2 + str(hc2) + '_' + str(runnum) + '_joint_' + str(ratio) + '_'+str(int(time.time()))
finalfile = timefile+'/final_result.csv' 
finalfile2 = timefile+'/final_result_best.csv'              

os.mkdir(timefile)
Labelfile   = timefile+'/predict_train.csv'
Labelfile_test   = timefile+'/predict_test.csv'                
all_data, all_score = get_data(connpath, scorepath)
all_data, all_score2 = get_data(connpath, scorepath2)
all_data, all_score3 = get_data(connpath, scorepath3)
all_data, all_score4 = get_data(connpath, scorepath4)
all_data, all_score5 = get_data(connpath, scorepath5)
all_data, all_score6 = get_data(connpath, scorepath6)
all_data, all_score7 = get_data(connpath, scorepath7)
all_data, all_score8 = get_data(connpath, scorepath8)

for i in range(len(all_data)):
    feature_matrix_ori = np.array(all_data[i])

num = all_data.shape[0]
kf = KFold(n_splits=num_folds, shuffle=True)
print("\n--------Split and Data loaded-----------\n")
fold = 0
true_out = np.squeeze(np.array([[]]))
pred_out = np.squeeze(np.array([[]]))

true_out2 = np.squeeze(np.array([[]]))
pred_out2 = np.squeeze(np.array([[]]))

true_out3 = np.squeeze(np.array([[]]))
pred_out3 = np.squeeze(np.array([[]]))

true_out4 = np.squeeze(np.array([[]]))
pred_out4 = np.squeeze(np.array([[]]))

true_out5 = np.squeeze(np.array([[]]))
pred_out5 = np.squeeze(np.array([[]]))

true_out6 = np.squeeze(np.array([[]]))
pred_out6 = np.squeeze(np.array([[]]))

true_out7 = np.squeeze(np.array([[]]))
pred_out7 = np.squeeze(np.array([[]]))

true_out8 = np.squeeze(np.array([[]]))
pred_out8 = np.squeeze(np.array([[]]))

test_corr_s = []
test_rmse_s = []
test_mae_s = []

test_corr_s2 = []
test_rmse_s2 = []
test_mae_s2 = []

test_corr_s3 = []
test_rmse_s3 = []
test_mae_s3 = []

test_corr_s4 = []
test_rmse_s4 = []
test_mae_s4 = []

test_corr_s5 = []
test_rmse_s5 = []
test_mae_s5 = []

test_corr_s6 = []
test_rmse_s6 = []
test_mae_s6 = []

test_corr_s7 = []
test_rmse_s7 = []
test_mae_s7 = []

test_corr_s8 = []
test_rmse_s8 = []
test_mae_s8 = []

for X_train, X_test in kf.split(list(range(1,num))):
    print(fold)
    fold = fold+1
    
    model = MT_GAT_topk_shareEn_multiple8_joint_last(hidden_channels, hidden_channels2, hc, hc2, hc3, hc4, hc5, hc6, hc7, hc8, ratio)
    print("Model:\n\t",model)
    print(torch.cuda.is_available())
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)
    
    fine_file = '/data/jing001/Multi-task-SZ/MT-SZ/'+runnum + '/' 
    train_file = fine_file + 'Train' +str(fold) +'.txt'
    test_file = fine_file + 'Test' +str(fold) +'.txt'
    X_train = []
    X_test = []
    with open(train_file) as f:
        for line in f.readlines():
            X_train.append(line.strip('\n'))
    X_train = [eval(i) for i in X_train]    
       
    with open(test_file) as f:
        for line in f.readlines():
            X_test.append(line.strip('\n'))
    X_test = [eval(i) for i in X_test]

    train_data = all_data[X_train]
    test_data = all_data[X_test]

    train_scoreo, test_scoreo = generate_train(all_score, X_train, X_test)    
    train_scoreo2, test_scoreo2 = generate_train(all_score2, X_train, X_test)  
    train_scoreo3, test_scoreo3 = generate_train(all_score3, X_train, X_test)  
    train_scoreo4, test_scoreo4 = generate_train(all_score4, X_train, X_test)  
    train_scoreo5, test_scoreo5 = generate_train(all_score5, X_train, X_test)  
    train_scoreo6, test_scoreo6 = generate_train(all_score6, X_train, X_test) 
    train_scoreo7, test_scoreo7 = generate_train(all_score7, X_train, X_test)
    train_scoreo8, test_scoreo8 = generate_train(all_score8, X_train, X_test) 
    
    target_scaler, train_score, test_score = transform(train_scoreo, test_scoreo)
    target_scaler2, train_score2, test_score2 = transform(train_scoreo2, test_scoreo2)
    target_scaler3, train_score3, test_score3 = transform(train_scoreo3, test_scoreo3)
    target_scaler4, train_score4, test_score4 = transform(train_scoreo4, test_scoreo4)
    target_scaler5, train_score5, test_score5 = transform(train_scoreo5, test_scoreo5)
    target_scaler6, train_score6, test_score6 = transform(train_scoreo6, test_scoreo6)
    target_scaler7, train_score7, test_score7 = transform(train_scoreo7, test_scoreo7)
    target_scaler8, train_score8, test_score8 = transform(train_scoreo8, test_scoreo8)

    index_test = np.reshape(np.arange(0,len(X_test)),(len(X_test),1))
    index_train = np.reshape(np.arange(0,len(X_train)),(len(X_train),1))

    training_dataset = create_dataset(train_data, index_train)
    testing_dataset = create_dataset(test_data, index_test)

    train_score_input = torch.tensor(train_score).to(device)
    test_score_input = torch.tensor(test_score).to(device)

    train_score_input2 = torch.tensor(train_score2).to(device)
    test_score_input2 = torch.tensor(test_score2).to(device)

    train_score_input3 = torch.tensor(train_score3).to(device)
    test_score_input3 = torch.tensor(test_score3).to(device)

    train_score_input4 = torch.tensor(train_score4).to(device)
    test_score_input4 = torch.tensor(test_score4).to(device)

    train_score_input5 = torch.tensor(train_score5).to(device)
    test_score_input5 = torch.tensor(test_score5).to(device)

    train_score_input6 = torch.tensor(train_score6).to(device)
    test_score_input6 = torch.tensor(test_score6).to(device)

    train_score_input7 = torch.tensor(train_score7).to(device)
    test_score_input7 = torch.tensor(test_score7).to(device)

    train_score_input8 = torch.tensor(train_score8).to(device)
    test_score_input8 = torch.tensor(test_score8).to(device)

    with open('traindata.txt', 'w') as f:
        f.write(str(training_dataset))
    print(len(training_dataset))
    
    train_loader = DataLoader(training_dataset, batch_size, shuffle = True, collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)))
    test_loader = DataLoader(testing_dataset, batch_size, shuffle=True, collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)))

    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=decay_rate)

    Accfile = timefile+'/performance_test'+str(fold)+'.csv'
    Accfile2 = timefile+'/performance_test_ori'+str(fold)+'.csv' 
    Labelfile = timefile+'/predict_train'+str(fold)+'.csv'
    Labelfile2 = timefile+'/predict_train_ori'+str(fold)+'.csv'
    Labelfile_test = timefile+'/predict_test'+str(fold)+'.csv'
    Labelfile_test2 = timefile+'/predict_test_ori'+str(fold)+'.csv'
    df_name = {'epoch','train_rmse','train_mae','train_corr','test_rmse', 'test_mae','test_corr','train_rmse2','train_mae2','train_corr2','test_rmse2', 'test_mae2','test_corr2', \
    'train_rmse3','train_mae3','train_corr3','test_rmse3', 'test_mae3','test_corr3','train_rmse4','train_mae4','train_corr4','test_rmse4', 'test_mae4','test_corr4', \
    'train_rmse5','train_mae5','train_corr5','test_rmse5', 'test_mae5','test_corr5','train_rmse6','train_mae6','train_corr6','test_rmse6', 'test_mae6','test_corr6',\
    'train_rmse7','train_mae7','train_corr7','test_rmse7', 'test_mae7','test_corr7','train_rmse8','train_mae8','train_corr8','test_rmse8', 'test_mae8','test_corr8'}
    df_name = pd.DataFrame(columns=df_name)
    df_name.to_csv(Accfile, mode='a+', index=None)
    df_name.to_csv(Accfile2, mode='a+', index=None)

    train_path = timefile+'/Train'+str(fold)+'.txt'
    test_path = timefile+'/Test'+str(fold)+'.txt'

    file=open(train_path,'w')  
    for line in X_train:
        line = str(line)
        # print(line)
        file.write(line+'\n')
    file.close()

    file=open(test_path,'w')  
    for line in X_test:
        line = str(line)
        # print(line)
        file.write(line+'\n')
    file.close()

    epochf = 0


    for epoch in range(1, epoch_num):
        #if epoch % decay_step == 0:
            #for p in optimizer.param_groups:
                #p['lr'] *= decay_rate
        train(model, train_loader, optimizer,train_score_input,train_score_input2,train_score_input3,train_score_input4, train_score_input5,train_score_input6, train_score_input7, train_score_input8, device)
        
        train_corr, train_rmse, train_mae, train_out, train_true, train_node_index, train_attention, \
        train_corr2, train_rmse2, train_mae2, train_out2, train_true2, train_node_index2, train_attention2, \
        train_corr3, train_rmse3, train_mae3, train_out3, train_true3, train_node_index3, train_attention3, \
        train_corr4, train_rmse4, train_mae4, train_out4, train_true4, train_node_index4, train_attention4, \
        train_corr5, train_rmse5, train_mae5, train_out5, train_true5, train_node_index5, train_attention5, \
        train_corr6, train_rmse6, train_mae6, train_out6, train_true6, train_node_index6, train_attention6, \
        train_corr7, train_rmse7, train_mae7, train_out7, train_true7, train_node_index7, train_attention7, \
        train_corr8, train_rmse8, train_mae8, train_out8, train_true8, train_node_index8, train_attention8, \
        train_attention9 = \
        test(model, train_loader,train_score_input,train_score_input2, train_score_input3,train_score_input4, train_score_input5,train_score_input6, train_score_input7, train_score_input8, device)
        #, train_index_sz,train_weight_sz, train_index_cog, train_weight_cog
        test_corr, test_rmse, test_mae, test_out, test_true, test_node_index, test_attention, \
        test_corr2, test_rmse2, test_mae2, test_out2, test_true2, test_node_index2, test_attention2,\
        test_corr3, test_rmse3, test_mae3, test_out3, test_true3, test_node_index3, test_attention3, \
        test_corr4, test_rmse4, test_mae4, test_out4, test_true4, test_node_index4, test_attention4, \
        test_corr5, test_rmse5, test_mae5, test_out5, test_true5, test_node_index5, test_attention5, \
        test_corr6, test_rmse6, test_mae6, test_out6, test_true6, test_node_index6, test_attention6, \
        test_corr7, test_rmse7, test_mae7, test_out7, test_true7, test_node_index7, test_attention7, \
        test_corr8, test_rmse8, test_mae8, test_out8, test_true8, test_node_index8, test_attention8, \
        test_attention9 = \
        test(model, test_loader,test_score_input,test_score_input2, test_score_input3,test_score_input4, test_score_input5,test_score_input6, test_score_input7, test_score_input8, device)
        #, test_index_sz,test_weight_sz, test_index_cog, test_weight_cog
        #######################inverse
        test_outt, test_truet, train_outt, train_truet = transform_inver(target_scaler, test_out, test_true,train_out, train_true)
        test_outt2, test_truet2, train_outt2, train_truet2 = transform_inver(target_scaler2, test_out2, test_true2,train_out2, train_true2)
        test_outt3, test_truet3, train_outt3, train_truet3 = transform_inver(target_scaler3, test_out3, test_true3,train_out3, train_true3)
        test_outt4, test_truet4, train_outt4, train_truet4 = transform_inver(target_scaler4, test_out4, test_true4,train_out4, train_true4)
        test_outt5, test_truet5, train_outt5, train_truet5 = transform_inver(target_scaler5, test_out5, test_true5,train_out5, train_true5)
        test_outt6, test_truet6, train_outt6, train_truet6 = transform_inver(target_scaler6, test_out6, test_true6,train_out6, train_true6)
        test_outt7, test_truet7, train_outt7, train_truet7 = transform_inver(target_scaler7, test_out7, test_true7,train_out7, train_true7)
        test_outt8, test_truet8, train_outt8, train_truet8 = transform_inver(target_scaler8, test_out8, test_true8,train_out8, train_true8)
        
        rmse_test, MAE_test, rmse_train, MAE_train = compute_RSME(test_outt, test_truet, train_outt,train_truet)
        rmse_test2, MAE_test2, rmse_train2, MAE_train2 = compute_RSME(test_outt2, test_truet2, train_outt2, train_truet2)
        rmse_test3, MAE_test3, rmse_train3, MAE_train3 = compute_RSME(test_outt3, test_truet3, train_outt3, train_truet3)
        rmse_test4, MAE_test4, rmse_train4, MAE_train4 = compute_RSME(test_outt4, test_truet4, train_outt4, train_truet4)
        rmse_test5, MAE_test5, rmse_train5, MAE_train5 = compute_RSME(test_outt5, test_truet5, train_outt5, train_truet5)
        rmse_test6, MAE_test6, rmse_train6, MAE_train6 = compute_RSME(test_outt6, test_truet6, train_outt6, train_truet6)
        rmse_test7, MAE_test7, rmse_train7, MAE_train7 = compute_RSME(test_outt7, test_truet7, train_outt7, train_truet7)
        rmse_test8, MAE_test8, rmse_train8, MAE_train8 = compute_RSME(test_outt8, test_truet8, train_outt8, train_truet8)
    
        ##########################################
        if epoch % 1 == 0:
            print(epoch)
            print(f'Epoch: {epoch:03d}, Train_corr: {train_corr:.4f}, Train_rmse: {train_rmse:.4f},Train_mae: {train_mae:.4f}')
            print(f'Epoch: {epoch:03d}, Test_corr: {test_corr:.4f}, Test_rmse: {test_rmse:.4f},Test_mae: {test_mae:.4f}')

            print(f'Epoch: {epoch:03d}, Train_corr2: {train_corr2:.4f}, Train_rmse2: {train_rmse2:.4f},Train_mae2: {train_mae2:.4f}')
            print(f'Epoch: {epoch:03d}, Test_corr2: {test_corr2:.4f}, Test_rmse2: {test_rmse2:.4f},Test_mae2: {test_mae2:.4f}')

            print(f'Epoch: {epoch:03d}, Train_corr3: {train_corr3:.4f}, Train_rmse3: {train_rmse3:.4f},Train_mae3: {train_mae3:.4f}')
            print(f'Epoch: {epoch:03d}, Test_corr3: {test_corr3:.4f}, Test_rmse3: {test_rmse3:.4f},Test_mae3: {test_mae3:.4f}')

            print(f'Epoch: {epoch:03d}, Train_corr4: {train_corr4:.4f}, Train_rmse4: {train_rmse4:.4f},Train_mae4: {train_mae4:.4f}')
            print(f'Epoch: {epoch:03d}, Test_corr4: {test_corr4:.4f}, Test_rmse4: {test_rmse4:.4f},Test_mae4: {test_mae4:.4f}')

            print(f'Epoch: {epoch:03d}, Train_corr5: {train_corr5:.4f}, Train_rmse5: {train_rmse5:.4f},Train_mae5: {train_mae5:.4f}')
            print(f'Epoch: {epoch:03d}, Test_corr5: {test_corr5:.4f}, Test_rmse5: {test_rmse5:.4f},Test_mae5: {test_mae5:.4f}')

            print(f'Epoch: {epoch:03d}, Train_corr6: {train_corr6:.4f}, Train_rmse6: {train_rmse6:.4f},Train_mae6: {train_mae6:.4f}')
            print(f'Epoch: {epoch:03d}, Test_corr6: {test_corr6:.4f}, Test_rmse6: {test_rmse6:.4f},Test_mae6: {test_mae6:.4f}')
            
            print(f'Epoch: {epoch:03d}, Train_corr7: {train_corr7:.4f}, Train_rmse7: {train_rmse7:.4f},Train_mae7: {train_mae7:.4f}')
            print(f'Epoch: {epoch:03d}, Test_corr7: {test_corr7:.4f}, Test_rmse7: {test_rmse7:.4f},Test_mae7: {test_mae7:.4f}')

            print(f'Epoch: {epoch:03d}, Train_corr8: {train_corr8:.4f}, Train_rmse8: {train_rmse8:.4f},Train_mae8: {train_mae8:.4f}')
            print(f'Epoch: {epoch:03d}, Test_corr8: {test_corr8:.4f}, Test_rmse8: {test_rmse8:.4f},Test_mae8: {test_mae8:.4f}')
            #print(train_corr.float(), train_corr.shape)
            train_corr, test_corr, test_rmse, test_mae = format_trans(train_corr, test_corr, test_rmse, test_mae)
            train_corr2, test_corr2, test_rmse2, test_mae2 = format_trans(train_corr2, test_corr2, test_rmse2, test_mae2)
            train_corr3, test_corr3, test_rmse3, test_mae3 = format_trans(train_corr3, test_corr3, test_rmse3, test_mae3)
            train_corr4, test_corr4, test_rmse4, test_mae4 = format_trans(train_corr4, test_corr4, test_rmse4, test_mae4)
            train_corr5, test_corr5, test_rmse5, test_mae5 = format_trans(train_corr5, test_corr5, test_rmse5, test_mae5)
            train_corr6, test_corr6, test_rmse6, test_mae6 = format_trans(train_corr6, test_corr6, test_rmse6, test_mae6)
            train_corr7, test_corr7, test_rmse7, test_mae7 = format_trans(train_corr7, test_corr7, test_rmse7, test_mae7)
            train_corr8, test_corr8, test_rmse8, test_mae8 = format_trans(train_corr8, test_corr8, test_rmse8, test_mae8)
            
            df =[epoch,train_rmse, train_mae, train_corr, test_rmse, test_mae,test_corr,train_rmse2, train_mae2, train_corr2, test_rmse2, test_mae2,test_corr2,\
            train_rmse3, train_mae3, train_corr3, test_rmse3, test_mae3,test_corr3,train_rmse4, train_mae4, train_corr4, test_rmse4, test_mae4,test_corr4,\
            train_rmse5, train_mae5, train_corr5, test_rmse5, test_mae5,test_corr5,train_rmse6, train_mae6, train_corr6, test_rmse6, test_mae6,test_corr6, \
            train_rmse7, train_mae7, train_corr7, test_rmse7, test_mae7,test_corr7,train_rmse8, train_mae8, train_corr8, test_rmse8, test_mae8,test_corr8]
            df = torch.Tensor(df)
            print(df.shape)
            df = pd.DataFrame(np.reshape(df.cpu().detach().numpy(),(1,49)))
            df.to_csv(Accfile, mode='a+',header=None,index=None) 

            df_ori =[epoch, rmse_train, MAE_train, train_corr, rmse_test, MAE_test,test_corr, rmse_train2, MAE_train2, train_corr2, rmse_test2, MAE_test2,test_corr2, \
            rmse_train3, MAE_train3, train_corr3, rmse_test3, MAE_test3,test_corr3,rmse_train4, MAE_train4, train_corr4, rmse_test4, MAE_test4,test_corr4,\
            rmse_train5, MAE_train5, train_corr5, rmse_test5, MAE_test5,test_corr5,rmse_train6, MAE_train6, train_corr6, rmse_test6, MAE_test6,test_corr6,
            rmse_train7, MAE_train7, train_corr7, rmse_test7, MAE_test7,test_corr7,rmse_train8, MAE_train8, train_corr8, rmse_test8, MAE_test8,test_corr8]
            df_ori = torch.Tensor(df_ori)
            print(df_ori.shape)
            df_ori = pd.DataFrame(np.reshape(df_ori.cpu().detach().numpy(),(1,49)))
            df_ori.to_csv(Accfile2, mode='a+',header=None,index=None) 

           
    write_attention(timefile, train_attention, train_node_index, fold, task)
    write_attention(timefile, train_attention2, train_node_index2, fold, task2)
    write_attention(timefile, train_attention3, train_node_index3, fold, task3)
    write_attention(timefile, train_attention4, train_node_index4, fold, task4)
    write_attention(timefile, train_attention5, train_node_index5, fold, task5)
    write_attention(timefile, train_attention6, train_node_index6, fold, task6)
    write_attention(timefile, train_attention7, train_node_index7, fold, task7)
    write_attention(timefile, train_attention8, train_node_index8, fold, task8)
    write_attention2(timefile, train_attention9, fold, 'joint')
  

    df_predict = {'Predicted':train_out, 'Actual':train_true, 'Predicted2':train_out2, 'Actual2':train_true2, 'Predicted3':train_out3, 'Actual3':train_true, 
    'Predicted4':train_out4, 'Actual4':train_true4, 'Predicted5':train_out5, 'Actual5':train_true5, 'Predicted6':train_out6, 'Actual6':train_true6,\
    'Predicted7':train_out7, 'Actual7':train_true7,'Predicted8':train_out8, 'Actual8':train_true8}
    df_predict = pd.DataFrame(data=df_predict, dtype=np.float32)
    df_predict.to_csv(Labelfile, mode='a+', header=True) 
    df_predict2 = {'Predicted':test_out, 'Actual':test_true, 'Predicted2':test_out2, 'Actual2':test_true2,'Predicted':test_out, 'Actual':test_true,\
    'Predicted4':test_out4, 'Actual4':test_true4,'Predicted5':test_out5, 'Actual5':test_true5,'Predicted6':test_out6, 'Actual6':test_true6, \
    'Predicted7':test_out7, 'Actual7':test_true7, 'Predicted8':test_out8, 'Actual8':test_true8}
    df_predict2 = pd.DataFrame(data=df_predict2, dtype=np.float32)
    df_predict2.to_csv(Labelfile_test, mode='a+', header=True)
    
    df_predict = {'Predicted':train_outt, 'Actual':train_truet, 'Predicted2':train_outt2, 'Actual2':train_truet2, 'Predicted3':train_outt3, 'Actual3':train_truet3, \
    'Predicted4':train_outt4, 'Actual4':train_truet4, 'Predicted5':train_outt5, 'Actual5':train_truet5, 'Predicted6':train_outt6, 'Actual6':train_truet6, \
    'Predicted7':train_outt7, 'Actual7':train_truet7, 'Predicted8':train_outt8, 'Actual8':train_truet8}
    df_predict = pd.DataFrame(data=df_predict, dtype=np.float32)
    df_predict.to_csv(Labelfile2, mode='a+', header=True) 
    df_predict2 = {'Predicted':test_outt, 'Actual':test_truet, 'Predicted2':test_outt2, 'Actual2':test_truet2, 'Predicted3':test_outt3, 'Actual3':test_truet3, \
    'Predicted4':test_outt4, 'Actual4':test_truet4, 'Predicted5':test_outt5, 'Actual5':test_truet5, 'Predicted6':test_outt6, 'Actual6':test_truet6, \
    'Predicted7':test_outt7, 'Actual7':test_truet7, 'Predicted8':test_outt8, 'Actual8':test_truet8}
    df_predict2 = pd.DataFrame(data=df_predict2, dtype=np.float32)
    df_predict2.to_csv(Labelfile_test2, mode='a+', header=True)

    test_corr_s, test_rmse_s, test_mae_s = app(test_corr_s, test_rmse_s, test_mae_s, test_corr,  rmse_test, MAE_test)
    test_corr_s2, test_rmse_s2, test_mae_s2 = app(test_corr_s2, test_rmse_s2, test_mae_s2, test_corr2,  rmse_test2, MAE_test2)
    test_corr_s3, test_rmse_s3, test_mae_s3 = app(test_corr_s3, test_rmse_s3, test_mae_s3, test_corr3,  rmse_test3, MAE_test3)
    test_corr_s4, test_rmse_s4, test_mae_s4 = app(test_corr_s4, test_rmse_s4, test_mae_s4, test_corr4,  rmse_test4, MAE_test4)
    test_corr_s5, test_rmse_s5, test_mae_s5 = app(test_corr_s5, test_rmse_s5, test_mae_s5, test_corr5,  rmse_test5, MAE_test5)
    test_corr_s6, test_rmse_s6, test_mae_s6 = app(test_corr_s6, test_rmse_s6, test_mae_s6, test_corr6,  rmse_test6, MAE_test6)
    test_corr_s7, test_rmse_s7, test_mae_s7 = app(test_corr_s7, test_rmse_s7, test_mae_s7, test_corr7,  rmse_test7, MAE_test7)
    test_corr_s8, test_rmse_s8, test_mae_s8 = app(test_corr_s8, test_rmse_s8, test_mae_s8, test_corr8,  rmse_test8, MAE_test8)

    pred_out = np.concatenate((pred_out, test_outt), axis=0)
    true_out = np.concatenate((true_out, test_truet), axis=0)
    pred_out2 = np.concatenate((pred_out2, test_outt2), axis=0)
    true_out2 = np.concatenate((true_out2, test_truet2), axis=0)
    pred_out3 = np.concatenate((pred_out3, test_outt3), axis=0)
    true_out3 = np.concatenate((true_out3, test_truet3), axis=0)
    pred_out4 = np.concatenate((pred_out4, test_outt4), axis=0)
    true_out4 = np.concatenate((true_out4, test_truet4), axis=0)
    pred_out5 = np.concatenate((pred_out5, test_outt5), axis=0)
    true_out5 = np.concatenate((true_out5, test_truet5), axis=0)
    pred_out6 = np.concatenate((pred_out6, test_outt6), axis=0)
    true_out6 = np.concatenate((true_out6, test_truet6), axis=0)
    pred_out7 = np.concatenate((pred_out7, test_outt7), axis=0)
    true_out7 = np.concatenate((true_out7, test_truet7), axis=0)
    pred_out8 = np.concatenate((pred_out8, test_outt8), axis=0)
    true_out8 = np.concatenate((true_out8, test_truet8), axis=0)

corr, rmse, mae = compute_corr(test_corr_s, test_rmse_s, test_mae_s)
corr2, rmse2, mae2 = compute_corr(test_corr_s2, test_rmse_s2, test_mae_s2)
corr3, rmse3, mae3 = compute_corr(test_corr_s3, test_rmse_s3, test_mae_s3)
corr4, rmse4, mae4 = compute_corr(test_corr_s4, test_rmse_s4, test_mae_s4)
corr5, rmse5, mae5 = compute_corr(test_corr_s5, test_rmse_s5, test_mae_s5)
corr6, rmse6, mae6 = compute_corr(test_corr_s6, test_rmse_s6, test_mae_s6)
corr7, rmse7, mae7 = compute_corr(test_corr_s7, test_rmse_s7, test_mae_s7)
corr8, rmse8, mae8 = compute_corr(test_corr_s8, test_rmse_s8, test_mae_s8)

print(corr, rmse, mae)
print(corr2, rmse2, mae2)
print(corr3, rmse3, mae3) 
print(corr4, rmse4, mae4) 
print(corr5, rmse5, mae5) 
print(corr6, rmse6, mae6)
print(corr7, rmse7, mae7)
print(corr8, rmse8, mae8)

final = [corr, rmse, mae, corr2, rmse2, mae2, corr3, rmse3, mae3, corr4, rmse4, mae4, corr5, rmse5, mae5, corr6, rmse6, mae6, corr7, rmse7, mae7, corr8, rmse8, mae8]
final = torch.Tensor(final)
final = pd.DataFrame(final)
final.to_csv(finalfile, mode='a+', header=True)




