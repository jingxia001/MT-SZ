import numpy as np
import torch
from torch_geometric.data import Data
from scipy.sparse import coo_matrix
from sklearn import preprocessing
import heapq

def get_data(connpath, scorepath, feature = False):
    all_data = np.load(connpath)
    all_score = np.load(scorepath)
    
    return all_data, all_score

# Using the PyTorch Geometric's Data class to load the data into the Data class needed to create the dataset

def create_dataset(data, indexx, features = None):
    dataset_list = []
    n = data.shape[0]
    kk = 26
    for i in range(len(data)):
      
        feature_matrix_ori = np.array(data[i])
        #print(feature_matrix_ori.shape)
        #target_scaler = preprocessing.StandardScaler().fit(feature_matrix_ori.reshape(-1,1))
        #feature_matrix_ori = target_scaler.transform(feature_matrix_ori.reshape(-1,1))[:,0]    
        #feature_matrix_ori = feature_matrix_ori.reshape(116,116)
        #feature_matrix_ori2 = (feature_matrix_ori-np.min(feature_matrix_ori))/(np.max(feature_matrix_ori)-np.min(feature_matrix_ori))
        feature_matrix_ori2 =  feature_matrix_ori/np.max(feature_matrix_ori)
    #
        feature_matrix = feature_matrix_ori2#[~np.eye(feature_matrix_ori2.shape[0],dtype=bool)].reshape(feature_matrix_ori2.shape[0],-1)
        
        edge_index_coo = np.triu_indices(264, k=1)
        edge_adj = np.zeros((264, 264))
        for ii in range(len(feature_matrix_ori2[1])):
            index_max = heapq.nlargest(kk, range(len(feature_matrix_ori[ii])),feature_matrix_ori[ii].take)
            edge_adj[ii][index_max] = feature_matrix_ori2[ii][index_max]
        
        edge_weight = edge_adj[edge_index_coo]
        edge_index_coo = torch.tensor(edge_index_coo)
       
        if features != None:
            feature_matrix = features[i][0]
       
        graph_data = Data(x = torch.tensor(feature_matrix, dtype = torch.float32), edge_index=edge_index_coo, edge_weight=torch.tensor(edge_weight, dtype = torch.float32), y = torch.tensor(indexx[i]))
        dataset_list.append(graph_data)
 
    return dataset_list

