import numpy as np
import torch
from torch import nn

class MF(nn.Module):
    def __init__(self, num_users, num_items, emb_size, num_features):
        super(MF, self).__init__()
        
        # hyperparameters
        self.num_features = num_features
        self.emb_size = emb_size
        
        # embeddings
        self.user_embedding = nn.Embedding(num_users, emb_size)
        self.item_embedding = nn.Embedding(num_items, emb_size)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        
    def forward(self, dense_features, user, item, cat_features):
        user_vector = self.user_embedding(user)
        item_vector = self.item_embedding(item)
        sparse_features = torch.concat([user_vector,item_vector,cat_features], dim=1)
        all_vectors = torch.concat([dense_features,sparse_features], dim=1)
        all_vectors = nn.BatchNorm1d(self.emb_size*2 + self.num_features)(all_vectors)
        
        # MF
        b_u = self.user_bias(user)
        b_i = self.item_bias(item)
        w_outputs = (user_vector * item_vector).sum(axis=1) + np.squeeze(b_u) + np.squeeze(b_i)

        return w_outputs.unsqueeze(1)
    
class FM(nn.Module):
    def __init__(self, num_users, num_items, emb_size, num_features):
        super(FM, self).__init__()
        
        # hyperparameters
        self.num_features = num_features
        self.emb_size = emb_size
    
        # embeddings
        self.user_embedding = nn.Embedding(num_users, emb_size)
        self.item_embedding = nn.Embedding(num_items, emb_size)
        
        # weights and biases
        self.weights = nn.Parameter(torch.randn(emb_size*2 + num_features, 10), requires_grad=True)

        # intialize layers
        self.lin = nn.Linear(emb_size*2 + num_features, 1)

    def forward(self, dense_features, user, item, cat_features):
        user_vector = self.user_embedding(user)
        item_vector = self.item_embedding(item)
        sparse_features = torch.concat([user_vector,item_vector,cat_features], dim=1)
        all_vectors = torch.concat([dense_features,sparse_features], dim=1)
        all_vectors = nn.BatchNorm1d(self.emb_size*2 + self.num_features)(all_vectors)
                
        # FM
        out_1 = torch.matmul(all_vectors, self.weights).pow(2).sum(1, keepdim=True) #S_1^2
        out_2 = torch.matmul(all_vectors.pow(2), self.weights.pow(2)).sum(1, keepdim=True) # S_2
        out_inter = 0.5*(out_1 - out_2)
        out_lin = self.lin(all_vectors)
        w_outputs = out_inter + out_lin
        
        return w_outputs

class DeepFM(nn.Module):
    def __init__(self, num_users, num_items, emb_size, deep_param, num_features, dropout=0.3):
        super(DeepFM, self).__init__()
        
        # hyperparameters
        self.num_features = num_features
        self.emb_size = emb_size
    
        # embeddings
        self.user_embedding = nn.Embedding(num_users, emb_size)
        self.item_embedding = nn.Embedding(num_items, emb_size)
        
        # weights and biases
        self.weights = nn.Parameter(torch.randn(emb_size*2 + num_features, 10), requires_grad=True)

        # initialize layers
        self.deep = nn.Sequential(
            nn.Linear(emb_size*2 + num_features, deep_param[0]),
            nn.BatchNorm1d(num_features=deep_param[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(deep_param[0], deep_param[1]),
            nn.BatchNorm1d(num_features=deep_param[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(deep_param[1], deep_param[2]),
            nn.BatchNorm1d(num_features=deep_param[2]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(deep_param[2],1)
        )
        self.lin = nn.Linear(emb_size*2 + num_features, 1)
        self.activation = nn.Linear(2,1)

    def forward(self, dense_features, user, item, cat_features):
        user_vector = self.user_embedding(user)
        item_vector = self.item_embedding(item)
        sparse_features = torch.concat([user_vector,item_vector,cat_features], dim=1)
        all_vectors = torch.concat([dense_features,sparse_features], dim=1)
        all_vectors = nn.BatchNorm1d(self.emb_size*2 + self.num_features)(all_vectors)
                
        # wide (FM)
        out_1 = torch.matmul(all_vectors, self.weights).pow(2).sum(1, keepdim=True) #S_1^2
        out_2 = torch.matmul(all_vectors.pow(2), self.weights.pow(2)).sum(1, keepdim=True) # S_2
        out_inter = 0.5*(out_1 - out_2)
        out_lin = self.lin(all_vectors)
        w_outputs = out_inter + out_lin
        
        # deep    
        d_output = self.deep(all_vectors)
        
#         return w_outputs + d_output + c_outputs self.bias_output
        return self.activation(torch.concat([w_outputs,d_output], dim=1))

class RecSys(nn.Module):
    def __init__(self, num_users, num_items, emb_size, deep_param, num_features, layer_num=3, dropout=0.3):
        super(RecSys, self).__init__()
        
        # hyperparameters
        self.num_features = num_features
        self.emb_size = emb_size
        self.layer_num = layer_num
    
        # embeddings
        self.user_embedding = nn.Embedding(num_users, emb_size)
        self.item_embedding = nn.Embedding(num_items, emb_size)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        
        # weights and biases
        self.kernels = nn.Parameter(torch.Tensor(layer_num, emb_size*2 + num_features, emb_size*2 + num_features))
        self.bias_cn = nn.Parameter(torch.Tensor(layer_num, emb_size*2 + num_features, 1))
        for i in range(self.kernels.shape[0]):
            nn.init.xavier_normal_(self.kernels[i])
        for i in range(self.bias_cn.shape[0]):
            nn.init.zeros_(self.bias_cn[i])
        self.weights = nn.Parameter(torch.randn(emb_size*2 + num_features, 10), requires_grad=True)

        # initialize layers
        self.deep = nn.Sequential(
            nn.Linear(emb_size*2 + num_features, deep_param[0]),
            nn.BatchNorm1d(num_features=deep_param[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(deep_param[0], deep_param[1]),
            nn.BatchNorm1d(num_features=deep_param[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(deep_param[1], deep_param[2]),
            nn.BatchNorm1d(num_features=deep_param[2]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(deep_param[2],1)
        )
        self.lin = nn.Linear(emb_size*2 + num_features, 1)
        self.activation = nn.Linear(2,1)

    def forward(self, dense_features, user, item, cat_features):
        user_vector = self.user_embedding(user)
        item_vector = self.item_embedding(item)
        sparse_features = torch.concat([user_vector,item_vector,cat_features], dim=1)
        all_vectors = torch.concat([dense_features,sparse_features], dim=1)
        all_vectors = nn.BatchNorm1d(self.emb_size*2 + self.num_features)(all_vectors)
        
        # wide (FM)
        out_1 = torch.matmul(all_vectors, self.weights).pow(2).sum(1, keepdim=True) #S_1^2
        out_2 = torch.matmul(all_vectors.pow(2), self.weights.pow(2)).sum(1, keepdim=True) # S_2
        out_inter = 0.5*(out_1 - out_2)
        out_lin = self.lin(all_vectors)
        w_outputs = out_inter + out_lin
        
        # cross
        x_0 = all_vectors.unsqueeze(2)
        x_l = x_0
        for i in range(self.layer_num):
            xl_w = torch.matmul(self.kernels[i], x_l)  # W * xi  (bs, in_features, 1)
            dot_ = xl_w + self.bias_cn[i]  # W * xi + b
            x_l = x_0 * dot_ + x_l  # x0 · (W * xi + b) +xl  Hadamard-product
        c_outputs = torch.squeeze(x_l, dim=2)

        # deep    
        d_output = self.deep(c_outputs)
        
#         return w_outputs + d_output + c_outputs self.bias_output
        return self.activation(torch.concat([w_outputs,d_output], dim=1))