import torch.nn.functional as F
from torch_geometric.nn import GATConv, LayerNorm,BatchNorm,GCNConv
from torch_geometric.nn import TopKPooling,SAGPooling,global_max_pool, global_mean_pool
import torch.nn as nn
from torch.nn import Linear
import torch


"""
Appling pyG lib
"""
class GATModel(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, output_dim, drop, nheads=8, k=8):
        super(GATModel, self).__init__()

        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.drop = drop
        self.heads = nheads
        self.k = k

        ###GAT
        self.conv1 = GATConv(node_feature_dim, hidden_dim, heads=nheads)
        self.conv2 = GATConv(nheads * hidden_dim, hidden_dim, heads=nheads)
        self.conv3 = GATConv(nheads * hidden_dim, hidden_dim, heads=nheads, concat=False)
        self.norm1 = LayerNorm(nheads * hidden_dim)
        self.norm2 = LayerNorm(nheads * hidden_dim)
        self.norm3 = LayerNorm(hidden_dim)
        # #GCN
        self.conv4 = GCNConv(node_feature_dim,hidden_dim)
        self.conv5 = GCNConv(hidden_dim,hidden_dim)
        self.conv6 = GCNConv(hidden_dim, hidden_dim)
        self.norm4 = LayerNorm(hidden_dim)
        # self.norm2 = LayerNorm(hidden_dim)
        # self.norm3 = LayerNorm(hidden_dim)

       
        self.lin00 = Linear(nheads * hidden_dim, hidden_dim)
        self.lin01 = Linear(hidden_dim, hidden_dim)
      

        self.proj_esm = Linear(1280, 512)
        self.proj_esm2 = Linear(512, 256)
        self.proj_esm3 = Linear(256, 128)
        self.proj_esm4 = Linear(128, 40)
        self.lin0 = Linear(2 * hidden_dim, hidden_dim)
        self.lin1 = Linear(hidden_dim, hidden_dim)
        self.lin2 = Linear(hidden_dim, hidden_dim)
        self.lin = Linear(hidden_dim, output_dim)

        self.topk_pool = TopKPooling(hidden_dim, ratio=k)
        self.sag_pool = SAGPooling(hidden_dim, ratio=k)
        
        self._reset_parameters()

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self,x,edge_index, batch):

        
        y=x
        y = self.conv4(y, edge_index)
        y=F.relu(y)
        y = global_max_pool(y, batch)
       
        z2 = y
   
        
        
        x = self.conv1(x, edge_index)
        x = self.norm1(x, batch)
        x = F.relu(x)
        x = F.dropout(x, p=self.drop, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.norm2(x, batch)
        x = F.relu(x)
        x = F.dropout(x, p=self.drop, training=self.training)
        

        x = self.conv3(x, edge_index)
        x = self.norm3(x, batch)
        x = F.relu(x)
       
        x1 = global_max_pool(x, batch)  # [batch_size, hidden_channels]

        z1 = x1
        x3 = torch.cat([z1, z2], dim=1)
        z = x3 
        x3 = self.lin0(x3)
        x3 = F.relu(x3)
        

        x3 = self.lin1(x3)
        x3 = F.relu(x3)
        
        x3 = self.lin2(x3)
        x3 = F.relu(x3)
        

        # z = x3  # extract last layer features

        x3 = self.lin(x3)

        return x3,z


