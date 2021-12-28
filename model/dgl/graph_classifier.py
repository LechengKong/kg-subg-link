from .rgcn_model import RGCN
from dgl import mean_nodes
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import copy
from torch.nn import LSTM
"""
File based off of dgl tutorial on RGCN
Source: https://github.com/dmlc/dgl/tree/master/examples/pytorch/rgcn
"""


class GraphClassifier(nn.Module):
    def __init__(self, params, relation2id):  # in_dim, h_dim, rel_emb_dim, out_dim, num_rels, num_bases):
        super().__init__()

        self.params = params
        self.relation2id = relation2id

        self.gnn = RGCN(params)  # in_dim, h_dim, h_dim, num_rels, num_bases)
        self.rel_emb = nn.Embedding(self.params.num_rels, self.params.rel_emb_dim, sparse=False)
        self.aug_rel_weight = nn.Parameter(torch.Tensor(self.params.aug_num_rels, self.params.rel_emb_dim))
        nn.init.xavier_uniform_(self.aug_rel_weight, gain=nn.init.calculate_gain('relu'))

        self.param_set_dim = self.params.emb_dim*self.params.num_gcn_layers

        dim_base_count = params.add_ht_emb*2 + params.sister_node_focus*1

        self.fc_layer = nn.Linear(dim_base_count * self.param_set_dim + self.params.rel_emb_dim * 1 + self.params.use_root_dist, 16)
        self.rl_layer = nn.ReLU()
        self.output_layer = nn.Linear(16, 1)

    def forward(self, data):
        g, rel_labels = data
        # ine = g.in_edges(0, 'all')
        # print(ine[0])
        # print(g.edata['type'][ine[2]])
        # ine = g.in_edges(1, 'all')
        # print(ine[0])
        # print(g.edata['type'][ine[2]])
        if self.params.use_neighbor_feature:
            g.ndata['feat'] = torch.cat((g.ndata['feat'],torch.mm(g.ndata['ratio'], self.aug_rel_weight)),1)
        g.ndata['h'] = self.gnn(g)
        
        bnn = g.batch_num_nodes()
        pre_sum = torch.cumsum(bnn, dim=0)
        ind_arr = torch.zeros(pre_sum[-1], dtype=torch.int64)
        ind_arr[pre_sum[:-1]] = 1
        ind_arr = torch.cumsum(ind_arr, dim=0)

        head_ids = (g.ndata['id'] == 1).nonzero().squeeze(1)
        # print(head_ids)

        tail_ids = (g.ndata['id'] == 2).nonzero().squeeze(1)
        g_out = mean_nodes(g, 'repr')
        # print(g.num_nodes())
        

        # g_rep = g_out.view(-1, self.param_set_dim)

        if self.params.add_ht_emb:
            head_embs = g.ndata['repr'][head_ids]
            # print(head_embs[0])
            tail_embs = g.ndata['repr'][tail_ids]
            # g_rep = torch.cat([g_rep,
            #                     head_embs.view(-1, self.param_set_dim),
            #                    tail_embs.view(-1, self.param_set_dim)], dim=1)
            g_rep = torch.cat([head_embs.view(-1, self.param_set_dim),
                               tail_embs.view(-1, self.param_set_dim)], dim=1)

        if self.params.use_root_dist:
            root_dist = g.ndata['rt_dist'][head_ids]
            g_rep = torch.cat([g_rep,
                            root_dist], dim=1)
        g_rep = torch.cat([g_rep,
                            self.rel_emb(rel_labels)], dim=1)
        output = self.fc_layer(g_rep)
        output = self.rl_layer(output)
        output = self.output_layer(output)
        return output


class FCLayers(nn.Module):
    def __init__(self, layers, input_dim, h_units, activation=F.relu):
        super().__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        for i in range(layers):
            if i == 0:
                self.layers.append(nn.Linear(input_dim, h_units[i]))
            else:
                self.layers.append(nn.Linear(h_units[i-1], h_units[i]))

    def forward(self, x):
        output = x
        for i, layer in enumerate(self.layers):
            output = layer(output)
            if i < len(self.layers)-1:
                output = self.activation(output)
        return output


class GraphClassifierWhole(nn.Module):
    def __init__(self, params, relation2id):  # in_dim, h_dim, rel_emb_dim, out_dim, num_rels, num_bases):
        super().__init__()

        self.params = params
        self.relation2id = relation2id

        self.gnn = RGCN(params)  # in_dim, h_dim, h_dim, num_rels, num_bases)

        self.rel_emb = nn.Embedding(self.params.num_rels, self.params.rel_emb_dim, sparse=False)

        self.dist_emb = nn.Embedding(self.params.shortest_path_dist+1, self.params.emb_dim, sparse=False)

        self.param_set_dim = self.params.emb_dim*self.params.num_gcn_layers

        dim_base_count = 2

        if self.params.use_lstm:
            mid_dim = self.params.lstm_hidden_size*2
        elif self.params.use_deep_set:
            mid_dim = self.params.deep_set_dim
        else:
            mid_dim = self.param_set_dim

        self.link_fc = FCLayers(3, dim_base_count * self.param_set_dim + self.params.rel_emb_dim * 1 +2*self.params.inp_dim*self.params.concat_init_feat+mid_dim*self.params.use_mid_repr+self.params.use_dist_emb, [128, 64, 1])

        self.head_fc = FCLayers(1, mid_dim+self.params.emb_dim, [self.params.inp_dim])
        self.tail_fc = FCLayers(1, mid_dim+self.params.emb_dim, [self.params.inp_dim])
        if self.params.use_deep_set:
            self.deep_set = FCLayers(2, self.param_set_dim, [int(self.params.deep_set_dim/2),self.params.deep_set_dim])

        if self.params.use_lstm:
            self.RNN = LSTM(self.param_set_dim, hidden_size=self.params.lstm_hidden_size, bidirectional=True, batch_first=True)


    def forward(self, data):
        g, (links, dist, inter_count,edge_ids) = data
        h = self.graph_update(g)
        return self.mlp_update(g, links, dist, inter_count, edge_ids, h)

    def graph_update(self, g):
        h = self.gnn(g)
        return h

    def mlp_update(self, g, links, dist, inter_count, edge_ids, h):
        head_repr = g.ndata['repr'][links[:,0]]

        tail_repr = g.ndata['repr'][links[:,2]]
        rel_repr = self.rel_emb(links[:,1])
        mid_repr = g.ndata['repr'].view(-1,self.param_set_dim)[torch.abs(inter_count)]*torch.sign(inter_count+1).unsqueeze(2)
        # print(edge_ids)
        # edge_repr = self.rel_emb(g.edata['type'][torch.abs(edge_ids)]%self.params.num_rels)*torch.sign(edge_ids+1).unsqueeze(2)
        # mid_repr = torch.cat([mid_repr,edge_repr],dim=-1)
        if self.params.use_lstm:
            mid_repr, (_,_) = self.RNN(mid_repr)
            mid_repr = mid_repr[:,0,:]
        else:
            if self.params.use_deep_set:
                mid_repr = self.deep_set(mid_repr).sum(dim=1)
            else:
                mid_repr = mid_repr.sum(dim=1)
            mid_repr = mid_repr/torch.clamp(torch.sum(inter_count!=-1,dim=1),min=1).unsqueeze(1)
            # mid_repr = (g.ndata['repr'].view(-1,self.param_set_dim)[torch.abs(inter_count)]*torch.sign(inter_count+1).unsqueeze(2))
        # print(mid_repr.size())
        dist_repr = self.dist_emb(dist)
        if self.params.label_reg:
            head_pred = self.head_fc(torch.cat([mid_repr, dist_repr], dim=1))
            tail_pred = self.tail_fc(torch.cat([mid_repr, dist_repr], dim=1))
            head_init_feat = g.ndata['feat'][links[:,0]]
            tail_init_feat = g.ndata['feat'][links[:,2]]
            # head_init_feat = g.ndata['repr'].view(-1,self.param_set_dim)[links[:,0]]
            # tail_init_feat = g.ndata['repr'].view(-1,self.param_set_dim)[links[:,2]]
        else:
            head_pred = None
            tail_pred = None
            head_init_feat = None
            tail_init_feat = None

        g_rep = torch.cat([head_repr.view(-1, self.param_set_dim),
                            tail_repr.view(-1, self.param_set_dim),
                            rel_repr], dim=1)
        
        if self.params.use_mid_repr:
            g_rep = torch.cat([g_rep, mid_repr],dim=1)
        
        if self.params.concat_init_feat:
            g_rep = torch.cat([g_rep,
                            g.ndata['feat'][links[:,0]],
                            g.ndata['feat'][links[:,2]]], dim=1)
        if self.params.use_dist_emb:
            g_rep = torch.cat([g_rep,
                            dist.unsqueeze(1)], dim=1)
        output = self.link_fc(g_rep)

        return output, head_pred, tail_pred, head_init_feat, tail_init_feat


class GraphClassifierMulti(nn.Module):
    def __init__(self, params, relation2id):  # in_dim, h_dim, rel_emb_dim, out_dim, num_rels, num_bases):
        super().__init__()

        self.params = params
        self.relation2id = relation2id
        # params2 = copy.deepcopy(params)
        # params2.edge_dropout = 0
        # params2.num_gcn_layers = 5

        self.gnn = RGCN(params)  # in_dim, h_dim, h_dim, num_rels, num_bases)
        self.gnn2 = RGCN(params)

        self.rel_emb = nn.Embedding(self.params.num_rels, self.params.rel_emb_dim, sparse=False)

        self.dist_emb = nn.Embedding(self.params.shortest_path_dist+1, self.params.emb_dim, sparse=False)

        self.param_set_dim = self.params.emb_dim*self.params.num_gcn_layers
        self.fixed_edge_dropout = nn.Dropout(1-1/self.params.edge_split)
        self.drop_ratio = 1/self.params.edge_split

        dim_base_count = 2

        if self.params.use_lstm:
            mid_dim = self.params.lstm_hidden_size*2
        elif self.params.use_deep_set:
            mid_dim = self.params.deep_set_dim
        else:
            mid_dim = self.param_set_dim+self.params.edges_division

        self.link_fc = FCLayers(3, dim_base_count * self.param_set_dim + self.params.rel_emb_dim * 1 +2*self.params.inp_dim*self.params.concat_init_feat+mid_dim*self.params.use_mid_repr, [128, 64, 1])

        self.head_fc = FCLayers(1, mid_dim+self.params.emb_dim, [self.params.inp_dim])
        self.tail_fc = FCLayers(1, mid_dim+self.params.emb_dim, [self.params.inp_dim])
        self.edge_labeler = FCLayers(1, self.params.emb_dim, [self.params.edges_division])
        if self.params.use_deep_set:
            self.deep_set = FCLayers(2, self.param_set_dim+self.params.edges_division, [int(self.params.deep_set_dim/2),self.params.deep_set_dim])

        if self.params.use_lstm:
            self.RNN = LSTM(self.param_set_dim+self.params.edges_division, hidden_size=self.params.lstm_hidden_size, bidirectional=True, batch_first=True)


    def forward(self, data):
        g, (links, dist, inter_count, edge_ids) = data
        h = self.graph_update(g)
        return self.mlp_update(g, links, dist, inter_count, edge_ids, h)

    def graph_update(self, g):
        # g.ndata['op_final'] = (g.ndata['op_final']*g.ndata['rep_mask'].unsqueeze(2)).view(g.ndata['op_final'].size()[0],-1)
        h = self.gnn(g)
        # with torch.no_grad():
        for i in range(self.params.edge_rep):
            p = torch.zeros(g.num_edges(),1,device=g.device)+self.drop_ratio
            mask = torch.bernoulli(p)
            # mask = self.fixed_edge_dropout(torch.ones(g.num_edges(),1, device=g.device))
            # print(mask.sum())
            h = self.gnn(g, mask)
            if i ==0:
                g.ndata['op_final'] = h
                g.edata['rep_mask'] = mask
            else:
                g.ndata['op_final'] = torch.cat([g.ndata['op_final'],h],dim=1)
                g.edata['rep_mask'] = torch.cat([g.edata['rep_mask'],mask],dim=1)
        def get_null_edge_emb(edges):
            all_dst = edges.dst['op_final'].view(-1,self.params.edge_rep, self.params.emb_dim)
            edge_mask_neg = torch.logical_not(edges.data['rep_mask'])
            all_dst = self.edge_labeler(all_dst)
            # print(edge_mask_neg[0])
            edge_repr = (all_dst*edge_mask_neg.unsqueeze(2)).sum(dim=1)/torch.clamp(edge_mask_neg.sum(dim=1), min=1).unsqueeze(1)
            # print(edge_repr[20])
            # return {'edge_div':self.edge_labeler(edge_repr)}
            return {'edge_div':edge_repr}
        g.apply_edges(get_null_edge_emb)
        return h

    def mlp_update(self, g, links, dist, inter_count, edge_ids, h):
        head_repr = g.ndata['repr'][links[:,0]]

        tail_repr = g.ndata['repr'][links[:,2]]
        rel_repr = self.rel_emb(links[:,1])
        edge_repr = g.edata['edge_div'][torch.abs(edge_ids)]*torch.sign(edge_ids+1).unsqueeze(2)
        # print(edge_repr[0])
        # print(edge_repr)
        mid_repr = torch.cat([g.ndata['repr'].view(-1,self.param_set_dim)[torch.abs(inter_count)]*torch.sign(inter_count+1).unsqueeze(2),edge_repr],dim=-1)
        if self.params.use_lstm:
            mid_repr, (_,_) = self.RNN(mid_repr)
            mid_repr = mid_repr[:,0,:]
        else:
            if self.params.use_deep_set:
                mid_repr = self.deep_set(mid_repr).sum(dim=1)
            else:
                mid_repr = mid_repr.sum(dim=1)
            mid_repr = mid_repr/torch.clamp(torch.sum(inter_count!=-1,dim=1),min=1).unsqueeze(1)
            # mid_repr = (g.ndata['repr'].view(-1,self.param_set_dim)[torch.abs(inter_count)]*torch.sign(inter_count+1).unsqueeze(2))
        # print(mid_repr.size())
        dist_repr = self.dist_emb(dist)
        if self.params.label_reg:
            head_pred = self.head_fc(torch.cat([mid_repr, dist_repr], dim=1))
            tail_pred = self.tail_fc(torch.cat([mid_repr, dist_repr], dim=1))
            head_init_feat = g.ndata['feat'][links[:,0]]
            tail_init_feat = g.ndata['feat'][links[:,2]]
            # head_init_feat = g.ndata['repr'].view(-1,self.param_set_dim)[links[:,0]]
            # tail_init_feat = g.ndata['repr'].view(-1,self.param_set_dim)[links[:,2]]
        else:
            head_pred = None
            tail_pred = None
            head_init_feat = None
            tail_init_feat = None

        g_rep = torch.cat([head_repr.view(-1, self.param_set_dim),
                            tail_repr.view(-1, self.param_set_dim),
                            rel_repr], dim=1)
        
        if self.params.use_mid_repr:
            g_rep = torch.cat([g_rep, mid_repr],dim=1)
        
        if self.params.concat_init_feat:
            g_rep = torch.cat([g_rep,
                            g.ndata['feat'][links[:,0]],
                            g.ndata['feat'][links[:,2]]], dim=1)
        output = self.link_fc(g_rep)

        return output, head_pred, tail_pred, head_init_feat, tail_init_feat

