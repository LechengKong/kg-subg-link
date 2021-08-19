from .rgcn_model import RGCN
from dgl import mean_nodes
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
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
        self.sim_rel_tail_emb = nn.Embedding(self.params.num_rels, self.params.rel_emb_dim, sparse=False)
        self.sim_rel_head_emb = nn.Embedding(self.params.num_rels, self.params.rel_emb_dim, sparse=False)

        self.param_set_dim = self.params.emb_dim*self.params.num_gcn_layers

        # if params.node_attn:
        self.A = nn.Linear(self.param_set_dim+self.params.rel_emb_dim, self.params.emb_dim)
        self.B = nn.Linear(self.params.emb_dim, self.params.num_gcn_layers)
        self.C = nn.Linear(self.param_set_dim*3+self.params.rel_emb_dim*2, self.params.emb_dim)
        self.D = nn.Linear(self.params.emb_dim, self.params.num_gcn_layers)
        self.E = nn.Linear(self.param_set_dim*3+self.params.rel_emb_dim*2, self.params.emb_dim)
        self.G = nn.Linear(self.params.emb_dim, self.params.num_gcn_layers)

        dim_base_count = params.add_ht_emb*2 + params.sister_node_focus*1 + 1

        self.ht_layer = nn.Linear(self.param_set_dim*6+self.params.rel_emb_dim*2, self.param_set_dim)
        self.ht_rl_layer = nn.ReLU()

        self.fc_layer = nn.Linear(dim_base_count * self.param_set_dim + self.params.rel_emb_dim * 1, 16)
        self.rl_layer = nn.ReLU()
        self.output_layer = nn.Linear(16, 1)

    def forward(self, data):
        g, rel_labels = data
        g.ndata['h'] = self.gnn(g)
        
        bnn = g.batch_num_nodes()
        pre_sum = torch.cumsum(bnn, dim=0)
        ind_arr = torch.zeros(pre_sum[-1], dtype=torch.int64)
        ind_arr[pre_sum[:-1]] = 1
        ind_arr = torch.cumsum(ind_arr, dim=0)

        head_ids = (g.ndata['id'] == 1).nonzero().squeeze(1)

        tail_ids = (g.ndata['id'] == 2).nonzero().squeeze(1)

        if self.params.node_attn:
            g.ndata['beta'] = torch.sigmoid(self.B(F.relu(self.A(torch.cat(
                [g.ndata['repr'].view(-1, self.params.num_gcn_layers*self.params.emb_dim),
                self.rel_emb(g.ndata['t_label']).view(-1, self.params.rel_emb_dim)], dim=1))))).unsqueeze(2)
            # g.ndata['beta'] = torch.sigmoid(self.B(F.relu(self.A(torch.cat(
            #     [g.ndata['repr'].view(-1, self.params.num_gcn_layers*self.params.emb_dim),
            #     g.ndata['repr'][head_ids][ind_arr].view(-1, self.params.num_gcn_layers*self.params.emb_dim),
            #     g.ndata['repr'][tail_ids][ind_arr].view(-1, self.params.num_gcn_layers*self.params.emb_dim),
            #     self.rel_emb(g.ndata['t_label']).view(-1, self.params.rel_emb_dim)], dim=1))))).unsqueeze(2)
            # print(g.ndata['beta'].flatten())
            # print(g.ndata['beta'].size())
            g_out = mean_nodes(g, 'repr', 'beta')
        else:
            g_out = mean_nodes(g, 'repr')
        # print(g.num_nodes())
        

        g_rep = g_out.view(-1, self.param_set_dim)

        if self.params.add_ht_emb:
            head_embs = g.ndata['repr'][head_ids]
            tail_embs = g.ndata['repr'][tail_ids]
            g_rep = torch.cat([g_rep,
                                head_embs.view(-1, self.param_set_dim),
                               tail_embs.view(-1, self.param_set_dim)], dim=1)

        if self.params.sister_node_focus:
            g.ndata['head_sister'] = g.ndata['head_sister'].unsqueeze(2)
            g.ndata['tail_sister'] = g.ndata['tail_sister'].unsqueeze(2)
            g.ndata['gamma'] = torch.sigmoid(self.D(F.relu(self.C(torch.cat(
                [g.ndata['repr'].view(-1, self.params.num_gcn_layers*self.params.emb_dim),
                g.ndata['repr'][head_ids][ind_arr].view(-1, self.params.num_gcn_layers*self.params.emb_dim),
                g.ndata['repr'][tail_ids][ind_arr].view(-1, self.params.num_gcn_layers*self.params.emb_dim),
                self.sim_rel_head_emb(g.ndata['head_sister_type']).view(-1, self.params.rel_emb_dim),
                self.sim_rel_head_emb(g.ndata['t_label']).view(-1, self.params.rel_emb_dim)], dim=1))))).unsqueeze(2)
            g.ndata['delta'] = torch.sigmoid(self.G(F.relu(self.E(torch.cat(
                [g.ndata['repr'].view(-1, self.params.num_gcn_layers*self.params.emb_dim),
                g.ndata['repr'][tail_ids][ind_arr].view(-1, self.params.num_gcn_layers*self.params.emb_dim),
                g.ndata['repr'][head_ids][ind_arr].view(-1, self.params.num_gcn_layers*self.params.emb_dim),
                self.sim_rel_tail_emb(g.ndata['tail_sister_type']).view(-1, self.params.rel_emb_dim),
                self.sim_rel_tail_emb(g.ndata['t_label']).view(-1, self.params.rel_emb_dim)], dim=1))))).unsqueeze(2)
            # g.ndata['gamma'] = torch.sigmoid(self.D(F.relu(self.C(torch.cat(
            #     [g.ndata['repr'].view(-1, self.params.num_gcn_layers*self.params.emb_dim),
            #     g.ndata['repr'][head_ids][ind_arr].view(-1, self.params.num_gcn_layers*self.params.emb_dim)], dim=1))))).unsqueeze(2)
            # g.ndata['delta'] = torch.sigmoid(self.G(F.relu(self.E(torch.cat(
            #     [g.ndata['repr'].view(-1, self.params.num_gcn_layers*self.params.emb_dim),
            #     g.ndata['repr'][tail_ids][ind_arr].view(-1, self.params.num_gcn_layers*self.params.emb_dim)], dim=1))))).unsqueeze(2)
            # print(g.ndata['repr'].size())
            # print(mean_nodes(g, 'repr', 'beta').size())
            g.ndata['head_gamma'] = g.ndata['gamma']*g.ndata['head_sister']
            g.ndata['tail_delta'] = g.ndata['delta']*g.ndata['tail_sister']
            # print(g.ndata['head_gamma'].flatten())
            g_head_sister = mean_nodes(g, 'repr', 'head_gamma')
            g_tail_sister = mean_nodes(g, 'repr', 'tail_delta')
            g_ht = torch.cat([g_head_sister.view(-1, self.param_set_dim)*g.ndata['repr'][head_ids].view(-1, self.param_set_dim),
                               g_tail_sister.view(-1, self.param_set_dim)*g.ndata['repr'][tail_ids].view(-1, self.param_set_dim),
                               g_head_sister.view(-1, self.param_set_dim),
                               g_tail_sister.view(-1, self.param_set_dim),
                                head_embs.view(-1, self.param_set_dim),
                               tail_embs.view(-1, self.param_set_dim),
                            self.sim_rel_head_emb(rel_labels),
                            self.sim_rel_tail_emb(rel_labels)], dim=1)

            ht_out = self.ht_rl_layer(self.ht_layer(g_ht))
            g_rep = torch.cat([g_rep,
                            ht_out], dim=1)
        g_rep = torch.cat([g_rep,
                            self.rel_emb(rel_labels)], dim=1)

        output = self.fc_layer(g_rep)
        output = self.rl_layer(output)
        output = self.output_layer(output)

        return output


class GraphClassifierSpe(nn.Module):
    def __init__(self, params, relation2id):  # in_dim, h_dim, rel_emb_dim, out_dim, num_rels, num_bases):
        super().__init__()

        self.params = params
        self.relation2id = relation2id

        self.gnn = RGCN(params)  # in_dim, h_dim, h_dim, num_rels, num_bases)
        self.rel_emb = nn.Embedding(self.params.num_rels, self.params.rel_emb_dim, sparse=False)

        self.param_set_dim = self.params.emb_dim*self.params.num_gcn_layers

        dim_base_count = params.add_ht_emb*2 + 1

        if self.params.simple_net:
            self.fc_layer = nn.Linear(self.params.eig_size, 16)
        else:
            self.fc_layer = nn.Linear(dim_base_count * self.param_set_dim + self.params.rel_emb_dim * 1+ self.params.eig_size, 16)
        self.rl_layer = nn.ReLU()
        self.output_layer = nn.Linear(16, 1)

    def forward(self, data):
        g, rel_labels, spe = data
        if self.params.simple_net:
            output = self.fc_layer(spe)
            output = self.rl_layer(output)
            output = self.output_layer(output)
            return output
        g.ndata['h'] = self.gnn(g)

        head_ids = (g.ndata['id'] == 1).nonzero().squeeze(1)

        tail_ids = (g.ndata['id'] == 2).nonzero().squeeze(1)

        g_out = mean_nodes(g, 'repr')
        g_rep = g_out.view(-1, self.param_set_dim)

        if self.params.add_ht_emb:
            head_embs = g.ndata['repr'][head_ids]
            tail_embs = g.ndata['repr'][tail_ids]
            g_rep = torch.cat([g_rep,
                                head_embs.view(-1, self.param_set_dim),
                               tail_embs.view(-1, self.param_set_dim)], dim=1)

        g_rep = torch.cat([g_rep,
                            self.rel_emb(rel_labels),
                            spe], dim=1)

        output = self.fc_layer(g_rep)
        output = self.rl_layer(output)
        output = self.output_layer(output)

        return output
