from .rgcn_model import RGCN
from dgl import mean_nodes
import torch.nn as nn
import torch
import torch.nn.functional as F
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

        self.param_set_dim = self.params.emb_dim*self.params.num_gcn_layers

        if params.node_attn:
            self.A = nn.Linear(self.param_set_dim+self.params.rel_emb_dim,self.params.emb_dim)
            self.B = nn.Linear(self.params.emb_dim, self.params.num_gcn_layers)

        dim_base_count = params.add_ht_emb*2 + params.sister_node_focus*2 + 1

        self.fc_layer = nn.Linear(dim_base_count * self.param_set_dim + self.params.rel_emb_dim, 1)

    def forward(self, data):
        g, rel_labels = data
        g.ndata['h'] = self.gnn(g)

        if self.params.node_attn:
            g.ndata['beta'] = torch.sigmoid(self.B(F.relu(self.A(torch.cat(
                [g.ndata['repr'].view(-1, self.params.num_gcn_layers*self.params.emb_dim),
                self.rel_emb(g.ndata['t_label']).view(-1, self.params.rel_emb_dim)], dim=1))))).unsqueeze(2)
            # print(g.ndata['beta'].size())
            g_out = mean_nodes(g, 'repr', 'beta')
        else:
            g_out = mean_nodes(g, 'repr')

        g_rep = g_out.view(-1, self.param_set_dim)

        if self.params.add_ht_emb:
            head_ids = (g.ndata['id'] == 1).nonzero().squeeze(1)
            head_embs = g.ndata['repr'][head_ids]

            tail_ids = (g.ndata['id'] == 2).nonzero().squeeze(1)
            tail_embs = g.ndata['repr'][tail_ids]
            g_rep = torch.cat([g_rep,
                                head_embs.view(-1, self.param_set_dim),
                               tail_embs.view(-1, self.param_set_dim)], dim=1)

        if self.params.sister_node_focus:
            g.ndata['head_sister'] = g.ndata['head_sister'].unsqueeze(2)
            g.ndata['tail_sister'] = g.ndata['tail_sister'].unsqueeze(2)
            g_head_sister = mean_nodes(g, 'repr', 'head_sister')
            g_tail_sister = mean_nodes(g, 'repr', 'tail_sister')
            g_rep = torch.cat([g_rep,
                               g_head_sister.view(-1, self.param_set_dim),
                               g_tail_sister.view(-1, self.param_set_dim),], dim=1)

        g_rep = torch.cat([g_rep,
                            self.rel_emb(rel_labels)], dim=1)

        output = self.fc_layer(g_rep)
        return output
