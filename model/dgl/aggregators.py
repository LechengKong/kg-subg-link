import abc
import torch.nn as nn
import torch
import torch.nn.functional as F


class MemAggregator(nn.Module):
    def __init__(self, emb_dim):
        super(MemAggregator, self).__init__()

    def forward(self, node):
        (N, D, C, R) = node.mailbox['nei_node_mem'].size()
        curr_emb = node.mailbox['curr_emb'][:, 0, :]  # (B, F)
        nei_msg = torch.bmm(node.mailbox['alpha'].transpose(1, 2), node.mailbox['msg']).squeeze(1)  # (B, F)
        # nei_msg, _ = torch.max(node.mailbox['msg'], 1)  # (B, F)
        curr_node_mem = node.mailbox['curr_node_mem'][:, 0]
        curr_rel_mem = node.mailbox['curr_rel_mem'][:, 0]
        node_mem = node.mailbox['nei_node_mem']
        rel_mem = node.mailbox['nei_rel_mem']
        head_rel_emb = node.mailbox['head_rel_emb']
        head_emb = node.mailbox['head_emb']
        combined_node = torch.cat([node_mem, head_emb.unsqueeze(2)], dim=2)
        added_rel = rel_mem+head_rel_emb.unsqueeze(2)
        combined_rel = torch.cat([added_rel, head_rel_emb.unsqueeze(2)], dim=2)
        total_node = torch.cat([curr_node_mem, combined_node.view(N, -1, R)], axis=1)
        total_rel = torch.cat([curr_rel_mem, combined_rel.view(N, -1, R)], axis=1)
        sum_emb = total_node+total_rel

        new_emb = self.update_embedding(curr_emb, nei_msg)

        diff_score = torch.linalg.norm(sum_emb - new_emb.unsqueeze(1), dim=2)

        topkind = torch.argsort(diff_score, descending=True)[:,:C]

        ind_helper = torch.arange(N, device=topkind.device)*C
        max_ind = ind_helper.unsqueeze(1)+topkind
        max_ind = max_ind.view(-1)
        new_node_mem = total_node.view(-1, R)[max_ind].view(N,C,R)
        new_rel_mem = total_rel.view(-1, R)[max_ind].view(N,C,R)


        return {'h': new_emb,'node_mem':new_node_mem,'rel_mem':new_rel_mem}

    def update_embedding(self, curr_emb, nei_msg):
        new_emb = nei_msg + curr_emb

        return new_emb

class Aggregator(nn.Module):
    def __init__(self, emb_dim):
        super(Aggregator, self).__init__()

    def forward(self, node):
        curr_emb = node.mailbox['curr_emb'][:, 0, :]  # (B, F)
        nei_msg = torch.bmm(node.mailbox['alpha'].transpose(1, 2), node.mailbox['msg']).squeeze(1)  # (B, F)
        # feat = node.mailbox['init_feat'][:, 0, :]
        # print('agg nei_msg:',nei_msg.size())
        # nei_msg, _ = torch.max(node.mailbox['msg'], 1)  # (B, F)

        new_emb = self.update_embedding(curr_emb, nei_msg)

        return {'h': new_emb}

    @abc.abstractmethod
    def update_embedding(curr_emb, nei_msg):
        raise NotImplementedError


class RepAggregator(nn.Module):
    def __init__(self, emb_dim):
        super(RepAggregator, self).__init__()

    def forward(self, node):
        (N,E,D) = node.mailbox['msg'].size()
        curr_emb = node.mailbox['curr_emb'][:, 0, :]  # (B, F)
        # nei_msg = torch.bmm(node.mailbox['alpha'].transpose(1, 2), node.mailbox['msg']).squeeze(1)  # (B, F)
        # feat = node.mailbox['init_feat'][:, 0, :]
        # print('agg nei_msg:',nei_msg.size())
        # nei_msg, _ = torch.max(node.mailbox['msg'], 1)  # (B, F)
        # print(node.mailbox['msg'].size())
        new_emb = curr_emb
        d = min(3, E)
        for i in range(d):
            new_emb = torch.cat([new_emb, node.mailbox['msg'][:,i,:]], dim=1)
        new_emb = torch.cat([new_emb, torch.zeros((N, (3-d)*D), device=new_emb.device)], dim=1)

        # new_emb = self.update_embedding(curr_emb, nei_msg)

        return {'h': new_emb}


class SumAggregator(Aggregator):
    def __init__(self, emb_dim):
        super(SumAggregator, self).__init__(emb_dim)

    def update_embedding(self, curr_emb, nei_msg):
        new_emb = nei_msg + curr_emb

        return new_emb


class MLPAggregator(Aggregator):
    def __init__(self, emb_dim):
        super(MLPAggregator, self).__init__(emb_dim)
        self.linear = nn.Linear(2 * emb_dim, emb_dim)

    def update_embedding(self, curr_emb, nei_msg):
        inp = torch.cat((nei_msg, curr_emb), 1)
        new_emb = F.relu(self.linear(inp))

        return new_emb


class GRUAggregator(Aggregator):
    def __init__(self, emb_dim):
        super(GRUAggregator, self).__init__(emb_dim)
        self.gru = nn.GRUCell(emb_dim, emb_dim)

    def update_embedding(self, curr_emb, nei_msg):
        new_emb = self.gru(nei_msg, curr_emb)

        return new_emb
