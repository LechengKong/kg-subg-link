import logging
from operator import pos
from threading import local
import torch
import numpy as np
import dgl
import time

from torch.utils.data import Dataset
from graph_util import  construct_graph_from_edges,subgraph_extraction_labeling_wiki, get_neighbor_nodes, extract_neighbor_nodes, sample_neg_link


class SubgraphDataset(Dataset):
    def __init__(self, triplets, dataset, params, adj_list, num_rels, num_entities, graph=None, neg_link_per_sample=1):
        self.edges = triplets[dataset]
        self.adj_list = adj_list
        self.coo_adj_list = [adj.tocoo() for adj in self.adj_list]
        self.num_edges = len(self.edges)
        self.num_nodes = num_entities
        self.num_rels = num_rels
        self.params = params
        self.graph = graph
        if self.graph is None:
            self.graph = construct_graph_from_edges(triplets['train'].T, self.num_nodes)
        self.adj_mat = self.graph.adjacency_matrix(transpose=False, scipy_fmt='csr')
        self.adj_mat += self.adj_mat.T

        self.max_n_label = [10, 10]
        self.neg_sample = neg_link_per_sample

        self.sample_size = self.num_edges

    def __len__(self):
        return self.sample_size

    def __getitem__(self, index):
        raise NotImplementedError

    def _get_main_subgraph(self, node_set):
        sample_nodes = extract_neighbor_nodes(node_set, self.adj_mat, h=self.params.hop, max_nodes_per_hop=500)
        sample_nodes = list(node_set) + list(sample_nodes)
        main_subgraph = self.graph.subgraph(sample_nodes)
        main_subgraph.edata['type'] = self.graph.edata['type'][main_subgraph.edata[dgl.EID]]
        p_id = main_subgraph.ndata[dgl.NID].numpy()
        local_adj_mat = main_subgraph.adjacency_matrix(transpose=False).to_dense().numpy()
        # local_adj_mat = main_subgraph.adjacency_matrix(transpose=False, scipy_fmt='csr')
        local_adj_mat += local_adj_mat.T
        node_to_id = {pid: i for i, pid in enumerate(p_id)}

        return main_subgraph, local_adj_mat, node_to_id

    def _get_ind_subgraph(self, nodes, target_rel, main_subgraph):
        ind_subgraph = main_subgraph.subgraph(nodes)
        ind_subgraph.edata['type'] = main_subgraph.edata['type'][ind_subgraph.edata[dgl.EID]]
        ind_subgraph.edata['label'] = torch.tensor(target_rel * np.ones(ind_subgraph.edata['type'].shape),
                                                   dtype=torch.long)
        return ind_subgraph
    
    def _get_labels(self, head, tail, rel, adj):
        nodes, label, enclosing_nodes, disconnected_nodes = subgraph_extraction_labeling_wiki([head, tail], rel, adj, h=self.params.hop, enclosing_sub_graph=self.params.enclosing_sub_graph, max_nodes_per_hop=self.params.max_nodes_per_hop)
        if self.params.node_path_only:
            nodes = np.array(nodes)[enclosing_nodes].tolist()
            label = label[enclosing_nodes]
        return nodes, label

    def _prepare_node_features(self, subgraph, n_labels, rel, n_feats=None):
        near_edges = subgraph.out_edges(0,'all')
        sister_nodes = near_edges[1][subgraph.edata['type'][near_edges[2]] == rel]
        subgraph.ndata['tail_sister'] = torch.tensor(np.zeros((subgraph.num_nodes(),1)), dtype=torch.int32)
        subgraph.ndata['tail_sister'][sister_nodes] = 1

        near_edges = subgraph.in_edges(1,'all')
        sister_nodes = near_edges[0][subgraph.edata['type'][near_edges[2]] == rel]
        subgraph.ndata['head_sister'] = torch.tensor(np.zeros((subgraph.num_nodes(),1)), dtype=torch.int32)
        subgraph.ndata['head_sister'][sister_nodes] = 1
        subgraph.ndata['t_label'] = torch.tensor(rel*np.ones((subgraph.num_nodes(),1)), dtype=torch.int32)

        n_nodes = subgraph.number_of_nodes()
        label_feats = np.zeros((n_nodes, self.max_n_label[0] + 1 + self.max_n_label[1] + 1))
        label_feats[np.arange(n_nodes), n_labels[:, 0]] = 1
        label_feats[np.arange(n_nodes), self.max_n_label[0] + 1 + n_labels[:, 1]] = 1
        n_feats = np.concatenate((label_feats, n_feats), axis=1) if n_feats is not None else label_feats
        subgraph.ndata['feat'] = torch.FloatTensor(n_feats)

        head_id = np.argwhere([label[0] == 0 and label[1] == 1 for label in n_labels])
        tail_id = np.argwhere([label[0] == 1 and label[1] == 0 for label in n_labels])
        n_ids = np.zeros(n_nodes)
        n_ids[head_id] = 1  # head
        n_ids[tail_id] = 2  # tail
        subgraph.ndata['id'] = torch.FloatTensor(n_ids)

        return subgraph



class SubgraphDatasetTrain(SubgraphDataset):
    def __init__(self, triplets, dataset, params, adj_list, num_rels, num_entities, neg_link_per_sample=1):
        super().__init__(triplets, dataset, params, adj_list, num_rels, num_entities, None, neg_link_per_sample)

        pos_g, pos_la, pos_rel, neg_g, neg_la, neg_rel = self.__getitem__(0)
        self.n_feat_dim = pos_g.ndata['feat'].shape[1]

    def __len__(self):
        return self.sample_size

    def __getitem__(self, index):
        st = time.time()
        head, rel, tail = self.edges[index]
        neg_links = sample_neg_link(self.coo_adj_list, rel, head, tail, self.num_nodes, self.neg_sample)
        pos_link = [head, rel, tail]
        nodes = [pos_link[0], pos_link[2]]+[link[0] for link in neg_links] + [link[2] for link in neg_links]
        node_set = set(nodes)
        main_subgraph, local_adj_mat, node_to_id = self._get_main_subgraph(node_set)
        local_adj_mat[node_to_id[pos_link[0]], node_to_id[pos_link[2]]]=0
        local_adj_mat[node_to_id[pos_link[2]], node_to_id[pos_link[0]]]=0  
        pos_nodes, pos_label = self._get_labels(node_to_id[pos_link[0]], node_to_id[pos_link[2]], rel, local_adj_mat)
        local_adj_mat[node_to_id[pos_link[0]], node_to_id[pos_link[2]]]=1
        local_adj_mat[node_to_id[pos_link[2]], node_to_id[pos_link[0]]]=1

        # if len(pos_nodes) == 2:
        #     print("Err")
        # print(pos_nodes)
        # print(len(pos_nodes))
        # if len(pos_nodes)==2 and not main_subgraph.has_edges_between(pos_nodes[1],pos_nodes[0]):
        #     print(rel)
        pos_subgraph = self._get_ind_subgraph(pos_nodes, rel, main_subgraph)
        pos_subgraph = self._prepare_node_features(pos_subgraph, pos_label, rel)
        logging.debug(f'sample one:{time.time()-st}')
        neg_subgraphs = []
        for i in range(self.neg_sample):
            neg_nodes, neg_label = self._get_labels(node_to_id[neg_links[i][0]], node_to_id[neg_links[i][2]], rel, local_adj_mat)
            neg_subgraph = self._get_ind_subgraph(neg_nodes, rel, main_subgraph)
            neg_subgraph = self._prepare_node_features(neg_subgraph, neg_label, rel)
            neg_subgraph.add_edges([0], [1])
            neg_subgraph.edata['type'][-1] = torch.tensor(neg_links[i][1], dtype=torch.int32)
            neg_subgraph.edata['label'][-1] = torch.tensor(neg_links[i][1], dtype=torch.int32)
            neg_subgraphs.append(neg_subgraph)

        logging.debug(f'sampleall:{time.time()-st}')
        return pos_subgraph, 1, pos_link[1], neg_subgraphs, [0] * len(neg_subgraphs), [neg_links[i][1] for i in
                                                                                       range(len(neg_subgraphs))]

class SubgraphDatasetVal(SubgraphDataset):
    def __init__(self, triplets, dataset, params, adj_list, num_rels, num_entities, graph=None, neg_link_per_sample=1):
        super().__init__(triplets, dataset, params, adj_list, num_rels, num_entities, graph, neg_link_per_sample)

        self.__getitem__(0)

    def __len__(self):
        return self.sample_size

    def __getitem__(self, index):
        # st = time.time()
        head, rel, tail = self.edges[index]
        neg_links = sample_neg_link(self.coo_adj_list, rel, head, tail, self.num_nodes, self.neg_sample)
        pos_link = [head, rel, tail]
        nodes = [pos_link[0], pos_link[2]] + [link[0] for link in neg_links] + [link[2] for link in neg_links]
        node_set = set(nodes)
        main_subgraph, local_adj_mat, node_to_id = self._get_main_subgraph(node_set)
        can_edges = [pos_link]+neg_links
        graphs = []
        for i, edge in enumerate(can_edges):
            pos_nodes, pos_label = self._get_labels(node_to_id[edge[0]], node_to_id[edge[2]], rel, local_adj_mat)
            # if i != 0:
            #     print(len(pos_nodes))
            # if i==0 and len(pos_nodes)==2 and not main_subgraph.has_edges_between(pos_nodes[0],pos_nodes[1]) and not main_subgraph.has_edges_between(pos_nodes[1],pos_nodes[0]):
            #     print(rel)
            pos_subgraph = self._get_ind_subgraph(pos_nodes, rel, main_subgraph)
            pos_subgraph = self._prepare_node_features(pos_subgraph, pos_label, rel)
            pos_subgraph.add_edges([0],[1])
            pos_subgraph.edata['type'][-1] = torch.tensor(rel, dtype=torch.int32)
            pos_subgraph.edata['label'][-1] = torch.tensor(rel, dtype=torch.int32)
            graphs.append(pos_subgraph)
        return graphs, [rel]*len(graphs), 0


class SubgraphDatasetWikiOnlineValSubset(Dataset):
    """Extracted, labeled, subgraph dataset -- DGL Only"""

    def __init__(self, data, params, can_path, hr_path, t_ind_path, neg_link_per_sample=1, use_feature=False):
        self.wiki_data = data
        self.edges = data.train_hrt
        self.num_rels = data.num_relations
        self.hr = np.load(hr_path)
        self.cans = np.load(can_path)
        self.t_ind = np.load(t_ind_path)
        self.num_edges = self.hr.shape[0]
        self.num_nodes = data.num_entities
        self.use_feature = use_feature
        self.params = params
        self.graph = construct_graph_from_edges(self.edges.T, self.num_nodes)
        self.adj_mat = self.graph.adjacency_matrix(transpose=False, scipy_fmt='csr')
        self.adj_mat += self.adj_mat.T

        self.max_n_label = [10, 10]
        self.neg_sample = neg_link_per_sample

        self.sample_size = self.num_edges

        # the effective number of relations after adding symmetric adjacency matrices and/or self connections
        self.aug_num_rels = data.num_relations
        pos_g, pos_la, pos_rel, neg_g, neg_la, neg_rel = self.__getitem__(0)
        self.n_feat_dim = pos_g.ndata['feat'].shape[1]

    def __len__(self):
        return self.sample_size

    def __getitem__(self, index):
        # st = time.time()
        head, rel = self.hr[index]
        can = self.cans[index]
        true_ind = self.t_ind[index]
        true_tail = can[true_ind]
        neg_tails = can.tolist()
        neg_tails.pop(true_ind)
        neg_tails = neg_tails[-self.neg_sample:]
        
        pos_link = [head, rel, true_tail]
        neg_links = [[head, rel, neg_tail] for neg_tail in neg_tails]
        
        nodes = [link[0] for link in neg_links] + [link[2] for link in neg_links] + [pos_link[0], pos_link[2]]
        node_set = set(nodes)
        sample_nodes = extract_neighbor_nodes(node_set, self.adj_mat, h=self.params.hop, max_nodes_per_hop=10000)
        sample_nodes = list(node_set) + list(sample_nodes)
        main_subgraph = self.graph.subgraph(sample_nodes)
        main_subgraph.edata['type'] = self.graph.edata['type'][main_subgraph.edata[dgl.EID]]
        p_id = main_subgraph.ndata[dgl.NID].numpy()
        local_adj_mat = main_subgraph.adjacency_matrix(transpose=False, scipy_fmt='csr')
        local_adj_mat += local_adj_mat.T
        node_to_id = {pid: i for i, pid in enumerate(p_id)}
        # print("adj:", time.time()-st)

        pos_nodes, pos_label, disconnected_nodes = subgraph_extraction_labeling_wiki([node_to_id[pos_link[0]], node_to_id[pos_link[2]]], pos_link[1], local_adj_mat, h=self.params.hop, max_nodes_per_hop=self.params.max_nodes_per_hop)
        # if len(pos_nodes) == 2:
        #     print(index)
        pos_subgraph = main_subgraph.subgraph(pos_nodes)
        pos_subgraph.edata['type'] = main_subgraph.edata['type'][pos_subgraph.edata[dgl.EID]]
        pos_subgraph.edata['label'] = torch.tensor(pos_link[1] * np.ones(pos_subgraph.edata['type'].shape),
                                                   dtype=torch.long)
        pos_subgraph.add_edges([0], [1])
        pos_subgraph.edata['type'][-1] = torch.tensor(pos_link[1], dtype=torch.int32)
        pos_subgraph.edata['label'][-1] = torch.tensor(pos_link[1], dtype=torch.int32)
        # map the id read by GraIL to the entity IDs as registered by the KGE embeddings
        pos_subgraph = self._prepare_features_new(pos_subgraph, pos_label, None)
        neg_subgraphs = []
        # print("sample 1:", time.time()-st)
        for i in range(self.neg_sample):
            neg_nodes, neg_label, disconnected_nodes = subgraph_extraction_labeling_wiki([node_to_id[neg_links[i][0]], node_to_id[neg_links[i][2]]], neg_links[i][1], local_adj_mat, h=self.params.hop, max_nodes_per_hop=self.params.max_nodes_per_hop)

            neg_subgraph = main_subgraph.subgraph(neg_nodes)
            neg_subgraph.edata['type'] = main_subgraph.edata['type'][neg_subgraph.edata[dgl.EID]]
            neg_subgraph.edata['label'] = torch.tensor(neg_links[i][1] * np.ones(neg_subgraph.edata['type'].shape),
                                                       dtype=torch.long)
            neg_subgraph.add_edges([0], [1])
            neg_subgraph.edata['type'][-1] = torch.tensor(neg_links[i][1], dtype=torch.int32)
            neg_subgraph.edata['label'][-1] = torch.tensor(neg_links[i][1], dtype=torch.int32)
            neg_subgraphs.append(self._prepare_features_new(neg_subgraph, neg_label, None))

        # print("sample all:", time.time()-st)
        return pos_subgraph, 1, pos_link[1], neg_subgraphs, [0] * len(neg_subgraphs), [neg_links[i][1] for i in
                                                                                       range(len(neg_subgraphs))]

    def _prepare_features_new(self, subgraph, n_labels, n_feats=None):
        # One hot encode the node label feature and concat to n_featsure
        n_nodes = subgraph.number_of_nodes()
        label_feats = np.zeros((n_nodes, self.max_n_label[0] + 1 + self.max_n_label[1] + 1))
        label_feats[np.arange(n_nodes), n_labels[:, 0]] = 1
        label_feats[np.arange(n_nodes), self.max_n_label[0] + 1 + n_labels[:, 1]] = 1
        # label_feats = np.zeros((n_nodes, self.max_n_label[0] + 1 + self.max_n_label[1] + 1))
        # label_feats[np.arange(n_nodes), 0] = 1
        # label_feats[np.arange(n_nodes), self.max_n_label[0] + 1] = 1
        n_feats = np.concatenate((label_feats, n_feats), axis=1) if n_feats is not None else label_feats
        subgraph.ndata['feat'] = torch.FloatTensor(n_feats)

        head_id = np.argwhere([label[0] == 0 and label[1] == 1 for label in n_labels])
        tail_id = np.argwhere([label[0] == 1 and label[1] == 0 for label in n_labels])
        n_ids = np.zeros(n_nodes)
        n_ids[head_id] = 1  # head
        n_ids[tail_id] = 2  # tail
        subgraph.ndata['id'] = torch.FloatTensor(n_ids)

        return subgraph


class SubgraphDatasetWikiLocalSubsetEval(Dataset):
    """Extracted, labeled, subgraph dataset -- DGL Only"""

    def __init__(self, data, params, g, adj_mat, can_path=None, hr_path=None, t_ind_path=None, sample_size=1000, neg_link_per_sample=1, use_feature=False):
        self.wiki_data = data
        self.num_rels = data.num_relations
        self.val_dict = data.valid_dict
        if can_path is None:
            self.val_dict = data.valid_dict
        else:
            self.val_dict['h,r->t']['hr'] = np.load(hr_path)
            self.val_dict['h,r->t']['t_candidate'] = np.load(can_path)
            self.val_dict['h,r->t']['t_correct_index'] = np.load(t_ind_path)
        self.edges = data.train_hrt
        self.num_edges = len(self.val_dict['h,r->t']['hr'])
        self.num_nodes = data.num_entities
        self.use_feature = use_feature
        self.params = params
        self.graph = g
        self.adj_mat = adj_mat
        self.max_n_label = [10, 10]
        self.neg_sample = neg_link_per_sample

        self.sample_size = self.num_edges

        # the effective number of relations after adding symmetric adjacency matrices and/or self connections
        self.aug_num_rels = data.num_relations
        self.__getitem__(0)

    def __len__(self):
        return self.sample_size

    def __getitem__(self, index):
        # st = time.time()
        p_candidates = self.val_dict['h,r->t']['t_candidate'][index].tolist()
        p_head = self.val_dict['h,r->t']['hr'][index, 0]
        rel = self.val_dict['h,r->t']['hr'][index, 1]
        base_nodes = set(p_candidates + [p_head])
        # print("get can", time.time()-st)
        sample_nodes = extract_neighbor_nodes(base_nodes, self.adj_mat, h=self.params.hop, max_nodes_per_hop=100000)
        # print("get nei", time.time()-st)
        sample_nodes = list(base_nodes) + list(sample_nodes)
        g = self.graph.subgraph(sample_nodes)
        # print("get subg", time.time()-st)
        g.edata['type'] = self.graph.edata['type'][g.edata[dgl.EID]]
        true_ind = self.val_dict['h,r->t']['t_correct_index'][index]
        p_id = g.ndata[dgl.NID].numpy()
        adj_mat = g.adjacency_matrix(transpose=False, scipy_fmt='csr')
        adj_mat += adj_mat.T
        # print('adjmat',time.time()-st)
        node_to_id = {pid: i for i, pid in enumerate(p_id)}
        candidates = [node_to_id[i] for i in p_candidates]
        head = node_to_id[p_head]
        # print('mis',time.time()-st)
        graphs = []
        for i, candidate in enumerate(candidates):
            pos_nodes, pos_label, disconnected_nodes = subgraph_extraction_labeling_wiki([head, candidate], rel,
                                                                              adj_mat, h=self.params.hop,max_nodes_per_hop=self.params.max_nodes_per_hop)
            pos_subgraph = g.subgraph(pos_nodes)
            pos_subgraph.edata['type'] = g.edata['type'][pos_subgraph.edata[dgl.EID]]
            pos_subgraph.edata['label'] = torch.tensor(rel * np.ones(pos_subgraph.edata['type'].shape),
                                                       dtype=torch.long)
            pos_subgraph.add_edges([0], [1])
            pos_subgraph.edata['type'][-1] = torch.tensor(rel, dtype=torch.int32)
            pos_subgraph.edata['label'][-1] = torch.tensor(rel, dtype=torch.int32)
            pos_subgraph = self._prepare_features_new(pos_subgraph, pos_label, None)
            graphs.append(pos_subgraph)
        return graphs, [rel]*len(graphs), true_ind

    def _prepare_features_new(self, subgraph, n_labels, n_feats=None):
        # One hot encode the node label feature and concat to n_featsure
        n_nodes = subgraph.number_of_nodes()
        label_feats = np.zeros((n_nodes, self.max_n_label[0] + 1 + self.max_n_label[1] + 1))
        label_feats[np.arange(n_nodes), n_labels[:, 0]] = 1
        label_feats[np.arange(n_nodes), self.max_n_label[0] + 1 + n_labels[:, 1]] = 1
        # label_feats = np.zeros((n_nodes, self.max_n_label[0] + 1 + self.max_n_label[1] + 1))
        # label_feats[np.arange(n_nodes), 0] = 1
        # label_feats[np.arange(n_nodes), self.max_n_label[0] + 1] = 1
        n_feats = np.concatenate((label_feats, n_feats), axis=1) if n_feats is not None else label_feats
        subgraph.ndata['feat'] = torch.FloatTensor(n_feats)

        head_id = np.argwhere([label[0] == 0 and label[1] == 1 for label in n_labels])
        tail_id = np.argwhere([label[0] == 1 and label[1] == 0 for label in n_labels])
        n_ids = np.zeros(n_nodes)
        n_ids[head_id] = 1  # head
        n_ids[tail_id] = 2  # tail
        subgraph.ndata['id'] = torch.FloatTensor(n_ids)

        self.n_feat_dim = n_feats.shape[1]  # Find cleaner way to do this -- i.e. set the n_feat_dim
        return subgraph


class SubgraphDatasetWikiLocalSubsetTest(Dataset):
    """Extracted, labeled, subgraph dataset -- DGL Only"""

    def __init__(self, data, params, can=None, hr=None, neg_link_per_sample=1, use_feature=False):
        self.wiki_data = data
        self.edges = data.train_hrt
        self.num_rels = data.num_relations
        self.val_dict = {}
        self.val_dict['h,r->t'] = {}
        self.val_dict['h,r->t']['hr'] = hr
        self.val_dict['h,r->t']['t_candidate'] = can
        self.num_edges = len(self.val_dict['h,r->t']['hr'])
        self.num_nodes = data.num_entities
        self.use_feature = use_feature
        self.params = params
        self.graph = construct_graph_from_edges(self.edges.T, self.num_nodes)
        self.adj_mat = self.graph.adjacency_matrix(transpose=False, scipy_fmt='csr')
        self.adj_mat += self.adj_mat.T
        self.max_n_label = [10, 10]
        self.neg_sample = neg_link_per_sample

        self.sample_size = self.num_edges

        # the effective number of relations after adding symmetric adjacency matrices and/or self connections
        self.aug_num_rels = data.num_relations
        g, r, t = self.__getitem__(0)
        self.n_feat_dim = g[0].ndata['feat'].shape[1]

    def __len__(self):
        return self.sample_size

    def __getitem__(self, index):
        # st = time.time()
        p_candidates = self.val_dict['h,r->t']['t_candidate'][index].tolist()
        p_head = self.val_dict['h,r->t']['hr'][index, 0]
        rel = self.val_dict['h,r->t']['hr'][index, 1]
        base_nodes = set(p_candidates + [p_head])
        # print("get can", time.time()-st)
        sample_nodes = get_neighbor_nodes(base_nodes, self.adj_mat, h=self.params.hop, max_nodes_per_hop=2000)
        # print("get nei", time.time()-st)
        sample_nodes = list(base_nodes) + list(sample_nodes)
        g = self.graph.subgraph(sample_nodes)
        # print("get subg", time.time()-st)
        g.edata['type'] = self.graph.edata['type'][g.edata[dgl.EID]]
        true_ind = 0
        p_id = g.ndata[dgl.NID].numpy()
        adj_mat = g.adjacency_matrix(transpose=False, scipy_fmt='csr')
        adj_mat += adj_mat.T
        # print('adjmat',time.time()-st)
        node_to_id = {pid: i for i, pid in enumerate(p_id)}
        candidates = [node_to_id[i] for i in p_candidates]
        head = node_to_id[p_head]
        # print('mis',time.time()-st)
        graphs = []
        for i, candidate in enumerate(candidates):
            pos_nodes, pos_label, disconnected_nodes = subgraph_extraction_labeling_wiki([head, candidate], rel,
                                                                              adj_mat, h=self.params.hop, max_nodes_per_hop=30)
            pos_subgraph = g.subgraph(pos_nodes)
            pos_subgraph.edata['type'] = g.edata['type'][pos_subgraph.edata[dgl.EID]]
            pos_subgraph.edata['label'] = torch.tensor(rel * np.ones(pos_subgraph.edata['type'].shape),
                                                       dtype=torch.long)
            pos_subgraph.add_edges([0], [1])
            pos_subgraph.edata['type'][-1] = torch.tensor(rel, dtype=torch.int32)
            pos_subgraph.edata['label'][-1] = torch.tensor(rel, dtype=torch.int32)
            pos_subgraph = self._prepare_features_new(pos_subgraph, pos_label, None)
            graphs.append(pos_subgraph)
        # print('sampleall',time.time()-st)
        return graphs, [rel]*len(graphs), true_ind

    def _prepare_features_new(self, subgraph, n_labels, n_feats=None):
        # One hot encode the node label feature and concat to n_featsure
        n_nodes = subgraph.number_of_nodes()
        label_feats = np.zeros((n_nodes, self.max_n_label[0] + 1 + self.max_n_label[1] + 1))
        label_feats[np.arange(n_nodes), n_labels[:, 0]] = 1
        label_feats[np.arange(n_nodes), self.max_n_label[0] + 1 + n_labels[:, 1]] = 1
        # label_feats = np.zeros((n_nodes, self.max_n_label[0] + 1 + self.max_n_label[1] + 1))
        # label_feats[np.arange(n_nodes), 0] = 1
        # label_feats[np.arange(n_nodes), self.max_n_label[0] + 1] = 1
        n_feats = np.concatenate((label_feats, n_feats), axis=1) if n_feats is not None else label_feats
        subgraph.ndata['feat'] = torch.FloatTensor(n_feats)

        head_id = np.argwhere([label[0] == 0 and label[1] == 1 for label in n_labels])
        tail_id = np.argwhere([label[0] == 1 and label[1] == 0 for label in n_labels])
        n_ids = np.zeros(n_nodes)
        n_ids[head_id] = 1  # head
        n_ids[tail_id] = 2  # tail
        subgraph.ndata['id'] = torch.FloatTensor(n_ids)

        self.n_feat_dim = n_feats.shape[1]  # Find cleaner way to do this -- i.e. set the n_feat_dim
        return subgraph

