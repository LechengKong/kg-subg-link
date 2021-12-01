import logging
from operator import pos
from threading import local
import torch
import numpy as np
import dgl
import time
import sys, os
import scipy.sparse as ssp
from scipy.sparse import csr_matrix

from torch.utils.data import Dataset
from graph_util import  construct_graph_from_edges,subgraph_extraction_labeling_wiki, get_neighbor_nodes, extract_neighbor_nodes, sample_neg_link, construct_reverse_graph_from_edges
from scipy.linalg import eig, eigh
from util import SmartTimer

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


class SubgraphDataset(Dataset):
    def __init__(self, triplets, dataset, params, adj_list, num_rels, num_entities, graph=None, neg_link_per_sample=1):
        self.edges = triplets[dataset]
        self.adj_list = adj_list
        self.coo_adj_list = [adj.tocoo() for adj in self.adj_list]
        self.num_edges = len(self.edges)
        self.graph_edge_size = len(triplets['train'])
        self.num_nodes = num_entities
        self.num_rels = num_rels
        self.params = params
        self.graph = graph
        if self.graph is None:
            self.graph = construct_reverse_graph_from_edges(triplets['train'].T, self.num_nodes, self.num_rels)
        self.adj_mat = self.graph.adjacency_matrix(transpose=False, scipy_fmt='csr')
        # self.adj_mat += self.adj_mat.T

        self.max_n_label = [10, 10]
        self.neg_sample = neg_link_per_sample

        self.sample_size = self.num_edges
        if self.params.use_neighbor_feature:
            self._get_neighbor_edge_ratio(self.graph, 'ratio')

    def __len__(self):
        return self.sample_size

    def __getitem__(self, index):
        raise NotImplementedError

    def _get_main_subgraph(self, node_set, max_nodes):
        sample_nodes = extract_neighbor_nodes(node_set, self.adj_mat, h=self.params.hop, max_nodes_per_hop=max_nodes)
        sample_nodes = list(node_set) + list(sample_nodes)
        main_subgraph = self.graph.subgraph(sample_nodes)
        main_subgraph.edata['type'] = self.graph.edata['type'][main_subgraph.edata[dgl.EID]]
        p_id = main_subgraph.ndata[dgl.NID].numpy()
        local_adj_mat = main_subgraph.adjacency_matrix(transpose=False).to_dense().numpy()
        # local_adj_mat = main_subgraph.adjacency_matrix(transpose=False, scipy_fmt='csr')
        # local_adj_mat += local_adj_mat.T
        node_to_id = {pid: i for i, pid in enumerate(p_id)}

        return main_subgraph, local_adj_mat, node_to_id

    def _get_ind_subgraph(self, nodes, target_rel, main_subgraph):
        ind_subgraph = main_subgraph.subgraph(nodes)
        ind_subgraph.edata['type'] = main_subgraph.edata['type'][ind_subgraph.edata[dgl.EID]]
        ind_subgraph.edata['label'] = torch.tensor(target_rel * np.ones(ind_subgraph.edata['type'].shape),
                                                   dtype=torch.long)
        return ind_subgraph
    
    def _get_labels(self, head, tail, rel, adj):
        nodes_int, label_int, enclosing_nodes, _, root_dist = subgraph_extraction_labeling_wiki([head, tail], rel, adj, h=self.params.hop, enclosing_sub_graph=False, max_nodes_per_hop=self.params.max_nodes_per_hop)
        nodes_un, label_un, _, disconnected_nodes, _ = subgraph_extraction_labeling_wiki([head, tail], rel, adj, h=self.params.hop, enclosing_sub_graph=False, max_nodes_per_hop=self.params.max_nodes_per_hop)
        if self.params.node_path_only:
            nodes = np.array(nodes_int)[enclosing_nodes].tolist()
            label = label_int[enclosing_nodes]
        if self.params.same_size_neighbor:
            nodes_int = np.array(nodes_int)[enclosing_nodes].tolist()
            nodes_un = np.array(nodes_un)[disconnected_nodes].tolist()
            label_int = label_int[enclosing_nodes]
            label_un = label_un[disconnected_nodes]
            # print(disconnected_nodes)
            nodes = nodes_int+nodes_un
            label = np.concatenate((label_int,label_un))
        return nodes, label, len(enclosing_nodes), root_dist

    def _get_neighbor_edge_ratio(self, subgraph, output_name):
        # subgraph.ndata['edge_ratio'] = torch.tensor(np.zeros(subgraph.num_nodes(), self.num_rels))
        subgraph.edata['type_onehot'] = torch.nn.functional.one_hot(subgraph.edata['type'].to(torch.int64), num_classes=self.num_rels*2).to(torch.float)
        subgraph.update_all(dgl.function.copy_e('type_onehot','er'), dgl.function.sum('er',output_name))

    def _prepare_node_features(self, subgraph, n_labels, rel, n_feats=None, root_dist=None):
        # if self.params.use_neighbor_feature:
        #     n_feats = subgraph.ndata['ratio']
        near_edges = subgraph.out_edges(0,'all')
        # sister_nodes = near_edges[1][subgraph.edata['type'][near_edges[2]] == rel]
        sister_nodes = near_edges[1]
        subgraph.ndata['tail_sister'] = torch.tensor(np.zeros((subgraph.num_nodes(),1)), dtype=torch.int32)
        subgraph.ndata['tail_sister'][sister_nodes] = 1
        subgraph.ndata['tail_sister'][1] = 0
        subgraph.ndata['tail_sister_type'] = torch.tensor(np.zeros((subgraph.num_nodes(),1)), dtype=torch.int32)
        subgraph.ndata['tail_sister_type'][sister_nodes] = subgraph.edata['type'][near_edges[2]].unsqueeze(1)
        subgraph.ndata['tail_sister_type'][1] = rel

        near_edges = subgraph.in_edges(1,'all')
        # sister_nodes = near_edges[0][subgraph.edata['type'][near_edges[2]] == rel]
        sister_nodes = near_edges[0]
        subgraph.ndata['head_sister'] = torch.tensor(np.zeros((subgraph.num_nodes(),1)), dtype=torch.int32)
        subgraph.ndata['head_sister'][sister_nodes] = 1
        subgraph.ndata['head_sister'][0] = 0
        subgraph.ndata['head_sister_type'] = torch.tensor(np.zeros((subgraph.num_nodes(),1)), dtype=torch.int32)
        subgraph.ndata['head_sister_type'][sister_nodes] = subgraph.edata['type'][near_edges[2]].unsqueeze(1)
        subgraph.ndata['head_sister_type'][0] = rel
        subgraph.ndata['t_label'] = torch.tensor(rel*np.ones((subgraph.num_nodes(),1)), dtype=torch.int32)

        n_nodes = subgraph.number_of_nodes()
        label_feats = np.zeros((n_nodes, self.max_n_label[0] + 1 + self.max_n_label[1] + 1))
        label_feats[np.arange(n_nodes), n_labels[:, 0]] = 1
        label_feats[np.arange(n_nodes), self.max_n_label[0] + 1 + n_labels[:, 1]] = 1
        n_feats = np.concatenate((label_feats, n_feats), axis=1) if n_feats is not None else label_feats
        subgraph.ndata['feat'] = torch.FloatTensor(n_feats)
        # subgraph.ndata['feat'] = torch.ones([n_nodes, 1], dtype=torch.float32)
        subgraph.ndata['h'] = torch.ones([n_nodes, self.params.emb_dim], dtype=torch.float32)

        head_id = np.argwhere([label[0] == 0 and label[1] == 1 for label in n_labels])
        tail_id = np.argwhere([label[0] == 1 and label[1] == 0 for label in n_labels])
        if root_dist is not None:
            subgraph.ndata['rt_dist'] = torch.zeros([n_nodes, 1], dtype=torch.float32)
            subgraph.ndata['rt_dist'][head_id] = root_dist
        n_ids = np.zeros(n_nodes)
        n_ids[head_id] = 1  # head
        n_ids[tail_id] = 2  # tail
        subgraph.ndata['id'] = torch.FloatTensor(n_ids)

        return subgraph

    def _get_spectrum(self, graph):
        if self.params.eig_size ==0 :
            return []
        adj_mat = graph.adjacency_matrix(transpose=False).to_dense().numpy()
        # local_adj_mat = main_subgraph.adjacency_matrix(transpose=False, scipy_fmt='csr')
        adj_mat += adj_mat.T
        # adj_mat = np.logical_or(adj_mat, adj_mat.T)
        eig_val, _ = eig(adj_mat)
        eig_val = np.real(eig_val)
        ind = int(min(self.params.eig_size, len(eig_val)))
        large_eig = np.argsort(np.abs(eig_val))[-ind::]
        # print(eig_val)
        eig_vec = np.zeros(self.params.eig_size)
        # eig_vec[:ind] = np.real(eig_val[:ind])
        eig_vec[:ind] = eig_val[large_eig[::-1]]
        return eig_vec
    
    def _get_spectrum_graph(self, nodes, enc_nodes, target_rel, subgraph, labels, add_ht=True):
        full_subg = self._get_ind_subgraph(nodes, target_rel, subgraph)
        if add_ht:
            full_subg.add_edges([0],[1])
            full_subg.edata['type'][-1] = torch.tensor(target_rel, dtype=torch.int32)
            full_subg.edata['label'][-1] = torch.tensor(target_rel, dtype=torch.int32)
        spe = self._get_spectrum(full_subg)
        sm_subg = self._get_ind_subgraph(nodes[:enc_nodes], target_rel, subgraph)

        return sm_subg, spe, labels[:enc_nodes]



class SubgraphDatasetTrain(SubgraphDataset):
    def __init__(self, triplets, dataset, params, adj_list, num_rels, num_entities, neg_link_per_sample=1):
        super().__init__(triplets, dataset, params, adj_list, num_rels, num_entities, None, neg_link_per_sample)

        pos_g, pos_la, pos_rel, neg_g, neg_la, neg_rel = self.__getitem__(113)
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
        main_subgraph, local_adj_mat, node_to_id = self._get_main_subgraph(node_set, self.params.train_max_n)
        # if self.params.use_neighbor_feature:
        #     self._get_neighbor_edge_ratio(main_subgraph, 'ratio')
        # local_adj_mat[node_to_id[pos_link[0]], node_to_id[pos_link[2]]]=0
        # local_adj_mat[node_to_id[pos_link[2]], node_to_id[pos_link[0]]]=0  
        pos_nodes, pos_label, enc_nodes, root_dist = self._get_labels(node_to_id[pos_link[0]], node_to_id[pos_link[2]], rel, local_adj_mat)
        # local_adj_mat[node_to_id[pos_link[0]], node_to_id[pos_link[2]]]=1
        # local_adj_mat[node_to_id[pos_link[2]], node_to_id[pos_link[0]]]=1

        # if len(pos_nodes) == 2:
        #     print("Err")
        # print(pos_nodes)
        # print(len(pos_nodes))
        # if len(pos_nodes)==2 and not main_subgraph.has_edges_between(pos_nodes[1],pos_nodes[0]):
        #     print(rel)
        pos_subgraph = self._get_ind_subgraph(pos_nodes, rel, main_subgraph)
        pos_subgraph = self._prepare_node_features(pos_subgraph, pos_label, rel, root_dist=root_dist)
        # pos_subgraph.add_edges([0],[1])
        # pos_subgraph.edata['type'][-1] = torch.tensor(rel, dtype=torch.int32)
        # pos_subgraph.edata['label'][-1] = torch.tensor(rel, dtype=torch.int32)
        # pos_subgraph.add_edges([1],[0])
        # pos_subgraph.edata['type'][-1] = torch.tensor(rel+self.num_rels, dtype=torch.int32)
        # pos_subgraph.edata['label'][-1] = torch.tensor(rel+self.num_rels, dtype=torch.int32)
        # print(pos_subgraph.ndata['ratio'])
        # print(pos_subgraph.num_nodes())
        # blockPrint()
        logging.debug(f'sample one:{time.time()-st}')
        neg_subgraphs = []
        for i in range(self.neg_sample):
            neg_nodes, neg_label, enc_nodes, root_dist = self._get_labels(node_to_id[neg_links[i][0]], node_to_id[neg_links[i][2]], rel, local_adj_mat)
            neg_subgraph = self._get_ind_subgraph(neg_nodes, rel, main_subgraph)
            neg_subgraph = self._prepare_node_features(neg_subgraph, neg_label, rel, root_dist=root_dist)
            # neg_subgraph.add_edges([0], [1])
            # neg_subgraph.edata['type'][-1] = torch.tensor(neg_links[i][1], dtype=torch.int32)
            # neg_subgraph.edata['label'][-1] = torch.tensor(neg_links[i][1], dtype=torch.int32)
            # neg_subgraph.add_edges([1], [0])
            # neg_subgraph.edata['type'][-1] = torch.tensor(neg_links[i][1]+self.num_rels, dtype=torch.int32)
            # neg_subgraph.edata['label'][-1] = torch.tensor(neg_links[i][1]+self.num_rels, dtype=torch.int32)
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
        main_subgraph, local_adj_mat, node_to_id = self._get_main_subgraph(node_set, self.params.test_max_n)
        # if self.params.use_neighbor_feature:
        #     self._get_neighbor_edge_ratio(main_subgraph, 'ratio')
        can_edges = [pos_link]+neg_links
        graphs = []
        for i, edge in enumerate(can_edges):
            pos_nodes, pos_label, enc_nodes, root_dist = self._get_labels(node_to_id[edge[0]], node_to_id[edge[2]], rel, local_adj_mat)
            # if i != 0:
            #     print(len(pos_nodes))
            # if i==0 and len(pos_nodes)==2 and not main_subgraph.has_edges_between(pos_nodes[0],pos_nodes[1]) and not main_subgraph.has_edges_between(pos_nodes[1],pos_nodes[0]):
            #     print(index)
            pos_subgraph = self._get_ind_subgraph(pos_nodes, rel, main_subgraph)
            pos_subgraph = self._prepare_node_features(pos_subgraph, pos_label, rel, root_dist=root_dist)
            # pos_subgraph.add_edges([0],[1])
            # pos_subgraph.edata['type'][-1] = torch.tensor(rel, dtype=torch.int32)
            # pos_subgraph.edata['label'][-1] = torch.tensor(rel, dtype=torch.int32)
            # pos_subgraph.add_edges([1],[0])
            # pos_subgraph.edata['type'][-1] = torch.tensor(rel+self.num_rels, dtype=torch.int32)
            # pos_subgraph.edata['label'][-1] = torch.tensor(rel+self.num_rels, dtype=torch.int32)
            # if index==43 and i==0:
            #     print(pos_subgraph.edges())
            #     print(pos_subgraph.edata['type'])
            graphs.append(pos_subgraph)
        return graphs, [rel]*len(graphs), 0


class SubgraphDatasetWhole(SubgraphDataset):
    def __init__(self, triplets, dataset, params, adj_list, num_rels, num_entities, neg_link_per_sample=1):
        super().__init__(triplets, dataset, params, adj_list, num_rels, num_entities, None, neg_link_per_sample)
        self.graph.ndata['feat'] = torch.ones([self.num_nodes, 1], dtype=torch.float32)
        self.__getitem__(113)
        self.n_feat_dim = self.graph.ndata['feat'].shape[1]

    def __len__(self):
        return self.sample_size

    def __getitem__(self, index):
        head, rel, tail = self.edges[index]
        neg_links = sample_neg_link(self.coo_adj_list, rel, head, tail, self.num_nodes, self.neg_sample)
        pos_link = [head, rel, tail]
        links = [pos_link]+neg_links
        link_arr = np.array(links)
        d_mat = np.clip(ssp.csgraph.shortest_path(self.adj_mat, indices=[link_arr[:,0]], directed=False, unweighted=True), 0, self.params.num_gcn_layers+1)
        dist = d_mat[0, np.arange(len(link_arr)), link_arr[:, 2]]
        return self.graph, links, dist.tolist()

class SubgraphDatasetOnlyLink(SubgraphDataset):
    def __init__(self, triplets, dataset, params, adj_list, num_rels, num_entities, neg_link_per_sample=1, sample_method=sample_neg_link):
        super().__init__(triplets, dataset, params, adj_list, num_rels, num_entities, None, neg_link_per_sample)
        self.init_dim = 10
        self.graph.ndata['feat'] = torch.ones([self.num_nodes, self.init_dim], dtype=torch.float32)
        self.graph.ndata['label'] = torch.ones([self.num_nodes, 1], dtype=torch.int64)
        self.sample_links = sample_method
        # self.graph.ndata['node_mem'] = torch.ones([self.num_nodes, 5, self.params.emb_dim], dtype=torch.float32)
        # self.graph.ndata['rel_mem'] = torch.zeros([self.num_nodes, 5, self.params.emb_dim], dtype=torch.float32)
        # self.re_label()
        if params.use_random_labels:
            self.re_label()
        self.__getitem__(142)
        self.n_feat_dim = self.graph.ndata['feat'].shape[1]

    def __len__(self):
        return self.sample_size

    def __getitem__(self, index):
        head, rel, tail = self.edges[index]
        neg_links = self.sample_links(self.coo_adj_list, rel, head, tail, self.num_nodes, self.neg_sample)
        pos_link = [head, rel, tail]
        links = [pos_link]+neg_links
        link_arr = np.array(links)
        dis_mat_head, pred_mat_head = ssp.csgraph.shortest_path(self.adj_mat, indices=head, directed=False, unweighted=True, return_predecessors=True)
        dis_mat_tail, pred_mat_tail = ssp.csgraph.shortest_path(self.adj_mat, indices=tail, directed=False, unweighted=True, return_predecessors=True)
        d_head = np.clip(dis_mat_head, 1, self.params.shortest_path_dist)
        d_tail = np.clip(dis_mat_tail, 1, self.params.shortest_path_dist)
        dist = np.zeros(len(links), dtype=int)
        head_ind = link_arr[:,0]==head
        tail_ind = link_arr[:,2]==tail
        dist[head_ind] = d_head[link_arr[head_ind, 2]]
        dist[tail_ind] = d_tail[link_arr[tail_ind, 0]]
        inter_count_head = np.zeros((np.sum(head_ind), self.params.shortest_path_dist+1), dtype=int)-1
        # pred_mat[0, np.arange(len(link_arr)), link_arr[:, 0]] = link_arr[:, 0]
        d = len(pred_mat_head)
        pred_mat_head = np.concatenate([pred_mat_head, np.array([d])])
        pred_mat_head[pred_mat_head==-9999] = d
        p = pred_mat_head[link_arr[head_ind, 2]]
        inter_count_head[:,0] = link_arr[head_ind,2]
        for i in range(1,self.params.shortest_path_dist+1):
            inter_count_head[:,i] = p
            p = pred_mat_head[p]
        def swap_fun(a):
            c = np.sum(a!=d)
            if c==0:
                return a
            a[:c]=a[c-1::-1]
            return a
        # np.put_along_axis(inter_count_head, dist[head_ind].reshape((-1,1))-1, d, axis=1)
        if self.params.path_add_head:
            inter_count_head[np.arange(len(inter_count_head)), (dist[head_ind]+1)%(self.params.shortest_path_dist+1)] = link_arr[head_ind,0]
        inter_count_head = np.apply_along_axis(swap_fun, 1, inter_count_head)
        # print('swap', time.time()-v)
        d = len(pred_mat_tail)
        inter_count_tail = np.zeros((np.sum(tail_ind), self.params.shortest_path_dist+1), dtype=int)-1
        pred_mat_tail = np.concatenate([pred_mat_tail, np.array([d])])
        pred_mat_tail[pred_mat_tail==-9999] = d
        p = pred_mat_tail[link_arr[tail_ind, 0]]
        inter_count_tail[:,0] = link_arr[tail_ind,0]
        for i in range(1,self.params.shortest_path_dist+1):
            inter_count_tail[:,i] = p
            p = pred_mat_tail[p]
        # np.put_along_axis(inter_count_tail, dist[tail_ind].reshape((-1,1))-1, d, axis=1)
        # inter_count[np.arange(len(inter_count)), dist.astype(int)]=-1
        if self.params.path_add_head:
            inter_count_tail[np.arange(len(inter_count_tail)),  (dist[tail_ind]+1)%(self.params.shortest_path_dist+1)] = link_arr[tail_ind,0]
        inter_count = np.zeros((len(links), self.params.shortest_path_dist+1), dtype=int)-1
        inter_count[head_ind] = inter_count_head
        inter_count[tail_ind] = inter_count_tail
        inter_count[inter_count==d] = -1
        inter_count[dist==self.params.shortest_path_dist,0]=link_arr[dist==self.params.shortest_path_dist,0]
        inter_count[dist==self.params.shortest_path_dist,1]=link_arr[dist==self.params.shortest_path_dist,1]
        # inter_count[dist==self.params.shortest_path_dist,2]=link_arr[dist==self.params.shortest_path_dist,0]
        # print(links)
        # print(dist)
        # print(inter_count)
        return links, dist.tolist(), inter_count.tolist()

    def re_label(self):
        # self.graph.ndata['label'][:, 0] = torch.randint(self.params.emb_dim, (1,self.graph.num_nodes()))
        # self.graph.ndata['feat'][:, :] = torch.nn.functional.one_hot(self.graph.ndata['label'], self.params.emb_dim).squeeze(1)
        self.graph.ndata['feat'][:, :] = torch.rand((self.graph.num_nodes(),self.init_dim))
        # +torch.arange(self.init_dim).unsqueeze(0)
        # self.graph.ndata['feat'][np.random.permutation(self.num_nodes)[:int(self.num_nodes/2)], 0] = 0
    
    def save_dist(self):
        d = []
        for i in range(len(self)):
            e = self[i]
            d.append(e[1][0])
        np.save('dist', np.array(d))


class MultiSampleDataset(Dataset):
    def __init__(self, triplets, dataset, params, adj_list, num_rels, num_entities, mode='train', ratio=0.1, neg_link_per_sample=1, sample_method=sample_neg_link):
        self.mode = mode
        self.edges = triplets[dataset]
        self.adj_list = adj_list
        self.coo_adj_list = [adj.tocoo() for adj in self.adj_list]
        self.num_edges = len(self.edges)
        self.num_nodes = num_entities
        self.num_rels = num_rels
        self.ratio = ratio
        self.params = params
        self.num_train_edges = 0
        self.num_graph_edges = 0
        self.graph = None
        self.adj_mat = None
        self.init_dim = 10
        if self.mode=='train':
            self.resample()
        else:
            self.train_edges = self.edges
            self.graph_edges = triplets['train']
        self.regraph()
        if params.use_random_labels:
            self.re_label()
        self.sample_links = sample_method

        self.neg_sample = neg_link_per_sample
        self.timer = SmartTimer(False)
        self.__getitem__(142)
        self.n_feat_dim = self.graph.ndata['feat'].shape[1]


    def __len__(self):
        return self.num_train_edges

    def __getitem__(self, index):
        self.timer.record()
        head, rel, tail = self.train_edges[index]
        neg_links = self.sample_links(self.coo_adj_list, rel, head, tail, self.num_nodes, self.neg_sample)
        pos_link = [head, rel, tail]
        links = [pos_link]+neg_links
        link_arr = np.array(links)
        self.timer.cal_and_update('sample')
        dis_mat_head, pred_mat_head = ssp.csgraph.shortest_path(self.adj_mat, indices=head, directed=False, unweighted=True, return_predecessors=True)
        dis_mat_tail, pred_mat_tail = ssp.csgraph.shortest_path(self.adj_mat, indices=tail, directed=False, unweighted=True, return_predecessors=True)
        self.timer.cal_and_update('ssp')
        d_head = np.clip(dis_mat_head, 1, self.params.shortest_path_dist)
        d_tail = np.clip(dis_mat_tail, 1, self.params.shortest_path_dist)
        dist = np.zeros(len(links), dtype=int)
        head_ind = link_arr[:,0]==head
        tail_ind = link_arr[:,2]==tail
        dist[head_ind] = d_head[link_arr[head_ind, 2]]
        dist[tail_ind] = d_tail[link_arr[tail_ind, 0]]
        dist_countdown = dist[head_ind]
        inter_count_head = np.zeros((np.sum(head_ind), self.params.shortest_path_dist+1), dtype=int)-1
        # pred_mat[0, np.arange(len(link_arr)), link_arr[:, 0]] = link_arr[:, 0]
        d = len(pred_mat_head)
        self.timer.cal_and_update('prepare')
        pred_mat_head = np.concatenate([pred_mat_head, np.array([d])])
        pred_mat_head[pred_mat_head==-9999] = d
        p = pred_mat_head[link_arr[head_ind, 2]]
        # inter_count_head[:,0] = link_arr[head_ind,2]
        inter_count_head[np.arange(len(inter_count_head)), dist_countdown] = link_arr[head_ind,2]
        dist_countdown = np.clip(dist_countdown-1, 0, self.params.shortest_path_dist)
        for i in range(1,self.params.shortest_path_dist+1):
            inter_count_head[np.arange(len(inter_count_head)),dist_countdown] = p
            dist_countdown = np.clip(dist_countdown-1, 0, self.params.shortest_path_dist)
            p = pred_mat_head[p]
        def swap_fun(a):
            c = np.sum(a!=d)
            if c==0:
                return a
            a[:c]=a[c-1::-1]
            return a
        # np.put_along_axis(inter_count_head, dist[head_ind].reshape((-1,1))-1, d, axis=1)
        self.timer.cal_and_update('findheadpath')
        # inter_count_head = np.apply_along_axis(swap_fun, 1, inter_count_head)
        inter_count_head[:,0] = link_arr[head_ind, 0]
        self.timer.cal_and_update('flipheadpath')
        if self.params.path_add_head:
            inter_count_head[np.arange(len(inter_count_head)), (dist[head_ind]+1)%(self.params.shortest_path_dist+1)] = link_arr[head_ind,0]
        # print('swap', time.time()-v)
        d = len(pred_mat_tail)
        inter_count_tail = np.zeros((np.sum(tail_ind), self.params.shortest_path_dist+1), dtype=int)-1
        pred_mat_tail = np.concatenate([pred_mat_tail, np.array([d])])
        pred_mat_tail[pred_mat_tail==-9999] = d
        p = pred_mat_tail[link_arr[tail_ind, 0]]
        inter_count_tail[:,0] = link_arr[tail_ind,0]
        for i in range(1,self.params.shortest_path_dist+1):
            inter_count_tail[:,i] = p
            p = pred_mat_tail[p]
        # np.put_along_axis(inter_count_tail, dist[tail_ind].reshape((-1,1))-1, d, axis=1)
        # inter_count[np.arange(len(inter_count)), dist.astype(int)]=-1
        self.timer.cal_and_update('findtailpath')
        if self.params.path_add_head:
            inter_count_tail[np.arange(len(inter_count_tail)),  (dist[tail_ind]+1)%(self.params.shortest_path_dist+1)] = link_arr[tail_ind,0]
        inter_count = np.zeros((len(links), self.params.shortest_path_dist+1), dtype=int)-1
        inter_count[head_ind] = inter_count_head
        inter_count[tail_ind] = inter_count_tail
        inter_count[inter_count==d] = -1
        inter_count[dist==self.params.shortest_path_dist]=-1
        self.timer.cal_and_update('finish')
        # inter_count[dist==self.params.shortest_path_dist,0]=link_arr[dist==self.params.shortest_path_dist,0]
        # inter_count[dist==self.params.shortest_path_dist,1]=link_arr[dist==self.params.shortest_path_dist,1]
        # inter_count[dist==self.params.shortest_path_dist,2]=link_arr[dist==self.params.shortest_path_dist,0]
        # print(links)
        # print(dist)
        # print(inter_count[2])
        return links, dist.tolist(), inter_count.tolist()

    def resample(self):
        print("train graph resampled")
        perm = np.random.permutation(self.num_edges)
        train_ind = int(self.num_edges*self.ratio)
        self.train_edges = self.edges[perm[:train_ind]]
        self.graph_edges = self.edges[perm[train_ind:]]

    def regraph(self):
        # train_edges, graph_edges = self.resample(self.ratio)
        self.num_train_edges = len(self.train_edges)
        self.num_graph_edges = len(self.graph_edges)
        self.graph = construct_reverse_graph_from_edges(self.graph_edges.T, self.num_nodes, self.num_rels)
        self.adj_mat = self.graph.adjacency_matrix(transpose=False, scipy_fmt='csr')
        self.graph.ndata['feat'] = torch.ones([self.num_nodes, self.init_dim], dtype=torch.float32)
        self.graph.ndata['label'] = torch.ones([self.num_nodes, 1], dtype=torch.int64)
        # return train_edges, graph_edges
        
    def re_label(self):
        # self.graph.ndata['label'][:, 0] = torch.randint(self.params.emb_dim, (1,self.graph.num_nodes()))
        # self.graph.ndata['feat'][:, :] = torch.nn.functional.one_hot(self.graph.ndata['label'], self.params.emb_dim).squeeze(1)
        self.graph.ndata['feat'][:, :] = torch.rand((self.graph.num_nodes(),self.init_dim))
        # +torch.arange(self.init_dim).unsqueeze(0)
        # self.graph.ndata['feat'][np.random.permutation(self.num_nodes)[:int(self.num_nodes/2)], 0] = 0



class SubgraphDatasetOnlyLinkVal(SubgraphDataset):
    def __init__(self, triplets, dataset, params, adj_list, num_rels, num_entities, neg_link_per_sample=1):
        super().__init__(triplets, dataset, params, adj_list, num_rels, num_entities, None, neg_link_per_sample)
        self.graph.ndata['feat'] = torch.ones([self.num_nodes, 1], dtype=torch.float32)
        self.__getitem__(113)
        self.n_feat_dim = self.graph.ndata['feat'].shape[1]

    def __len__(self):
        return self.sample_size

    def __getitem__(self, index):
        head, rel, tail = self.edges[index]
        neg_links = sample_neg_link(self.coo_adj_list, rel, head, tail, self.num_nodes, self.neg_sample)
        pos_link = [head, rel, tail]
        links = [pos_link]+neg_links
        link_arr = np.array(links)
        d_mat = np.clip(ssp.csgraph.shortest_path(self.adj_mat, indices=[link_arr[:,0]], directed=False, unweighted=True), 0, self.params.num_gcn_layers+1)
        dist = d_mat[0, np.arange(len(link_arr)), link_arr[:, 2]]
        return links, dist.tolist()


class SubgraphDatasetSpeTrain(SubgraphDataset):
    def __init__(self, triplets, dataset, params, adj_list, num_rels, num_entities, neg_link_per_sample=1):
        super().__init__(triplets, dataset, params, adj_list, num_rels, num_entities, None, neg_link_per_sample)

        pos_g, pos_la, pos_rel, pos_spe, neg_g, neg_la, neg_rel, neg_spe = self.__getitem__(113)
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
        main_subgraph, local_adj_mat, node_to_id = self._get_main_subgraph(node_set, self.params.train_max_n)
        if self.params.use_neighbor_feature:
            self._get_neighbor_edge_ratio(main_subgraph, 'ratio')
        local_adj_mat[node_to_id[pos_link[0]], node_to_id[pos_link[2]]]=0
        local_adj_mat[node_to_id[pos_link[2]], node_to_id[pos_link[0]]]=0  
        pos_nodes, pos_label, enc_nodes, root_dist = self._get_labels(node_to_id[pos_link[0]], node_to_id[pos_link[2]], rel, local_adj_mat)
        local_adj_mat[node_to_id[pos_link[0]], node_to_id[pos_link[2]]]=1
        local_adj_mat[node_to_id[pos_link[2]], node_to_id[pos_link[0]]]=1

        # if len(pos_nodes) == 2:
        #     print("Err")
        # print(pos_nodes)
        # print(len(pos_nodes))
        # if len(pos_nodes)==2 and not main_subgraph.has_edges_between(pos_nodes[1],pos_nodes[0]):
        #     print(rel)
        pos_subgraph, pos_spe, pos_label = self._get_spectrum_graph(pos_nodes, enc_nodes, rel, main_subgraph, pos_label,False) 
        pos_subgraph = self._prepare_node_features(pos_subgraph, pos_label, rel)
        logging.debug(f'sample one:{time.time()-st}')
        # print(pos_subgraph.edges())
        # blockPrint()
        neg_subgraphs = []
        neg_spes = []
        for i in range(self.neg_sample):
            neg_nodes, neg_label, enc_nodes, root_dist = self._get_labels(node_to_id[neg_links[i][0]], node_to_id[neg_links[i][2]], rel, local_adj_mat)
            neg_subgraph, neg_spe, neg_label = self._get_spectrum_graph(neg_nodes, enc_nodes, rel, main_subgraph, neg_label)
            neg_subgraph = self._prepare_node_features(neg_subgraph, neg_label, rel)
            neg_subgraph.add_edges([0], [1])
            neg_subgraph.edata['type'][-1] = torch.tensor(neg_links[i][1], dtype=torch.int32)
            neg_subgraph.edata['label'][-1] = torch.tensor(neg_links[i][1], dtype=torch.int32)
            neg_subgraphs.append(neg_subgraph)
            neg_spes.append(neg_spe)

        logging.debug(f'sampleall:{time.time()-st}')
        return pos_subgraph, 1, pos_link[1], pos_spe, neg_subgraphs, [0] * len(neg_subgraphs), [neg_links[i][1] for i in
                                                                                       range(len(neg_subgraphs))], neg_spes

class SubgraphDatasetSpeVal(SubgraphDataset):
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
        main_subgraph, local_adj_mat, node_to_id = self._get_main_subgraph(node_set, self.params.test_max_n)
        if self.params.use_neighbor_feature:
            self._get_neighbor_edge_ratio(main_subgraph, 'ratio')
        can_edges = [pos_link]+neg_links
        graphs = []
        spes = []
        for i, edge in enumerate(can_edges):
            pos_nodes, pos_label, enc_nodes, root_dist = self._get_labels(node_to_id[edge[0]], node_to_id[edge[2]], rel, local_adj_mat)
            # if i != 0:
            #     print(len(pos_nodes))
            # if i==0 and len(pos_nodes)==2 and not main_subgraph.has_edges_between(pos_nodes[0],pos_nodes[1]) and not main_subgraph.has_edges_between(pos_nodes[1],pos_nodes[0]):
            #     print(index)
            pos_subgraph, pos_spe, pos_label = self._get_spectrum_graph(pos_nodes, enc_nodes, rel, main_subgraph, pos_label) 
            pos_subgraph = self._prepare_node_features(pos_subgraph, pos_label, rel)
            pos_subgraph.add_edges([0],[1])
            pos_subgraph.edata['type'][-1] = torch.tensor(rel, dtype=torch.int32)
            pos_subgraph.edata['label'][-1] = torch.tensor(rel, dtype=torch.int32)
            # if index==43 and i==0:
            #     print(pos_subgraph.edges())
            #     print(pos_subgraph.edata['type'])
            graphs.append(pos_subgraph)
            spes.append(pos_spe)
        return graphs, [rel]*len(graphs), 0, spes


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

        pos_nodes, pos_label, disconnected_nodes, root_dist = subgraph_extraction_labeling_wiki([node_to_id[pos_link[0]], node_to_id[pos_link[2]]], pos_link[1], local_adj_mat, h=self.params.hop, max_nodes_per_hop=self.params.max_nodes_per_hop)
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
            neg_nodes, neg_label, disconnected_nodes, root_dist = subgraph_extraction_labeling_wiki([node_to_id[neg_links[i][0]], node_to_id[neg_links[i][2]]], neg_links[i][1], local_adj_mat, h=self.params.hop, max_nodes_per_hop=self.params.max_nodes_per_hop)

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
            pos_nodes, pos_label, disconnected_nodes, root_dist = subgraph_extraction_labeling_wiki([head, candidate], rel,
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
            pos_nodes, pos_label, disconnected_nodes, root_dist = subgraph_extraction_labeling_wiki([head, candidate], rel,
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

