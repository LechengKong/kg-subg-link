import os
from datasets import SubgraphDatasetWikiOnlineValSubset, SubgraphDatasetWikiLocalSubsetEval
from ogb.lsc import WikiKG90MDataset as wikiData
import torch
import numpy as np
from torch_util import collate_dgl, move_batch_to_device_dgl, collate_dgl_val, move_batch_to_device_dgl_val
from model.dgl.graph_classifier import GraphClassifier as dgl_model
from managers.trainer import Trainer
from managers.evaluator import Evaluator
import random


class Mem:

    def __init__(self):
        self.hop = 3
        self.enclosing_sub_graph = False
        self.max_nodes_per_hop = 30
        self.num_neg_samples_per_link = 2
        self.root_path = "/project/tantra/jerry.kong/ogb_project/dataset/wikikg90m_kddcup2021"
        self.adj_path = "/project/tantra/jerry.kong/ogb_project/dataset/wikikg90m_kddcup2021/adj_mat"
        self.num_rels = 1315
        self.rel_emb_dim = 48
        self.add_ht_emb = True
        self.num_gcn_layers = 6
        self.emb_dim = 48
        self.max_label_value = 10
        self.inp_dim = 1010
        self.attn_rel_emb_dim = 48
        self.aug_num_rels = 1315
        self.num_bases = 4
        self.num_hidden_layers = 4
        self.dropout = 0.25
        self.edge_dropout = 0.25
        self.has_attn = True
        self.gnn_agg_type = 'sum'
        self.optimizer = 'Adam'
        self.lr = 0.0001
        self.l2 = 1e-3
        self.batch_size = 4
        self.num_workers = 32
        self.num_epochs = 20
        self.save_every = 1
        self.exp_dir = "/project/tantra/jerry.kong/ogb_project/dataset/wikikg90m_kddcup2021/"
        self.margin = 5
        self.train_edges = 10000
        self.val_size = 1000
        self.eval_every_iter = 1
        self.early_stop = 5
        self.split = 'val'
        self.make_data = False
        self.val_batch_size = 2
        self.candidate_size = 1001
        self.prefetch_val = 2
        self.retrain = False
        self.retrain_seed = 111
        self.model_name = "graph_classifier_from_val_3hop_sm"
        self.feat_size = 768


if __name__ == '__main__':
    params = Mem()
    print("program start")
    if params.retrain:
        print("retrain model")
        torch.manual_seed(params.retrain_seed)
        random.seed(params.retrain_seed)
        np.random.seed(params.retrain_seed)
    else:
        torch.manual_seed(10)
        random.seed(10)
        np.random.seed(10)


    params.db_path = os.path.join(params.root_path,
                                  f'dbs/subgraphs_en_{params.enclosing_sub_graph}_neg_{params.num_neg_samples_per_link}_hop_{params.hop}')
    params.db_path_val = os.path.join(params.root_path,
                                  f'dbs/subgraphs_en_{params.enclosing_sub_graph}_neg_{params.num_neg_samples_per_link}_hop_{params.hop}_split_{params.split}')
    params.can_path_val = "/project/tantra/jerry.kong/ogb_project/dataset/wikikg90m_kddcup2021/filter_data/p_val_can.npy"
    params.t_ind_path_val = "/project/tantra/jerry.kong/ogb_project/dataset/wikikg90m_kddcup2021/filter_data/p_val_t_cur.npy"
    params.hr_path_val = "/project/tantra/jerry.kong/ogb_project/dataset/wikikg90m_kddcup2021/filter_data/p_val_hr.npy"
    params.can_path_train = "/project/tantra/jerry.kong/ogb_project/dataset/wikikg90m_kddcup2021/filter_data/train_can.npy"
    params.t_ind_path_train = "/project/tantra/jerry.kong/ogb_project/dataset/wikikg90m_kddcup2021/filter_data/train_t_cur.npy"
    params.hr_path_train = "/project/tantra/jerry.kong/ogb_project/dataset/wikikg90m_kddcup2021/filter_data/train_hr.npy"
    dataset = wikiData(root="/project/tantra/jerry.kong/ogb_project/dataset")
    params.num_rels = dataset.num_relations

    if torch.cuda.is_available():
        params.device = torch.device('cuda:0')
    else:
        params.device = torch.device('cpu')

    params.collate_fn = collate_dgl
    params.collate_fn_val = collate_dgl_val
    params.move_batch_to_device = move_batch_to_device_dgl
    params.move_batch_to_device_val = move_batch_to_device_dgl_val
    rel_to_id = np.arange(params.num_rels)
    
    torch.multiprocessing.set_sharing_strategy('file_system')

    train = SubgraphDatasetWikiOnlineValSubset(dataset, params, params.can_path_train, params.hr_path_train, params.t_ind_path_train, neg_link_per_sample=30, use_feature=False)
    test = SubgraphDatasetWikiLocalSubsetEval(dataset, params, train.graph, train.adj_mat, params.can_path_val, params.hr_path_val, t_ind_path=params.t_ind_path_val, sample_size=1000, use_feature=False)
    params.inp_dim = train.n_feat_dim
    graph_classifier = dgl_model(params, rel_to_id).to(device=params.device)

    state_d = None
    if params.retrain:
        state_d = torch.load(os.path.join(params.exp_dir, params.model_name+".pth"), map_location=params.device)

    validator = Evaluator(params, graph_classifier, test)
    trainer = Trainer(params, graph_classifier, train, state_dict=state_d, valid_evaluator=validator)
    trainer.train()
