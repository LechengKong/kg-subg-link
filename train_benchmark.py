import torch
import random
import numpy as np
import os.path as osp
import logging

from data_util import process_files
from datasets import SubgraphDatasetTrain, SubgraphDatasetVal, SubgraphDatasetContextTrain, SubgraphDatasetContextVal
from torch_util import collate_dgl, collate_dgl_val, move_batch_to_device_dgl, move_batch_to_device_dgl_val
from model.dgl.graph_classifier import GraphClassifier as dgl_model
from managers.trainer import Trainer
from managers.evaluator import Evaluator
from torch.utils.data import Subset

class Mem:

    def __init__(self):
        self.hop = 2
        self.enclosing_sub_graph = False
        self.max_nodes_per_hop = 40
        self.num_neg_samples_per_link = 2
        self.root_path = "/project/tantra/jerry.kong/ogb_project/ogb-grail-mod/data"
        self.data_path = "/project/tantra/jerry.kong/ogb_project/ogb-grail-mod/data"
        self.data_set = "WN18RR"
        self.num_rels = 1315
        self.rel_emb_dim = 16
        self.add_ht_emb = True
        self.num_gcn_layers = 3
        self.emb_dim = 16
        self.max_label_value = 10
        self.inp_dim = 1010
        self.attn_rel_emb_dim = 8
        self.aug_num_rels = 1315
        self.num_bases = 4
        self.num_hidden_layers = 2
        self.dropout = 0.5
        self.edge_dropout = 0.5
        self.has_attn = True
        self.gnn_agg_type = 'sum'
        self.optimizer = 'Adam'
        self.lr = 0.001
        self.l2 = 1e-2
        self.batch_size = 16
        self.num_workers = 16
        self.num_epochs = 20
        self.save_every = 1
        self.margin = 10
        self.train_edges = 86835
        self.val_size = 3034
        self.eval_every_iter = 1
        self.early_stop = 5
        self.split = 'val'
        self.make_data = False
        self.val_batch_size = 8
        self.candidate_size = 1001
        self.prefetch_val = 2
        self.retrain = False
        self.retrain_seed = 112
        self.model_name = "graph_classifier_context"
        self.feat_size = 768
        self.use_data = ["train", "test", "valid"]
        self.device = torch.device('cpu')


if __name__ == '__main__':
    params = Mem()
    if params.retrain:
        print("retrain model")
        torch.manual_seed(params.retrain_seed)
        random.seed(params.retrain_seed)
        np.random.seed(params.retrain_seed)
    else:
        torch.manual_seed(10)
        random.seed(10)
        np.random.seed(10)

    if torch.cuda.is_available():
        params.device = torch.device('cuda:0')
    else:
        params.device = torch.device('cpu')

    logging.basicConfig(level=logging.INFO)

    database_path = osp.join(params.data_path, params.data_set)
    files = {}
    for f in params.use_data:
        files[f] = osp.join(database_path,f'{f}.txt')

    adj_list, converted_triplets, entity2id, relation2id, id2entity, id2relation = process_files(files)
    params.num_entities = len(entity2id)
    params.num_rels = len(relation2id)
    print(f'Dataset {params.data_set} has {params.num_entities} entities and {params.num_rels} relations')

    params.collate_fn = collate_dgl
    params.collate_fn_val = collate_dgl_val
    params.move_batch_to_device = move_batch_to_device_dgl
    params.move_batch_to_device_val = move_batch_to_device_dgl_val
    torch.multiprocessing.set_sharing_strategy('file_system')

    train = SubgraphDatasetContextTrain(converted_triplets, params, adj_list, params.num_rels, params.num_entities, neg_link_per_sample=25)
    val = SubgraphDatasetContextVal(converted_triplets, 'valid', params, adj_list, params.num_rels, params.num_entities, graph=train.graph, neg_link_per_sample=50)
    params.train_edges = len(train)
    params.val_size = len(val)
    print(f'Training set has {params.train_edges} edges, Val set has {params.val_size} edges')
    # evens = list(range(0,len(train),2))
    # odds = list(range(1,len(train),2))
    # print("len:",len(train),len(val))
    # print("sublen:",len(evens),len(odds))
    # train_subset = Subset(train,evens)
    # val_subset = Subset(val, odds)
    # print("setlen:",len(train_subset),len(val_subset))
    params.inp_dim = train.n_feat_dim
    graph_classifier = dgl_model(params, relation2id).to(device=params.device)

    state_d = None
    if params.retrain:
        state_d = torch.load(osp.join(params.root_path, params.model_name+".pth"), map_location=params.device)

    validator = Evaluator(params, graph_classifier, val)
    trainer = Trainer(params, graph_classifier, train, state_dict=state_d, valid_evaluator=validator)
    trainer.train()
