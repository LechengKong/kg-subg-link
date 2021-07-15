import torch
import random
import numpy as np
import os.path as osp
import logging

from data_util import process_files
from datasets import SubgraphDatasetTrain, SubgraphDatasetVal
from torch_util import collate_dgl, collate_dgl_val, move_batch_to_device_dgl, move_batch_to_device_dgl_val
from model.dgl.graph_classifier import GraphClassifier as dgl_model
from managers.trainer import Trainer
from managers.evaluator import Evaluator

class Mem:

    def __init__(self):
        self.hop = 3
        self.enclosing_sub_graph = False
        self.max_nodes_per_hop = 30
        self.num_neg_samples_per_link = 2
        self.root_path = "/project/tantra/jerry.kong/ogb_project/ogb-grail-mod/data"
        self.data_path = "/project/tantra/jerry.kong/ogb_project/ogb-grail-mod/data"
        self.data_set = "WN18RR"
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
        self.num_workers = 8
        self.num_epochs = 20
        self.save_every = 1
        self.margin = 5
        self.train_edges = 10000
        self.val_size = 3034
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

    train = SubgraphDatasetTrain(converted_triplets['train'], params, adj_list, params.num_rels, params.num_entities, neg_link_per_sample=10)
    val = SubgraphDatasetVal(converted_triplets['valid'], params, adj_list, params.num_rels, params.num_entities, neg_link_per_sample=500)
    params.inp_dim = train.n_feat_dim
    graph_classifier = dgl_model(params, relation2id).to(device=params.device)

    state_d = None
    if params.retrain:
        state_d = torch.load(osp.join(params.root_path, params.model_name+".pth"), map_location=params.device)

    validator = Evaluator(params, graph_classifier, val)
    trainer = Trainer(params, graph_classifier, train, state_dict=state_d, valid_evaluator=validator)
    trainer.train()
