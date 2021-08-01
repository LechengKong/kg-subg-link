import torch
import random
import numpy as np
import os.path as osp
import logging

from data_util import process_files
from datasets import SubgraphDatasetTrain, SubgraphDatasetVal, SubgraphDatasetContextTrain, SubgraphDatasetContextVal, SubgraphDatasetConnectTrain, SubgraphDatasetConnectVal, SubgraphDatasetNoSubTrain, SubgraphDatasetNoSubVal
from torch_util import collate_dgl, collate_dgl_val, move_batch_to_device_dgl, move_batch_to_device_dgl_val
from model.dgl.graph_classifier import GraphClassifier as dgl_model
from managers.trainer import Trainer
from managers.evaluator import Evaluator

class Mem:

    def __init__(self):
        self.hop = 2
        self.enclosing_sub_graph = False
        self.max_nodes_per_hop = 30
        self.num_neg_samples_per_link = 2
        self.root_path = "/project/tantra/jerry.kong/ogb_project/ogb-grail-mod/data"
        self.data_path = "/project/tantra/jerry.kong/ogb_project/ogb-grail-mod/data"
        self.data_set = "WN18RR_v3_resample"
        self.ind_data_set = "WN18RR_v3_ind_resample"
        self.num_rels = 1315
        self.rel_emb_dim = 8
        self.add_ht_emb = True
        self.num_gcn_layers = 3
        self.emb_dim = 8
        self.max_label_value = 10
        self.inp_dim = 1010
        self.attn_rel_emb_dim = 8
        self.aug_num_rels = 11
        self.num_bases = 2
        self.num_hidden_layers = 4
        self.dropout = 0
        self.edge_dropout = 0.5
        self.has_attn = True
        self.gnn_agg_type = 'sum'
        self.optimizer = 'Adam'
        self.lr = 0.001
        self.l2 = 1e-2
        self.batch_size = 16
        self.num_workers = 32
        self.num_epochs = 20
        self.save_every = 1
        self.margin = 10
        self.train_edges = 25901
        self.val_size = 3097
        self.eval_every_iter = 1
        self.early_stop = 5
        self.split = 'val'
        self.make_data = False
        self.val_batch_size = 4
        self.candidate_size = 1001
        self.prefetch_val = 2
        self.retrain = False
        self.eval = False
        self.retrain_seed = 111
        self.model_name = "graph_classifier_att_sm_ind"
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
    elif not params.eval:
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
    # print(relation2id)
    train_num_entities = len(entity2id)
    train_num_rel = len(relation2id)
    params.num_rels = train_num_rel
    print(f'Dataset {params.data_set} has {train_num_entities} entities and {train_num_rel} relations')

    ind_database_path = osp.join(params.data_path, params.ind_data_set)
    files = {}
    for f in params.use_data:
        files[f] = osp.join(ind_database_path,f'{f}.txt')

    adj_list_ind, converted_triplets_ind, entity2id_ind, relation2id_ind, id2entity_ind, id2relation_ind = process_files(files, relation2id=relation2id)
    
    ind_num_entities = len(entity2id_ind)
    ind_num_rel = len(relation2id_ind)
    print(f'Dataset {params.ind_data_set} has {ind_num_entities} entities and {ind_num_rel} relations')

    params.collate_fn = collate_dgl
    params.collate_fn_val = collate_dgl_val
    params.move_batch_to_device = move_batch_to_device_dgl
    params.move_batch_to_device_val = move_batch_to_device_dgl_val
    torch.multiprocessing.set_sharing_strategy('file_system')

    train = SubgraphDatasetTrain(converted_triplets, params, adj_list, train_num_rel, train_num_entities, neg_link_per_sample=50)
    if params.eval:
        val = SubgraphDatasetVal(converted_triplets_ind, 'test', params, adj_list_ind, ind_num_rel, ind_num_entities, neg_link_per_sample=50)
    else:
        val = SubgraphDatasetVal(converted_triplets_ind, 'valid', params, adj_list_ind, ind_num_rel, ind_num_entities, neg_link_per_sample=50)
    params.train_edges = len(train)
    params.val_size = len(val)
    print(f'Training set has {params.train_edges} edges, Val set has {params.val_size} edges')
    params.inp_dim = train.n_feat_dim
    graph_classifier = dgl_model(params, relation2id).to(device=params.device)

    state_d = None
    if params.retrain:
        state_d = torch.load(osp.join(params.root_path, params.model_name+".pth"), map_location=params.device)

    if params.eval:
        state_d = torch.load(osp.join(params.root_path, "best_"+params.model_name+".pth"), map_location=params.device)
        graph_classifier.load_state_dict(state_d['state_dict'])
        validator = Evaluator(params, graph_classifier, val)
        n=5
        h10 = []
        apr = []
        for i in range(n):
            res = validator.eval()
            h10.append(res['h10'])
            apr.append(res['apr'])
        print(h10, np.mean(np.array(h10)))
        print(apr, np.mean(np.array(apr)))
        # np.save('att', res['h10l'])
    else:
        validator = Evaluator(params, graph_classifier, val)
        trainer = Trainer(params, graph_classifier, train, state_dict=state_d, valid_evaluator=validator)
        trainer.train()
