import torch
import random
import numpy as np
import os.path as osp
import logging

from data_util import process_files
from datasets import SubgraphDatasetTrain, SubgraphDatasetVal, SubgraphDatasetSpeVal, SubgraphDatasetSpeTrain
from torch_util import set_data_passing
from model.dgl.graph_classifier import GraphClassifier as dgl_model
from model.dgl.graph_classifier import GraphClassifierSpe as spe_dgl_model
from managers.trainer import Trainer
from managers.evaluator import Evaluator

class Mem:

    def __init__(self):
        self.root_path = "/project/tantra/jerry.kong/ogb_project/ogb-grail-mod/data"
        self.data_path = "/project/tantra/jerry.kong/ogb_project/ogb-grail-mod/data"
        self.data_set = "WN18RR_v4"
        self.ind_data_set = "WN18RR_v4_ind"
        self.num_rels = 1315
        self.aug_num_rels = 1315
        self.max_label_value = 10
        self.hop = 3
        self.intersection_nodes = True
        self.node_path_only = True
        self.same_size_neighbor = False
        self.max_nodes_per_hop = 30
        self.rel_emb_dim = 16
        self.emb_dim = 16
        self.attn_rel_emb_dim = 16
        self.add_ht_emb = True
        self.has_attn = True
        self.node_attn = False
        self.sister_node_focus = False
        self.use_neighbor_feature = False
        self.num_gcn_layers = 4
        self.num_bases = 4
        self.dropout = 0
        self.edge_dropout = 0.5
        self.gnn_agg_type = 'sum'
        self.optimizer = 'Adam'
        self.lr = 0.001
        self.l2 = 1e-2
        self.batch_size = 32
        self.num_workers = 32
        self.num_epochs = 5
        self.save_every = 1
        self.eval_every_iter = 1
        self.margin = 5
        self.train_edges = 25901
        self.val_size = 3097
        self.early_stop = 5
        self.val_batch_size = 8
        self.prefetch_val = 2
        self.return_pred_res = False
        self.retrain = False
        self.eval = False
        self.retrain_seed = 114
        self.model_name = "graph_classifier_att_sm_ind"
        self.use_data = ["train", "test", "valid"]
        self.device = torch.device('cpu')
        self.eig_size = 20
        self.use_spe = False
        self.simple_net = False


if __name__ == '__main__':
    params = Mem()
    if params.retrain:
        print("retrain model")
        torch.manual_seed(params.retrain_seed)
        random.seed(params.retrain_seed)
        np.random.seed(params.retrain_seed)
    elif not params.eval:
        torch.manual_seed(11)
        random.seed(11)
        np.random.seed(11)

    if torch.cuda.is_available():
        params.device = torch.device('cuda:0')
    else:
        params.device = torch.device('cpu')

    if params.use_spe:
        params.same_size_neighbor = True

    if params.same_size_neighbor:
        params.intersection_nodes = False
        params.node_path_only = False
    else:
        params.intersection_nodes = True
        params.node_path_only = True
    

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
    params.aug_num_rels = train_num_rel
    print(f'Dataset {params.data_set} has {train_num_entities} entities and {train_num_rel} relations')

    ind_database_path = osp.join(params.data_path, params.ind_data_set)
    files = {}
    for f in params.use_data:
        files[f] = osp.join(ind_database_path,f'{f}.txt')

    adj_list_ind, converted_triplets_ind, entity2id_ind, relation2id_ind, id2entity_ind, id2relation_ind = process_files(files, relation2id=relation2id)
    
    ind_num_entities = len(entity2id_ind)
    ind_num_rel = len(relation2id_ind)
    print(f'Dataset {params.ind_data_set} has {ind_num_entities} entities and {ind_num_rel} relations')

    # params.collate_fn = collate_dgl
    # params.collate_fn_val = collate_dgl_val
    # params.move_batch_to_device = move_batch_to_device_dgl
    # params.move_batch_to_device_val = move_batch_to_device_dgl_val
    set_data_passing(params)
    torch.multiprocessing.set_sharing_strategy('file_system')
    if params.use_spe:
        TrainSet = SubgraphDatasetSpeTrain
        TestSet = SubgraphDatasetSpeVal
    else:
        TrainSet = SubgraphDatasetTrain
        TestSet = SubgraphDatasetVal

    train = TrainSet(converted_triplets, 'train', params, adj_list, train_num_rel, train_num_entities, neg_link_per_sample=1)
    if params.eval:
        val = TestSet(converted_triplets_ind, 'test', params, adj_list_ind, ind_num_rel, ind_num_entities, neg_link_per_sample=50)
    else:
        val = TestSet(converted_triplets_ind, 'valid', params, adj_list_ind, ind_num_rel, ind_num_entities, neg_link_per_sample=50)
    params.train_edges = len(train)
    params.val_size = len(val)
    print(f'Training set has {params.train_edges} edges, Val set has {params.val_size} edges')
    params.inp_dim = train.n_feat_dim
    if params.use_spe:
        graph_classifier = spe_dgl_model(params, relation2id).to(device=params.device)
    else:
        graph_classifier = dgl_model(params, relation2id).to(device=params.device)

    state_d = None
    if params.retrain:
        state_d = torch.load(osp.join(params.root_path, params.model_name+".pth"), map_location=params.device)

    if params.eval:
        params.return_pred_res = True
        state_d = torch.load(osp.join(params.root_path, "best_"+params.model_name+".pth"), map_location=params.device)
        graph_classifier.load_state_dict(state_d['state_dict'])
        validator = Evaluator(params, graph_classifier, val)
        n=5
        mrr = []
        h10 = []
        apr = []
        for i in range(n):
            res = validator.eval()
            mrr.append(res['mrr'])
            h10.append(res['h10'])
            apr.append(res['apr'])
        print(mrr, np.mean(np.array(mrr)))
        print(h10, np.mean(np.array(h10)))
        print(apr, np.mean(np.array(apr)))
        np.save('el', res['h10l'])
    else:
        validator = Evaluator(params, graph_classifier, val)
        trainer = Trainer(params, graph_classifier, train, state_dict=state_d, valid_evaluator=validator)
        trainer.train()
