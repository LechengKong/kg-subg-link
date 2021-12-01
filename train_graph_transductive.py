import torch
import random
import argparse
import numpy as np
import os.path as osp
import logging

from data_util import process_files
from datasets import SubgraphDatasetWhole, SubgraphDatasetOnlyLink, MultiSampleDataset
from torch_util import set_data_passing
from model.dgl.graph_classifier import GraphClassifier as dgl_model
from model.dgl.graph_classifier import GraphClassifierSpe as spe_dgl_model
from model.dgl.graph_classifier import GraphClassifierWhole as whole_dgl_model
from managers.trainer_whole import Trainer
from managers.evaluator_whole import Evaluator, EvaluatorVarLen
from graph_util import *

class Mem:

    def __init__(self):
        self.root_path = "/project/tantra/jerry.kong/ogb_project/ogb-grail-mod/data"
        self.data_path = "/project/tantra/jerry.kong/ogb_project/ogb-grail-mod/data"
        self.data_set = "WN18RR"
        self.ind_data_set = "WN18RR"
        self.num_rels = 1315
        self.aug_num_rels = 1315
        self.max_label_value = 10
        self.hop = 3
        self.intersection_nodes = True
        self.node_path_only = True
        self.same_size_neighbor = False
        self.max_nodes_per_hop = 15
        self.train_max_n = 500
        self.test_max_n = 25000
        self.rel_emb_dim = 16
        self.emb_dim = 16
        self.attn_rel_emb_dim = 16
        self.lstm_hidden_size = 16
        self.add_ht_emb = True
        self.has_attn = False
        self.node_attn = False
        self.sister_node_focus = False
        self.use_neighbor_feature = False
        self.num_gcn_layers = 6
        self.num_bases = 4
        self.dropout = 0
        self.edge_dropout = 0.3
        self.gnn_agg_type = 'sum'
        self.optimizer = 'Adam'
        self.lr = 0.001
        self.l2 = 1e-2
        self.batch_size = 128
        self.num_workers = 32
        self.num_epochs = 50
        self.save_every = 1
        self.eval_every_iter = 3
        self.margin = 15
        self.train_edges = 25901
        self.val_size = 1000
        self.train_neg_sample_size = 32
        self.val_neg_sample_size = 50
        self.sample_graph_ratio = 0.6
        self.early_stop = 100
        self.shortest_path_dist = 30
        self.val_batch_size = 1
        self.prefetch_val = 2
        self.return_pred_res = False
        self.retrain = False
        self.eval = False
        self.retrain_seed = 113
        self.use_data = ["train", "test", "valid"]
        self.device = torch.device('cpu')
        self.eig_size = 20
        self.use_spe = False
        self.simple_net = False
        self.use_root_dist = False
        self.whole_graph = False
        self.only_link_sample = True
        self.use_random_labels = True
        self.batch_random = True
        self.label_reg = True
        self.path_add_head = True
        self.use_lstm = True
        self.use_mid_repr = True
        self.no_regraph = True
        self.use_deep_set = False
        self.use_label_pred_out = False
        self.concat_init_feat = True
        self.eval_sample_method = sample_filtered_neg_head
        self.multisample_dataset = True
        self.reptition = 5
        self.eval_rep = 10
        self.save_res = True
        self.prefix = ("rand_" if self.use_random_labels else "")+("batchr_" if self.batch_random else "")+("reg_" if self.label_reg else "")+("addhead_" if self.path_add_head else "")+("lstm_" if self.use_lstm else "")+("noregraph_" if self.no_regraph else "") +self.ind_data_set+"_"
        self.model_name = self.prefix+"gnn"  #rand"lstm_gnn_tst" #"lstm_gnn"fbv1 "lstm_gnn_of" fbv1 random
        self.res_save_name = self.prefix+"metricmean"

    def update_prefix(self):
        self.prefix = ("rand_" if self.use_random_labels else "")+("batchr_" if self.batch_random else "")+("reg_" if self.label_reg else "")+("addhead_" if self.path_add_head else "")+("lstm_" if self.use_lstm else "")+("noregraph_" if self.no_regraph else "") +self.ind_data_set+"_"
        self.model_name = self.prefix+"gnn"  #rand"lstm_gnn_tst" #"lstm_gnn"fbv1 "lstm_gnn_of" fbv1 random
        self.res_save_name = self.prefix+"metricmean"


if __name__ == '__main__':
    params = Mem()
    print(params.eval_sample_method)
    parser = argparse.ArgumentParser(description='gnn')

    parser.add_argument("--use_random_labels", type=bool, default=False)
    parser.add_argument("--batch_random", type=bool, default=False)
    parser.add_argument("--label_reg", type=bool, default=False)
    parser.add_argument("--path_add_head", type=bool, default=False)
    parser.add_argument("--use_lstm", type=bool, default=False)
    parser.add_argument("--no_regraph", type=bool, default=False)
    parser.add_argument("--reptition", type=int, default=5)
    parser.add_argument("--sample_graph_ratio", type=float, default=0.5)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--eval", type=bool, default=False)
    parser.add_argument("--eval_rep", type=int, default=10)
    args = parser.parse_args()
    params.use_random_labels = args.use_random_labels
    params.batch_random = args.batch_random
    params.label_reg = args.label_reg
    params.path_add_head = args.path_add_head
    params.use_lstm = args.use_lstm
    params.no_regraph = args.no_regraph
    params.reptition = args.reptition
    params.sample_graph_ratio = args.sample_graph_ratio
    params.num_epochs = args.num_epochs
    params.eval = args.eval
    params.eval_rep = args.eval_rep
    params.update_prefix()
    print(params.prefix)
    if params.retrain:
        print("retrain model")
        torch.manual_seed(params.retrain_seed)
        random.seed(params.retrain_seed)
        np.random.seed(params.retrain_seed)
    # elif not params.eval:
    #     torch.manual_seed(11)
    #     random.seed(11)
    #     np.random.seed(11)
    elif not params.eval:
        torch.manual_seed(7)
        random.seed(7)
        np.random.seed(7)

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
    params.aug_num_rels = train_num_rel*2
    print(f'Dataset {params.data_set} has {train_num_entities} entities and {train_num_rel} relations')

    ind_database_path = osp.join(params.data_path, params.ind_data_set)
    files = {}
    for f in params.use_data:
        files[f] = osp.join(ind_database_path,f'{f}.txt')

    adj_list_ind, converted_triplets_ind, entity2id_ind, relation2id_ind, id2entity_ind, id2relation_ind = process_files(files, relation2id=relation2id)
    
    ind_num_entities = len(entity2id_ind)
    ind_num_rel = len(relation2id_ind)
    print(f'Dataset {params.ind_data_set} has {ind_num_entities} entities and {ind_num_rel} relations')
    set_data_passing(params)
    torch.multiprocessing.set_sharing_strategy('file_system')
    if params.only_link_sample:
        TrainSet = SubgraphDatasetOnlyLink
        TestSet = SubgraphDatasetOnlyLink
    else:
        TrainSet = SubgraphDatasetWhole
        TestSet = SubgraphDatasetWhole

    if params.multisample_dataset:
        TrainSet = MultiSampleDataset
        TestSet = MultiSampleDataset

    # train = TrainSet(converted_triplets, 'train', params, adj_list, train_num_rel, train_num_entities, ratio=params.sample_graph_ratio, neg_link_per_sample=params.train_neg_sample_size)
    train = TrainSet(converted_triplets, 'train', params, adj_list, train_num_rel, train_num_entities, neg_link_per_sample=params.train_neg_sample_size, ratio=params.sample_graph_ratio)
    if params.eval:
        # val = TrainSet(converted_triplets, 'train', params, adj_list, train_num_rel, train_num_entities, ratio=params.sample_graph_ratio, neg_link_per_sample=50)
        val = TestSet(converted_triplets_ind, 'test', params, adj_list_ind, ind_num_rel, ind_num_entities, neg_link_per_sample=params.val_neg_sample_size, sample_method=params.eval_sample_method, mode='valid')
        # val = TestSet(converted_triplets, 'valid', params, adj_list, train_num_rel, train_num_entities, neg_link_per_sample=params.val_neg_sample_size)
        # val.save_dist()
    else:
        # val = TestSet(converted_triplets_ind, 'valid', params, adj_list_ind, ind_num_rel, ind_num_entities, neg_link_per_sample=50)
        val_head = TestSet(converted_triplets_ind, 'valid', params, adj_list_ind, ind_num_rel, ind_num_entities, neg_link_per_sample=params.val_neg_sample_size, sample_method=sample_filtered_neg_tail, mode='valid')
        val_tail = TestSet(converted_triplets_ind, 'valid', params, adj_list_ind, ind_num_rel, ind_num_entities, neg_link_per_sample=params.val_neg_sample_size, sample_method=sample_filtered_neg_head, mode='valid')
    params.train_edges = len(train)
    params.val_size = len(val_head)
    print(f'Training set has {params.train_edges} edges, Val set has {params.val_size} edges')
    params.inp_dim = train.n_feat_dim
    if params.use_neighbor_feature:
        params.inp_dim += params.rel_emb_dim

    params2 = Mem()
    params2.inp_dim = 1
    params2.emb_dim = 32
    params2.attn_rel_emb_dim = 32
    params2.dropout = 0.5
    params2.edge_dropout = 0.5

    state_d = None
    if params.retrain:
        state_d = torch.load(osp.join(params.root_path, params.model_name+"_0.pth"), map_location=params.device)

    if params.eval:
        mrr = []
        h10 = []
        for i in range(params.reptition):
            graph_classifier = whole_dgl_model(params, params2, relation2id).to(device=params.device)
            state_d = torch.load(osp.join(params.root_path, "best_"+params.model_name+'_'+str(i)+".pth"), map_location=params.device)
            graph_classifier.load_state_dict(state_d['state_dict'])
            validator = EvaluatorVarLen(params, graph_classifier, val)
            res = validator.eval(params.eval_rep)
            mrr.append(res['mrr'])
            h10.append(res['h10'])
        mrr = np.array(mrr)
        h10 = np.array(h10)
        print(mrr, np.mean(mrr), np.std(mrr))
        print(h10, np.mean(h10), np.std(h10))
    else:
        val_list = []
        train_list = []
        res = []
        for i in range(params.reptition):
            graph_classifier = whole_dgl_model(params, params2, relation2id).to(device=params.device)
            validator = EvaluatorVarLen(params, graph_classifier, [val_head,val_tail])
            trainer = Trainer(params, graph_classifier, train, state_dict=state_d, valid_evaluator=validator, label=i)
            res.append(trainer.train())
            val_list.append(validator)
            train_list.append(trainer)
        if params.save_res:
            np.save('log/'+params.prefix+'allres', np.array(res))
        print('after train evaluation:', params.prefix)
        val_head = TestSet(converted_triplets_ind, 'valid', params, adj_list_ind, ind_num_rel, ind_num_entities, neg_link_per_sample=params.val_neg_sample_size, sample_method=sample_filtered_neg_tail, mode='valid')
        val_tail = TestSet(converted_triplets_ind, 'valid', params, adj_list_ind, ind_num_rel, ind_num_entities, neg_link_per_sample=params.val_neg_sample_size, sample_method=sample_filtered_neg_head, mode='valid')
        mrr = []
        h10 = []
        for i in range(params.reptition):
            graph_classifier = whole_dgl_model(params, params2, relation2id).to(device=params.device)
            state_d = torch.load(osp.join(params.root_path, "best_"+params.model_name+'_'+str(i)+".pth"), map_location=params.device)
            graph_classifier.load_state_dict(state_d['state_dict'])
            validator = EvaluatorVarLen(params, graph_classifier, [val_head,val_tail])
            res = validator.eval(params.eval_rep)
            mrr.append(res['mrr'])
            h10.append(res['h10'])
        mrr = np.array(mrr)
        h10 = np.array(h10)
        print(mrr, np.mean(mrr), np.std(mrr))
        print(h10, np.mean(h10), np.std(h10))
