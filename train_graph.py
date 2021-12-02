import torch
import random
import argparse
import numpy as np
import os.path as osp
import logging

from data_util import process_files
from datasets import SubgraphDatasetWhole, SubgraphDatasetOnlyLink, MultiSampleDataset, FullGraphDataset
from torch_util import set_data_passing
from model.dgl.graph_classifier import GraphClassifier as dgl_model
from model.dgl.graph_classifier import GraphClassifierSpe as spe_dgl_model
from model.dgl.graph_classifier import GraphClassifierWhole as whole_dgl_model
from managers.trainer_whole import Trainer
from managers.evaluator_whole import Evaluator, EvaluatorVarLen
from graph_util import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='gnn')

    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--root_path", type=str, default="/project/tantra/jerry.kong/ogb_project/ogb-grail-mod/data")
    parser.add_argument("--log_path", type=str, default="./log")
    parser.add_argument("--data_set", type=str, default="fb237_v1")
    parser.add_argument("--ind_data_set", type=str, default="fb237_v1_ind")
    parser.add_argument("--transductive", type=bool, default=False)

    parser.add_argument("--rel_emb_dim", type=int, default=64)
    parser.add_argument("--emb_dim", type=int, default=64)
    parser.add_argument("--attn_rel_emb_dim", type=int, default=64)
    parser.add_argument("--lstm_hidden_size", type=int, default=64)
    parser.add_argument("--num_gcn_layers", type=int, default=6)
    parser.add_argument("--num_bases",type=int, default=4)
    parser.add_argument("--dropout",type=float,default=0)
    parser.add_argument("--edge_dropout",type=float,default=0.5)
    parser.add_argument("--has_attn", type=bool, default=False)

    parser.add_argument("--gnn_agg_type",type=str, default='sum')
    parser.add_argument("--optimizer",type=str, default='Adam')
    parser.add_argument("--lr",type=float,default=0.001)
    parser.add_argument("--l2",type=float,default=0.01)
    parser.add_argument("--batch_size",type=int, default=128)
    parser.add_argument("--val_batch_size", type=int, default=1)
    parser.add_argument("--margin",type=float, default=15)

    parser.add_argument("--num_workers",type=int, default=16)
    parser.add_argument("--prefetch_val", type=int, default=2)
    parser.add_argument("--save_every",type=int, default=1)
    parser.add_argument("--eval_every_iter",type=int, default=1)
    parser.add_argument("--early_stop",type=int, default=100)
    parser.add_argument("--save_res", type=bool, default=True)
    parser.add_argument("--return_pred_res",type=bool, default=False)

    parser.add_argument("--train_neg_sample_size",type=int, default=50)
    parser.add_argument("--val_neg_sample_size",type=int, default=50)
    parser.add_argument("--regraph", type=bool, default=False)
    parser.add_argument("--sample_graph_ratio", type=float, default=0.5)

    parser.add_argument("--retrain", type=bool, default=False)
    parser.add_argument("--retrain_seed", type=int, default=100)
    parser.add_argument("--reptition", type=int, default=5)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--eval", type=bool, default=False)
    parser.add_argument("--eval_rep", type=int, default=10)

    parser.add_argument("--shortest_path_dist", type=int, default=30)
    parser.add_argument("--use_deep_set", type=bool, default=False)
    parser.add_argument("--concat_init_feat", type=bool, default=True)
    parser.add_argument("--add_ht_emb", type=bool, default=True)
    parser.add_argument("--use_random_labels", type=bool, default=False)
    parser.add_argument("--batch_random", type=bool, default=False)
    parser.add_argument("--label_reg", type=bool, default=False)
    parser.add_argument("--path_add_head", type=bool, default=False)
    parser.add_argument("--use_lstm", type=bool, default=False)
    parser.add_argument("--use_mid_repr", type=bool, default=True)

    parser.add_argument("--gpuid", type=int, default=0)
    params = parser.parse_args()
    params.max_label_value=10
    params.prefix = ("rand_" if params.use_random_labels else "")+("batchr_" if params.batch_random else "")+("reg_" if params.label_reg else "")+("addhead_" if params.path_add_head else "")+("lstm_" if params.use_lstm else "")+("regraph_" if params.regraph else "") +params.ind_data_set+"_"
    params.model_name = params.prefix+"gnn"
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
        params.device = torch.device('cuda:'+str(params.gpuid))
    else:
        params.device = torch.device('cpu')

    logging.basicConfig(level=logging.INFO)

    database_path = osp.join(params.data_path, params.data_set)
    files = {}
    use_data = ["train", "test", "valid"]
    for f in use_data:
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
    for f in use_data:
        files[f] = osp.join(ind_database_path,f'{f}.txt')

    adj_list_ind, converted_triplets_ind, entity2id_ind, relation2id_ind, id2entity_ind, id2relation_ind = process_files(files, relation2id=relation2id)
    
    ind_num_entities = len(entity2id_ind)
    ind_num_rel = len(relation2id_ind)
    print(f'Dataset {params.ind_data_set} has {ind_num_entities} entities and {ind_num_rel} relations')
    set_data_passing(params)
    torch.multiprocessing.set_sharing_strategy('file_system')

    TrainSet = FullGraphDataset
    TestSet = FullGraphDataset

    # train = TrainSet(converted_triplets, 'train', params, adj_list, train_num_rel, train_num_entities, ratio=params.sample_graph_ratio, neg_link_per_sample=params.train_neg_sample_size)
    train = TrainSet(converted_triplets, 'train', params, adj_list, train_num_rel, train_num_entities, neg_link_per_sample=params.train_neg_sample_size, ratio=params.sample_graph_ratio)
    if params.transductive:
        test =[TestSet(converted_triplets_ind, 'test', params, adj_list_ind, ind_num_rel, ind_num_entities, neg_link_per_sample=params.val_neg_sample_size, sample_method=sample_filtered_neg_tail, mode='valid'), TestSet(converted_triplets_ind, 'test', params, adj_list_ind, ind_num_rel, ind_num_entities, neg_link_per_sample=params.val_neg_sample_size, sample_method=sample_filtered_neg_head, mode='valid')]
        val = [TestSet(converted_triplets_ind, 'valid', params, adj_list_ind, ind_num_rel, ind_num_entities, neg_link_per_sample=params.val_neg_sample_size, sample_method=sample_filtered_neg_tail, mode='valid'),TestSet(converted_triplets_ind, 'valid', params, adj_list_ind, ind_num_rel, ind_num_entities, neg_link_per_sample=params.val_neg_sample_size, sample_method=sample_filtered_neg_head, mode='valid')]
    else:
        test = [TestSet(converted_triplets_ind, 'test', params, adj_list_ind, ind_num_rel, ind_num_entities, neg_link_per_sample=params.val_neg_sample_size, sample_method=sample_neg_link, mode='valid')]
        val = [TestSet(converted_triplets, 'valid', params, adj_list, train_num_rel, train_num_entities, neg_link_per_sample=params.val_neg_sample_size, sample_method=sample_neg_link, mode='valid')]
    params.train_edges = len(train)
    params.val_size = len(val[0])
    print(f'Training set has {params.train_edges} edges, Val set has {params.val_size} edges')
    params.inp_dim = train.n_feat_dim

    state_d = None
    if params.retrain:
        state_d = torch.load(osp.join(params.root_path, params.model_name+"_0.pth"), map_location=params.device)

    if params.eval:
        mrr = []
        h10 = []
        for i in range(params.reptition):
            graph_classifier = whole_dgl_model(params, relation2id).to(device=params.device)
            state_d = torch.load(osp.join(params.root_path, "best_"+params.model_name+'_'+str(i)+".pth"), map_location=params.device)
            graph_classifier.load_state_dict(state_d['state_dict'])
            validator = EvaluatorVarLen(params, graph_classifier, test)
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
            graph_classifier = whole_dgl_model(params, relation2id).to(device=params.device)
            validator = EvaluatorVarLen(params, graph_classifier, val)
            trainer = Trainer(params, graph_classifier, train, state_dict=state_d, valid_evaluator=validator, label=i)
            res.append(trainer.train())
            val_list.append(validator)
            train_list.append(trainer)
        if params.save_res:
            np.save(params.log_path+'/'+params.prefix+'allres', np.array(res))
        print('after train evaluation:', params.prefix)
        mrr = []
        h10 = []
        for i in range(params.reptition):
            graph_classifier = whole_dgl_model(params, relation2id).to(device=params.device)
            state_d = torch.load(osp.join(params.root_path, "best_"+params.model_name+'_'+str(i)+".pth"), map_location=params.device)
            graph_classifier.load_state_dict(state_d['state_dict'])
            validator = EvaluatorVarLen(params, graph_classifier, test)
            res = validator.eval(params.eval_rep)
            mrr.append(res['mrr'])
            h10.append(res['h10'])
        mrr = np.array(mrr)
        h10 = np.array(h10)
        print(mrr, np.mean(mrr), np.std(mrr))
        print(h10, np.mean(h10), np.std(h10))
