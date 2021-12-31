from dgl.transform import reverse
import torch
import random
import argparse
import numpy as np
import os.path as osp
import logging
import dgl
import scipy.io as io

from data_util import process_files
from datasets import SubgraphDatasetWhole, SubgraphDatasetOnlyLink, MultiSampleDataset, FullGraphDataset
from torch_util import set_data_passing
from model.dgl.graph_classifier import GraphClassifier as dgl_model
from model.dgl.graph_classifier import GraphClassifierMulti as multi_dgl_model
from model.dgl.graph_classifier import GraphClassifierWhole as single_dgl_model
from managers.trainer_whole import Trainer as TrainerHeterogeneous
from managers.trainer_homogeneous import Trainer as TrainerHomogeneous
from managers.evaluator_whole import EvaluatorVarLen, EvaluatorVarLenHomogeneous
from graph_util import *
from scipy.sparse import csr_matrix, tril


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='gnn')

    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--root_path", type=str, default="/project/tantra/jerry.kong/ogb_project/ogb-grail-mod/data")
    parser.add_argument("--log_path", type=str, default="./log")
    parser.add_argument("--data_set", type=str, default="fb237_v1")
    parser.add_argument("--ind_data_set", type=str, default="fb237_v1_ind")
    parser.add_argument("--transductive", type=bool, default=False)
    parser.add_argument("--homogeneous", type=bool, default=False)

    parser.add_argument("--use_multi_model", type=bool, default=False)
    parser.add_argument("--rel_emb_dim", type=int, default=32)
    parser.add_argument("--emb_dim", type=int, default=32)
    parser.add_argument("--attn_rel_emb_dim", type=int, default=32)
    parser.add_argument("--lstm_hidden_size", type=int, default=32)
    parser.add_argument("--num_gcn_layers", type=int, default=6)
    parser.add_argument("--num_bases",type=int, default=2)
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

    parser.add_argument("--num_workers",type=int, default=8)
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
    parser.add_argument("--edge_rep", type=int, default=4)
    parser.add_argument("--edge_split", type=float, default=2)
    parser.add_argument("--edges_division", type=int, default=10)

    parser.add_argument("--retrain", type=bool, default=False)
    parser.add_argument("--retrain_seed", type=int, default=100)
    parser.add_argument("--reptition", type=int, default=5)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--eval", type=bool, default=False)
    parser.add_argument("--eval_rep", type=int, default=10)

    parser.add_argument("--shortest_path_dist", type=int, default=15)
    parser.add_argument("--use_deep_set", type=bool, default=False)
    parser.add_argument("--deep_set_dim", type=int, default=1024)
    parser.add_argument("--concat_init_feat", type=bool, default=False)
    parser.add_argument("--add_ht_emb", type=bool, default=True)
    parser.add_argument("--use_random_labels", type=bool, default=False)
    parser.add_argument("--batch_random", type=bool, default=False)
    parser.add_argument("--label_reg", type=bool, default=False)
    parser.add_argument("--path_add_head", type=bool, default=False)
    parser.add_argument("--use_lstm", type=bool, default=False)
    parser.add_argument("--use_mid_repr", type=bool, default=True)
    parser.add_argument("--use_dist_emb", type=bool, default=False)

    parser.add_argument("--gpuid", type=int, default=0)
    params = parser.parse_args()
    params.max_label_value=10
    params.prefix = ("lstm_" if params.use_lstm else "")+("deepset_" if params.use_deep_set else "")+("multi_"+str(params.edge_split)+"_" if params.use_multi_model else "") +params.ind_data_set+"_"
    params.model_name = params.prefix+"gnn"
    print(params.prefix)
    if params.retrain:
        print("retrain model")
        torch.manual_seed(params.retrain_seed)
        random.seed(params.retrain_seed)
        np.random.seed(params.retrain_seed)
    elif not params.eval or params.homogeneous:
        torch.manual_seed(101)
        random.seed(101)
        np.random.seed(101)
    # else:
    #     torch.manual_seed(100)
    #     random.seed(100)
    #     np.random.seed(100)

    if torch.cuda.is_available():
        params.device = torch.device('cuda:'+str(params.gpuid))
    else:
        params.device = torch.device('cpu')

    logging.basicConfig(level=logging.INFO)

    if params.homogeneous:
        if params.data_set == 'celegan':
            data = np.genfromtxt(osp.join(params.data_path,'celegan.txt'), delimiter=',')
            data = data.astype(int)
            train_num_entities = np.max(data)+1
            ind_num_entities = train_num_entities
            head, tail = data[:,0], data[:,1]
        elif params.data_set == 'pb':
            data = np.genfromtxt(osp.join(params.data_path,'pb.txt'), delimiter=',')
            data = data.astype(int)
            train_num_entities = np.max(data)+1
            ind_num_entities = train_num_entities
            head, tail = data[:,0], data[:,1]
        else:
            Amat = io.loadmat(osp.join(params.data_path,params.data_set+'.mat'))['net']
            train_num_entities = Amat.shape[0]
            ind_num_entities = train_num_entities
            edge_mat = tril(Amat)
            head, tail = edge_mat.nonzero()
        train_num_rel = 1
        ind_num_rel = 1
        relation2id = None
        k = np.ones((train_num_entities,train_num_entities))
        k[head,tail]=0
        k[tail,head]=0
        nh,nt = k.nonzero()
        neg_perm = np.random.permutation(len(nt))
        perm = np.random.permutation(len(head))
        train_ind = int(len(perm)*0.8)
        test_ind = int(len(perm)*0.9)
        new_mat = np.zeros((len(head),3),dtype=int)
        new_mat[:,0] = head
        new_mat[:,2] = tail
        neg_mat = np.zeros((len(head),3), dtype=int)
        neg_mat[:,0] = nh[neg_perm[:len(head)]]
        neg_mat[:,2] = nt[neg_perm[:len(head)]]
        converted_triplets = {"train":new_mat[perm[:train_ind]], "train_neg":neg_mat[perm[:train_ind]], "test":new_mat[perm[train_ind:test_ind]],"test_neg":neg_mat[perm[train_ind:test_ind]], "valid":new_mat[perm[test_ind:]], "valid_neg":neg_mat[perm[test_ind:]]}
        converted_triplets_ind = converted_triplets
        rel_mat = converted_triplets['train']
        adj_list = [csr_matrix((np.ones(len(rel_mat)),(rel_mat[:,0],rel_mat[:,1])), shape=(train_num_entities,train_num_entities))]
        adj_list_ind = adj_list
        params.num_rels = train_num_rel
        params.aug_num_rels = train_num_rel
    else:
        database_path = osp.join(params.data_path, params.data_set)
        files = {}
        use_data = ["train", "test", "valid"]
        for f in use_data:
            files[f] = osp.join(database_path,f'{f}.txt')

        adj_list, converted_triplets, entity2id, relation2id, id2entity, id2relation = process_files(files)
        # print(relation2id)
        train_num_entities = len(entity2id)
        train_num_rel = len(relation2id)

        ind_database_path = osp.join(params.data_path, params.ind_data_set)
        files = {}
        for f in use_data:
            files[f] = osp.join(ind_database_path,f'{f}.txt')

        adj_list_ind, converted_triplets_ind, entity2id_ind, relation2id_ind, id2entity_ind, id2relation_ind = process_files(files, relation2id=relation2id)
        
        ind_num_entities = len(entity2id_ind)
        ind_num_rel = len(relation2id_ind)
        params.num_rels = train_num_rel
        params.aug_num_rels = train_num_rel*2
    print(f'Dataset {params.data_set} has {train_num_entities} entities and {train_num_rel} relations')
    print(f'Dataset {params.ind_data_set} has {ind_num_entities} entities and {ind_num_rel} relations')
    set_data_passing(params)
    torch.multiprocessing.set_sharing_strategy('file_system')

    if params.homogeneous:
        TrainSet = SubgraphDatasetOnlyLink
        TestSet = SubgraphDatasetOnlyLink
    else:
        TrainSet = FullGraphDataset
        TestSet = FullGraphDataset

    # train = TrainSet(converted_triplets, 'train', params, adj_list, train_num_rel, train_num_entities, ratio=params.sample_graph_ratio, neg_link_per_sample=params.train_neg_sample_size)
    # if not params.transductive:
    #     t_data = converted_triplets['train']
    #     perm = np.random.permutation(len(t_data))
    #     train_ind = int(len(perm)*0.7)
    #     train_triplets = {}
    #     train_triplets['valid'] = converted_triplets['valid']
    #     train_triplets['test'] = converted_triplets['test']
    #     train_triplets['train'] = t_data[perm[:train_ind]]
    #     val_triplets = {}
    #     val_triplets['valid'] = converted_triplets['valid']
    #     val_triplets['test'] = converted_triplets['test']
    #     val_triplets['train'] = t_data[perm[train_ind:]]
    # else:
    train_triplets = converted_triplets
    val_triplets = converted_triplets
    t_data = train_triplets['train']
    perm = np.random.permutation(len(t_data))
    train_ind = int(len(perm)*0.95)
    g_edge = t_data[perm[:train_ind]]
    t_edge = t_data[perm[train_ind:]]
    p_alledge = {}
    p_alledge['train']=g_edge
    p_alledge['valid']=t_edge
    if not params.homogeneous:
        train = TrainSet(train_triplets, 'train', params, adj_list, train_num_rel, train_num_entities, neg_link_per_sample=params.train_neg_sample_size, ratio=params.sample_graph_ratio)
    else:
        train = TrainSet(train_triplets, 'train', params, adj_list, train_num_rel, train_num_entities, neg_link_per_sample=params.train_neg_sample_size,sample_method=sample_arb_neg, mode='train')
        params.train_neg_sample_size = 1
    if params.transductive:
        if params.homogeneous:
            test =[TestSet(converted_triplets_ind, 'test', params, adj_list_ind, ind_num_rel, ind_num_entities, neg_link_per_sample=params.val_neg_sample_size, sample_method=sample_arb_neg)]
            val = [TestSet(converted_triplets_ind, 'valid', params, adj_list_ind, ind_num_rel, ind_num_entities, neg_link_per_sample=params.val_neg_sample_size, sample_method=sample_arb_neg)]
            inval = [TestSet(converted_triplets_ind, 'train', params, adj_list_ind, train_num_rel, train_num_entities, neg_link_per_sample=params.val_neg_sample_size, sample_method=sample_arb_neg)]
            # inval = [TestSet(p_alledge, 'valid', params, adj_list, train_num_rel, train_num_entities, neg_link_per_sample=params.val_neg_sample_size, sample_method=sample_arb_neg)]
        else:
            test =[TestSet(converted_triplets_ind, 'test', params, adj_list_ind, ind_num_rel, ind_num_entities, neg_link_per_sample=params.val_neg_sample_size, sample_method=sample_filtered_neg_tail, mode='valid'), TestSet(converted_triplets_ind, 'test', params, adj_list_ind, ind_num_rel, ind_num_entities, neg_link_per_sample=params.val_neg_sample_size, sample_method=sample_filtered_neg_head, mode='valid')]
            val = [TestSet(converted_triplets_ind, 'valid', params, adj_list_ind, ind_num_rel, ind_num_entities, neg_link_per_sample=params.val_neg_sample_size, sample_method=sample_filtered_neg_tail, mode='valid'),TestSet(converted_triplets_ind, 'valid', params, adj_list_ind, ind_num_rel, ind_num_entities, neg_link_per_sample=params.val_neg_sample_size, sample_method=sample_filtered_neg_head, mode='valid')]
            inval = [TestSet(p_alledge, 'valid', params, adj_list, train_num_rel, train_num_entities, neg_link_per_sample=params.val_neg_sample_size, sample_method=sample_filtered_neg_tail, mode='valid'),TestSet(p_alledge, 'valid', params, adj_list, train_num_rel, train_num_entities, neg_link_per_sample=params.val_neg_sample_size, sample_method=sample_filtered_neg_head, mode='valid')]
    else:
        test = [TestSet(converted_triplets_ind, 'test', params, adj_list_ind, ind_num_rel, ind_num_entities, neg_link_per_sample=params.val_neg_sample_size, sample_method=sample_neg_link, mode='valid')]
        # test = [TestSet(converted_triplets, 'test', params, adj_list, train_num_rel, train_num_entities, neg_link_per_sample=params.val_neg_sample_size, sample_method=sample_neg_link, mode='valid')]
        val = [TestSet(val_triplets, 'valid', params, adj_list, train_num_rel, train_num_entities, neg_link_per_sample=params.val_neg_sample_size, sample_method=sample_neg_link, mode='valid')]
        inval = [TestSet(p_alledge, 'valid', params, adj_list, train_num_rel, train_num_entities, neg_link_per_sample=params.val_neg_sample_size, sample_method=sample_neg_link, mode='valid')]
    params.train_edges = len(train)
    params.val_size = len(val[0])
    print(f'Training set has {params.train_edges} edges, Val set has {params.val_size} edges')
    params.inp_dim = train.n_feat_dim

    state_d = None
    if params.retrain:
        state_d = torch.load(osp.join(params.root_path, params.model_name+"_0.pth"), map_location=params.device)

    if params.use_multi_model:
        whole_dgl_model = multi_dgl_model
    else:
        whole_dgl_model = single_dgl_model

    if params.homogeneous:
        Trainer = TrainerHomogeneous
        Evaluator = EvaluatorVarLenHomogeneous
    else:
        Trainer = TrainerHeterogeneous
        Evaluator = EvaluatorVarLen

    if params.eval:
        eval_metric = ['mrr','h10','h1','auc','ap']
        res_col = {}
        in_res_col = {}
        for n in eval_metric:
            res_col[n] = []
            in_res_col[n] = []
        for i in range(params.reptition):
            graph_classifier = whole_dgl_model(params, relation2id).to(device=params.device)
            state_d = torch.load(osp.join(params.root_path, "best_"+params.model_name+'_'+str(i)+".pth"), map_location=params.device)
            graph_classifier.load_state_dict(state_d['state_dict'])
            validator = Evaluator(params, graph_classifier, test)
            # invalidator = EvaluatorVarLen(params, graph_classifier, inval)
            res = validator.eval(params.eval_rep)
            # inres = invalidator.eval(params.eval_rep)
            for n in eval_metric:
                res_col[n].append(res[n])
                # in_res_col[n].append(inres[n])
        for n in eval_metric:
            f = np.array(res_col[n])
            print(n, f, np.mean(f), np.std(f))
        # for n in eval_metric:
        #     f = np.array(in_res_col[n])
        #     print('in',n, f, np.mean(f), np.std(f))
    else:
        val_list = []
        train_list = []
        res = []
        for i in range(params.reptition):
            graph_classifier = whole_dgl_model(params, relation2id).to(device=params.device)
            validator = Evaluator(params, graph_classifier, val)
            trainer = Trainer(params, graph_classifier, train, state_dict=state_d, valid_evaluator=validator, label=i)
            res.append(trainer.train())
            val_list.append(validator)
            train_list.append(trainer)
        if params.save_res:
            np.save(params.log_path+'/'+params.prefix+'allres', np.array(res))
        print('after train evaluation:', params.prefix)
        mrr = []
        h10 = []
        h1 = []
        inmrr = []
        inh10 = []
        inh1 = []
        for i in range(params.reptition):
            graph_classifier = whole_dgl_model(params, relation2id).to(device=params.device)
            state_d = torch.load(osp.join(params.root_path, "best_"+params.model_name+'_'+str(i)+".pth"), map_location=params.device)
            graph_classifier.load_state_dict(state_d['state_dict'])
            validator = Evaluator(params, graph_classifier, test)
            invalidator = Evaluator(params, graph_classifier, inval)
            res = validator.eval(params.eval_rep)
            inres = invalidator.eval(params.eval_rep)
            mrr.append(res['mrr'])
            h10.append(res['h10'])
            h1.append(res['h1'])
            inmrr.append(inres['mrr'])
            inh10.append(inres['h10'])
            inh1.append(inres['h1'])
        res_col = [mrr,h10,h1,inmrr,inh10,inh1]
        for d in res_col:
            f = np.array(d)
            print(f, np.mean(f), np.std(f))
