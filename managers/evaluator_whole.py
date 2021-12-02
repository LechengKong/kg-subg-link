import os
from networkx.algorithms.link_prediction import preferential_attachment
import numpy as np
import torch
import pdb
from sklearn import metrics
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from sklearn import metrics
import time
from util import SmartTimer


class Evaluator():
    def __init__(self, params, graph_classifier, data):
        self.params = params
        self.graph_classifier = graph_classifier
        self.data = data
        # self.g = self.data.graph.to(params.device)

    def eval(self, save=False):
        # print("eval")
        mrr_scores = []
        h10_scores = []
        all_labels = []
        pre_scores = []
        h10_list = []
        if self.params.use_random_labels:
            self.data.re_label()
        if save:
            dataloader = DataLoader(self.data, batch_size=self.params.val_batch_size, num_workers=self.params.num_workers, collate_fn=self.params.collate_fn_val, prefetch_factor=self.params.prefetch_val, pin_memory=True)
            num_batches = len(dataloader)
            save_batch = int(num_batches/10)
            all_scores = np.zeros((int(save_batch*self.params.val_batch_size), 100))
            file_n = 0
            batch_count = 0
        else:
            # dataloader = DataLoader(self.data, batch_size=self.params.val_batch_size, num_workers=self.params.num_workers, collate_fn=self.params.collate_fn_val, prefetch_factor=self.params.prefetch_val, pin_memory=True , sampler=RandomSampler(self.data, replacement=True, num_samples=self.params.val_size))
            dataloader = DataLoader(self.data, batch_size=self.params.val_batch_size, num_workers=self.params.num_workers, collate_fn=self.params.collate_fn, prefetch_factor=self.params.prefetch_val, pin_memory=True )
        self.graph_classifier.eval()
        with torch.no_grad():
            g = self.data.graph.to(self.params.device)
            h = self.graph_classifier.graph_update(g)
            pbar = tqdm(dataloader)
            d_l = []
            v = time.time()
            for b_num, batch in enumerate(pbar):
                data = self.params.move_batch_to_device(batch, self.params.device)
                d_l.append(data[1][0].item())
                if self.params.only_link_sample:
                    # g = self.data.graph.to(self.params.device)
                    # score_pos = self.graph_classifier((g,data))
                    score_pos,_,_,_,_ = self.graph_classifier.mlp_update(g, data[0], data[1],data[2],h)
                else:
                    score_pos = self.graph_classifier(data)
                scores = score_pos.view(-1, self.params.val_neg_sample_size+1)
                scores = scores.cpu().numpy()
                (b_size, dim_size) = scores.shape
                if save:
                    all_scores[self.params.val_batch_size*batch_count:self.params.val_batch_size*(batch_count+1),:]= scores
                else:
                    tp = np.zeros(b_size, dtype=int)
                    neg_link_ind = (tp+1)%dim_size

                    true_labels = np.zeros(scores.shape)
                    true_labels[np.arange(b_size), tp] = 1

                    ranking = np.argsort(scores, axis=1)
                    h10_ranking = ranking[:,-10:]
                    true_ranking = np.sum(h10_ranking==tp[:,np.newaxis], axis=1)
                    h10_list += true_ranking.tolist()
                    h10_scores.append(np.mean(true_ranking))
                    mrr_scores.append(metrics.label_ranking_average_precision_score(true_labels, scores))
                    all_labels += true_labels[np.arange(b_size), tp].tolist()
                    all_labels += true_labels[np.arange(b_size), neg_link_ind].tolist()
                    pre_scores += scores[np.arange(b_size), tp].tolist()
                    pre_scores += scores[np.arange(b_size), neg_link_ind].tolist()
        if save:
            np.save("/project/tantra/jerry.kong/ogb_project/dataset/wikikg90m_kddcup2021/filter_pred_lg8/pred_"+str(file_n), all_scores)
            return all_scores
        else:
            mrr = 0
            for v in mrr_scores:
                mrr += v
            if len(mrr_scores)!=0:
                mrr /= len(mrr_scores)
            h10 = 0
            for v in h10_scores:
                h10+=v
            if len(h10_scores)!=0:
                h10 /= len(h10_scores)
            auc_pr = metrics.average_precision_score(all_labels, pre_scores)
            np.save('dist', np.array(d_l))
            if self.params.return_pred_res:
                return {'mrr': mrr, 'h10': h10, 'apr':auc_pr, 'h10l': np.array(h10_list)}
            else:
                return {'mrr': mrr, 'h10': h10, 'apr':auc_pr}


class EvaluatorVarLen():
    def __init__(self, params, graph_classifier, data_list):
        self.params = params
        self.graph_classifier = graph_classifier
        self.data_list = data_list
        self.timer = SmartTimer(False)
        # self.g = self.data.graph.to(params.device)

    def eval_data(self, data, rep=10, save=False):
        # print("eval")
        if self.params.use_random_labels:
            data.re_label()
        dataloader = DataLoader(data, batch_size=self.params.val_batch_size, num_workers=self.params.num_workers, collate_fn=self.params.collate_fn, prefetch_factor=2, pin_memory=True)
        # dataloader = DataLoader(self.data, batch_size=self.params.val_batch_size, num_workers=self.params.num_workers, collate_fn=self.params.collate_fn, sampler =RandomSampler(self.data, replacement=True,num_samples=10))
        # sampler = RandomSampler(self.data, num_samples=10)
        self.graph_classifier.eval()
        with torch.no_grad():
            g = data.graph.to(self.params.device)
            h = self.graph_classifier.graph_update(g)
            pbar = tqdm(dataloader)
            d_l = []
            h10list = []
            mrrlist = []
            for i in range(rep):
                all_rankings = []
                self.timer.record()
                for b_num, batch in enumerate(pbar):
                    self.timer.cal_and_update('data')
                    sp = batch.ls
                    data = self.params.move_batch_to_device(sp, self.params.device)
                    self.timer.cal_and_update('move')
                    d_l.append(data[1].cpu().numpy())
                    score_pos,h_pred, t_pred, h_true, t_true = self.graph_classifier.mlp_update(g, data[0], data[1],data[2],h)
                    self.timer.cal_and_update('model')
                    scores = score_pos.cpu().numpy().flatten()
                    self.timer.cal_and_update('scdetach')
                    batch_len = data[4].cpu().numpy()
                    self.timer.cal_and_update('detach')
                    cur_head_pointer = 0
                    for bl in batch_len:
                        next_hp = cur_head_pointer+bl
                        b_score = scores[cur_head_pointer:next_hp]
                        cur_head_pointer = next_hp
                        order = np.argsort(b_score)
                        all_rankings.append(len(order)-np.where(order==0)[0][0])
                    self.timer.cal_and_update('mis')
                all_rankings = np.array(all_rankings)
                # print(all_rankings)
                np.save('ark',all_rankings)
                h10 = np.mean(all_rankings<=10)
                mrr = np.mean(1/all_rankings)
                # np.save('dist', np.array(d_l))
                h10list.append(h10)
                mrrlist.append(mrr)
            # print(h10list)
            # print(mrrlist)
            if self.params.return_pred_res:
                return {'mrr': mrr, 'h10': h10, 'apr':0, 'h10l': (all_rankings<=10).astype(int)}
            else:
                return {'mrr': np.array(mrrlist).mean(), 'h10': np.array(h10list).mean(), 'apr':0}
                

    def eval(self, rep=10, save=False):
        rescollect = {}
        for data in self.data_list:
            res = self.eval_data(data, rep, save)
            for k in res:
                if k not in rescollect:
                    rescollect[k]=0
                rescollect[k]+=res[k]
        for k in rescollect:
            rescollect[k] = rescollect[k]/len(self.data_list)
        return rescollect
