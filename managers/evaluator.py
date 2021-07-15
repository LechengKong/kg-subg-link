import os
import numpy as np
import torch
import pdb
from sklearn import metrics
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from sklearn import metrics
import time


class Evaluator():
    def __init__(self, params, graph_classifier, data):
        self.params = params
        self.graph_classifier = graph_classifier
        self.data = data

    def eval(self, save=False):
        # print("eval")
        mrr_scores = []
        h10_scores = []
        if save:
            dataloader = DataLoader(self.data, batch_size=self.params.val_batch_size, num_workers=self.params.num_workers, collate_fn=self.params.collate_fn_val, prefetch_factor=self.params.prefetch_val, pin_memory=True)
            num_batches = len(dataloader)
            save_batch = int(num_batches/10)
            all_scores = np.zeros((int(save_batch*self.params.val_batch_size), 100))
            file_n = 0
            batch_count = 0
        else:
            dataloader = DataLoader(self.data, batch_size=self.params.val_batch_size, num_workers=self.params.num_workers, collate_fn=self.params.collate_fn_val, prefetch_factor=self.params.prefetch_val, pin_memory=True , sampler=RandomSampler(self.data, replacement=True, num_samples=self.params.val_size))
        # vs = time.time()
        self.graph_classifier.eval()
        with torch.no_grad():
            pbar = tqdm(dataloader)
            for b_num, batch in enumerate(pbar):
                # print("intertime", time.time()-vs)
                data_pos, targets_pos = self.params.move_batch_to_device_val(batch, self.params.device)
                # print("movetime", time.time()-vs)
                # print([self.data.id2relation[r.item()] for r in data_pos[1]])
                # pdb.set_trace()
                try:
                    score_pos = self.graph_classifier(data_pos)
                    scores = score_pos.view(len(targets_pos), -1)
                    scores = scores.cpu().numpy()
                    (b_size, dim_size) = scores.shape
                    if save:
                        all_scores[self.params.val_batch_size*batch_count:self.params.val_batch_size*(batch_count+1),:]= scores
                    else:
                        tp = targets_pos.cpu().numpy()

                        true_labels = np.zeros(scores.shape)
                        true_labels[np.arange(b_size), tp] = 1

                        ranking = np.argsort(scores, axis=1)
                        h10_ranking = ranking[:,10:]
                        true_ranking = np.sum(h10_ranking==tp[:,np.newaxis], axis=1)
                        h10_scores.append(np.mean(true_ranking))
                        mrr_scores.append(metrics.label_ranking_average_precision_score(true_labels, scores))
                except RuntimeError:
                    print("oom")
                    if save:
                        all_scores[self.params.val_batch_size*batch_count:self.params.val_batch_size*(batch_count+1),:]=np.arange(100)
                    torch.cuda.empty_cache()
                finally:
                    if save:
                        batch_count+=1
                        if batch_count==save_batch:
                            batch_count = 0
                            np.save("/project/tantra/jerry.kong/ogb_project/dataset/wikikg90m_kddcup2021/filter_pred_lg8/pred_"+str(file_n), all_scores)
                            all_scores = np.zeros((int(save_batch*self.params.val_batch_size), 100))
                            file_n+=1
                    continue
                # print("gputime", time.time()-vs)
                # print("calctime",time.time()-vs)
                # vs = time.time()

        # acc = metrics.accuracy_score(labels, preds)
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
            return {'mrr': mrr, 'h10': h10}
