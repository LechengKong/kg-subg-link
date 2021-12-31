import os
import logging
import numpy as np
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

from sklearn import metrics

from util import SmartTimer


class Trainer():
    def __init__(self, params, graph_classifier, train, state_dict=None, valid_evaluator=None, label=0):
        self.graph_classifier = graph_classifier
        self.valid_evaluator = valid_evaluator
        self.params = params
        self.train_data = train
        self.label = label
        self.timer = SmartTimer(False)
        # self.g = self.train_data.graph.to(params.device)

        model_params = list(self.graph_classifier.parameters())
        logging.info('Total number of parameters: %d' % sum(map(lambda x: x.numel(), model_params)))
        # print(graph_classifier)

        if params.optimizer == "SGD":
            self.optimizer = optim.SGD(model_params, lr=params.lr, momentum=params.momentum, weight_decay=self.params.l2)
        if params.optimizer == "Adam":
            self.optimizer = optim.Adam(model_params, lr=params.lr, weight_decay=self.params.l2)

        self.updates_counter = 0
        self.start_epoch = 0
        self.epoch = 1
        if state_dict is not None:
            self.start_epoch = state_dict['epoch']
            self.optimizer.load_state_dict(state_dict['optimizer'])
            self.graph_classifier.load_state_dict(state_dict['state_dict'])

        self.criterion = nn.MarginRankingLoss(self.params.margin, reduction='sum')
        self.mscriterion = nn.MSELoss(reduction='sum')
        # self.mscriterion = nn.CrossEntropyLoss()

        self.reset_training_state()

    def reset_training_state(self):
        self.best_metric = 0
        self.last_metric = 0
        self.not_improved_count = 0
        self.should_train = True

    def train_epoch(self):
        total_loss = 0
        reg_loss = 0
        mrr_score = 0
        all_rankings = []
        all_labels = []
        all_scores = []
        result = None
        if self.params.regraph:
            self.train_data.resample()
            self.train_data.regraph()
        # dataloader = DataLoader(self.train_data, batch_size=self.params.batch_size,  num_workers=self.params.num_workers, collate_fn = self.params.collate_fn, shuffle=True)
        dataloader = DataLoader(self.train_data, batch_size=self.params.batch_size, num_workers=self.params.num_workers, collate_fn=self.params.collate_fn, sampler =RandomSampler(self.train_data, num_samples=10, replacement=True))
        self.graph_classifier.train()
        pbar = tqdm(dataloader)
        self.timer.record()
        # with torch.autograd.detect_anomaly():
        for batch in pbar:
            sp = batch.ls
            self.timer.cal_and_update('data')
            if self.params.use_random_labels and self.params.batch_random:
                self.train_data.re_label()
            self.timer.cal_and_update('relabel')
            data = self.params.move_batch_to_device(sp, self.params.device)
            self.timer.cal_and_update('move')
            self.optimizer.zero_grad()
            g = self.train_data.graph.to(self.params.device)
            g.edata['mask'][data[3]]=0
            scores, h_pred, t_pred, h_true, t_true = self.graph_classifier((g,(data[0],data[1],data[2],data[4])))
            self.timer.cal_and_update('model')
            scores_mat = scores.view(-1, self.params.train_neg_sample_size+1)
            # scores_mat = torch.clamp(scores_mat, -, 64)
            score_pos = scores_mat[:,0]
            score_neg = scores_mat[:,1:]
            # max_score_neg, _ = score_neg.max(dim=1)
            if self.params.label_reg:
                loss1 = self.criterion(score_pos.unsqueeze(1), score_neg, torch.Tensor([1]).to(device=self.params.device))
                loss2 = self.mscriterion(h_pred, h_true) + self.mscriterion(t_pred, t_true)
                loss = loss1+2*loss2
            else:
                # weight = torch.exp(0.3*score_neg)
                # weight = weight/weight.sum(dim=-1).unsqueeze(1)
                # loss = torch.mean(-torch.log(torch.sigmoid(score_pos)+0.000001)-(weight*torch.log(1-torch.sigmoid(score_neg)+0.000001)).sum(dim=-1))
                loss = self.criterion(score_pos.unsqueeze(1), score_neg, torch.Tensor([1]).to(device=self.params.device))
                # exp_score = torch.exp(0.5*score_neg)
                # assert torch.isnan(exp_score).sum().item() == 0
                # print(score_neg[1])
                # print(exp_score[1])
                # weight = exp_score/(exp_score.sum(dim=1)).unsqueeze(1)
                # assert torch.isnan(weight).sum().item() == 0
                # loss = (-torch.log(torch.sigmoid(score_pos))-(weight*torch.log(1-torch.sigmoid(score_neg))).sum(dim=1)).sum()
            # loss = self.mscriterion(h_pred, h_true.view(-1)) + self.mscriterion(t_pred, t_true.view(-1))
            loss.backward()
            self.optimizer.step()
            self.timer.cal_and_update('back')
            with torch.no_grad():
                score_mat = scores_mat.cpu().numpy()
                score_sort = np.argsort(score_mat, axis = 1)
                rankings = len(scores_mat[0])-np.where(score_sort==0)[1]
                all_rankings +=rankings.tolist()
                total_loss += loss.item()
                if self.params.label_reg:
                    reg_loss += loss2.item()
            self.timer.cal_and_update('mis')
                
        self.updates_counter += 1
        if self.valid_evaluator and self.params.eval_every_iter and self.updates_counter % self.params.eval_every_iter == 0:
            # print('should eval')
            tic = time.time()
            result = self.valid_evaluator.eval(self.params.eval_rep)
            print('\nPerformance:' + str(result) + 'in ' + str(time.time() - tic))

            if result['mrr'] >= self.best_metric:
                self.save_classifier()
                self.best_metric = result['mrr']
                self.not_improved_count = 0

            else:
                self.not_improved_count += 1
                if self.not_improved_count > self.params.early_stop:
                    logging.info(f"Validation performance didn\'t improve for {self.params.early_stop} epochs. Training stops.")
                    self.should_train = False
            self.last_metric = result['mrr']

        torch.cuda.empty_cache()
        return total_loss/self.params.train_edges, reg_loss/self.params.train_edges, np.mean(1/np.array(all_rankings)), np.mean(np.array(all_rankings)<=10), np.mean(np.array(all_rankings)==1), result

    def train(self):
        self.reset_training_state()
        all_metric = []
        for epoch in range(self.start_epoch+1, self.params.num_epochs + self.start_epoch + 1):
            self.epoch = epoch
            time_start = time.time()
            loss, reg_loss, mrr, h10, h1, eval_result = self.train_epoch()
            if eval_result is not None:
                all_metric.append([])
                all_metric[-1]+=[loss, reg_loss, mrr]
                for v in eval_result.values():
                    all_metric[-1].append(v)
            time_elapsed = time.time() - time_start
            print(f'Epoch {epoch} with loss: {loss}, mrr:{mrr}, h10:{h10}, h1:{h1}, reg_loss:{reg_loss}, best VAL mrr: {self.best_metric} in {time_elapsed}')
            if not self.should_train:
                break
            if epoch % self.params.save_every == 0 and self.params.save_res:
                torch.save({'epoch': self.epoch, 'state_dict': self.graph_classifier.state_dict(), 'optimizer': self.optimizer.state_dict()}, os.path.join(self.params.root_path, self.params.model_name+'_'+str(self.label)+'.pth'))
        return np.array(all_metric)
        # np.save('lstmmetric', np.array(all_metric))

    def save_classifier(self):
        if self.params.save_res:
            torch.save({'epoch': self.epoch, 'state_dict': self.graph_classifier.state_dict(), 'optimizer': self.optimizer.state_dict()}, os.path.join(self.params.root_path, 'best_'+self.params.model_name+'_'+str(self.label)+'.pth'))
            logging.info('Better models found w.r.t accuracy. Saved it!')
