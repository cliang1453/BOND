# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import json
import torch.nn.functional as F
import torch

logger = logging.getLogger(__name__)

def soft_frequency(logits, power=2, probs=False):
    """
    Unsupervised Deep Embedding for Clustering Analysis
    https://arxiv.org/abs/1511.06335
    """
    if not probs:
        softmax = torch.nn.Softmax(dim=1)
        y = softmax(logits.view(-1, logits.shape[-1])).view(logits.shape)
    else:
        y = logits
    f = torch.sum(y, dim=(0, 1))
    t = y**power / f
    p = t/torch.sum(t, dim=2, keepdim=True)

    return p

def multi_source_label_refine(args, hp_labels, combined_labels, pred_labels, pad_token_label_id, pred_logits=None):

    if args.self_training_label_mode == "hard":
        if args.self_training_hp_label == 0:
            pass
        elif args.self_training_hp_label == 1:
            pred_labels[combined_labels==pad_token_label_id] = pad_token_label_id
        elif args.self_training_hp_label == 2:
            pred_labels[combined_labels==pad_token_label_id] = pad_token_label_id
            pred_labels[hp_labels>0] = hp_labels[hp_labels>0]
        elif args.self_training_hp_label == 3:
            pred_labels[hp_labels>0] = hp_labels[hp_labels>0]
        elif 4 <= args.self_training_hp_label < 6:
            if pred_logits is None:
                raise RuntimeError('Please provide pred_logits')
            softmax = torch.nn.Softmax(dim=1)
            y = softmax(pred_logits.view(-1, pred_logits.shape[-1])).view(pred_logits.shape)
            _threshold = args.self_training_hp_label%1
            pred_labels[y.max(dim=-1)[0]>_threshold] = pad_token_label_id
            if args.self_training_hp_label < 5:
                pred_labels[combined_labels==pad_token_label_id] = pad_token_label_id
        elif 6 <= args.self_training_hp_label <= 7:
            _threshold = args.self_training_hp_label%1
            softmax = torch.nn.Softmax(dim=1)
            y = softmax(pred_logits.view(-1, pred_logits.shape[-1])).view(pred_logits.shape)
            _confidence = y.max(dim=-1)[0]
            pred_labels[_confidence< _threshold] = combined_labels[_confidence< _threshold] 
        else:
            raise NotImplementedError('error')

        return pred_labels, None
    elif args.self_training_label_mode == "soft":
        if args.self_training_hp_label == 0:
            label_mask = None
        elif args.self_training_hp_label == 1:
            label_mask = combined_labels!=pad_token_label_id
        elif args.self_training_hp_label in [2,3]:
            label_mask = combined_labels!=pad_token_label_id
            for i in range(1,pred_labels.shape[2]):
                _labeli = [0]*pred_labels.shape[2]
                _labeli[i] = 1
                _labeli = torch.tensor(_labeli).to(pred_labels)
                pred_labels[hp_labels==i] = _labeli
            if args.self_training_hp_label == 3:
                label_mask = None
        elif 4 <= args.self_training_hp_label < 6:
            _threshold = args.self_training_hp_label%1
            label_mask = (pred_labels.max(dim=-1)[0]>_threshold)
            if args.self_training_hp_label < 5:
                label_mask = label_mask & (combined_labels!=pad_token_label_id)
        elif 6 <= args.self_training_hp_label < 7:
            _threshold = args.self_training_hp_label%1
            _confidence = pred_labels.max(dim=-1)[0]
            for i in range(1,pred_labels.shape[0]):
                for j in range(1,pred_labels.shape[1]):
                    if _confidence[i,j] < _threshold:
                        _distantlabel = combined_labels[i,j]
                        pred_labels[i,j] *= 0
                        pred_labels[i,j,_distantlabel] = 1
        elif 7 <= args.self_training_hp_label < 9:
            _threshold = args.self_training_hp_label%1
            label_mask = (pred_labels.max(dim=-1)[0]>_threshold)
            if args.self_training_hp_label < 8:
                label_mask = label_mask & (combined_labels!=pad_token_label_id)
            # overwrite by hp_labels
            for i in range(0,pred_labels.shape[2]):
                _labeli = [0]*pred_labels.shape[2]
                _labeli[i] = 1
                _labeli = torch.tensor(_labeli).to(pred_labels)
                pred_labels[hp_labels==i] = _labeli
        else:
            raise NotImplementedError('error')

        return pred_labels, label_mask

def get_mt_loss(s_logits, t_logits, class_name, _lambda):
    
    if class_name is None:
        return 0
    s_logits = s_logits.view(-1, s_logits.size(-1)).float()
    t_logits = t_logits.view(-1, t_logits.size(-1)).float()
    if class_name == "prob":
        logprob_stu = F.log_softmax(s_logits, 1)
        logprob_tea = F.log_softmax(t_logits, 1)
        return F.mse_loss(logprob_tea.exp(),logprob_stu.exp())*_lambda
    elif class_name == "logit":
        return F.mse_loss(s_logits.view(-1),t_logits.view(-1))*_lambda
    elif class_name == "smart":
        prob_stu = F.log_softmax(s_logits, 1).exp()
        prob_tea = F.log_softmax(t_logits, 1).exp()
        r_stu = -(1/(prob_stu+1e-6)-1+1e-6).detach().log()
        r_tea = -(1/(prob_tea+1e-6)-1+1e-6).detach().log()
        return (prob_stu*(r_stu-r_tea)*2).mean()*_lambda
    elif class_name == 'kl':
        logprob_stu = F.log_softmax(s_logits, 1)
        prob_tea = F.log_softmax(t_logits, 1).exp()
        return -(prob_tea*logprob_stu).sum(-1).mean()*_lambda
    elif class_name == 'distill':
        temp = 2
        logprob_stu = F.log_softmax(s_logits/temp, 1)
        prob_tea = F.log_softmax(t_logits/temp, 1).exp()
        return -(prob_tea*logprob_stu).sum(-1).mean()*_lambda

def mt_update(t_params, s_params, average="exponential", alpha=0.995, step=None):

    for (t_name, t_param), (s_name, s_param) in zip(t_params, s_params):
        if t_name != s_name:
            logger.error("t_name != s_name: {} {}".format(t_name, s_name))
            raise ValueError
        param_new = s_param.data.to(t_param.device)
        if average == "exponential":
            t_param.data.add_( (1-alpha)*(param_new-t_param.data) )
        elif average == "simple":
            virtual_decay = 1 / float(step)
            diff = (param_new - t_param.data) * virtual_decay
            t_param.data.add_(diff)

def opt_grad(loss, in_var, optimizer):
    
    if hasattr(optimizer, 'scalar'):
        loss = loss * optimizer.scaler.loss_scale
    return torch.autograd.grad(loss, in_var)