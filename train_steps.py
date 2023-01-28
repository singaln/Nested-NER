#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time    : 2022/2/7 11:06
# @Author  : SinGaln

"""
starformer模型训练的step
"""
import torch
# from apex import amp
import torch.nn as nn
from utils import loss_fun
from bert_optimization import BertAdam

def set_optimizer(model, lr, train_steps=None):
    param_optimizer = list(model.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=lr,
                         warmup=0.1,
                         t_total=train_steps)
    return optimizer

# starformer的train_step部分
def starformer_steps(args, batch, model, lr, device, adam_epsilon, warmup_steps, train_dataloader, epochs, batch_size):
    model.train()
    optimizer = set_optimizer(model, lr, train_steps=(int(len(train_dataloader) / batch_size) + 1) * epochs)
    _, input_ids, attention_mask, token_type_ids, labels = batch
    input_ids, attention_mask, token_type_ids, labels = input_ids.to(device), attention_mask.to(
        device), token_type_ids.to(device), labels.to(device)
    logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                   entity_label_ids=labels)
    # pred, loss = outputs[0], outputs[1]
    loss = loss_fun(labels, logits)
    loss = loss.mean()
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.25)
    optimizer.step()
    # scheduler.step()
    return loss, logits, labels
