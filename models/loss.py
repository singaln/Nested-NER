#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time    : 2022/1/24 11:31
# @Author  : SinGaln

import torch
import torch.nn as nn

def multilabel_categorical_crossentropy(y_pred, y_true):
    """
    详情见: https://kexue.fm/archives/7359
    y_true和y_pred的shape一致，y_true的元素非0即1，
    1表示对应的类为目标类，0表示对应的类为非目标类。
    """
    y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
    y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
    y_pred_pos = (y_pred - (1 - y_true) * 1e12)  # mask the pred outputs of neg classes
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

    return neg_loss + pos_loss

def loss_fun(y_true, y_pred):
    """
    :param y_true: [batch_size, num_labels, seq_len, seq_len]
    :param y_pred: [batch_size, num_labels, seq_len, seq_len]
    :return: loss scores
    """
    bh = y_pred.shape[0] * y_pred.shape[1]
    y_pred = torch.reshape(y_pred, (bh, -1))
    y_true = torch.reshape(y_true, (bh, -1))
    loss = torch.mean(multilabel_categorical_crossentropy(y_true, y_pred))
    return loss


class Span_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_func = torch.nn.CrossEntropyLoss(reduction="none")

    def forward(self,span_logits,span_label,seq_mask, num_label):
        # batch_size,seq_len,hidden=span_label.shape
        span_label = span_label.view(size=(-1,))
        span_logits = span_logits.view(size=(-1, num_label))
        span_loss = self.loss_func(input=span_logits, target=span_label)

        # start_extend = seq_mask.unsqueeze(2).expand(-1, -1, seq_len)
        # end_extend = seq_mask.unsqueeze(1).expand(-1, seq_len, -1)
        span_mask = seq_mask.view(size=(-1,))
        span_loss *=span_mask
        avg_se_loss = torch.sum(span_loss) / seq_mask.size()[0]
        # avg_se_loss = torch.sum(sum_loss) / bsz
        return avg_se_loss