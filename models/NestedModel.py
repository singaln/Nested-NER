#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time    : 2022/2/11 15:12
# @Author  : SinGaln
import math
import torch
from torch import nn
# from .bert import AutoBertModel, BertPreTrainedModel
from transformers import BertModel, BertPreTrainedModel

def sequence_masking(x, mask, value='-inf', axis=None):
    if mask is None:
        return x
    else:
        if value == '-inf':
            value = -1e12
        elif value == 'inf':
            value = 1e12
        assert axis > 0, 'axis must be greater than 0'
        for _ in range(axis - 1):
            mask = torch.unsqueeze(mask, 1)
        for _ in range(x.ndim - mask.ndim):
            mask = torch.unsqueeze(mask, mask.ndim)
        return x * mask + value * (1 - mask)


def add_mask_tril(logits, mask):
    if mask.dtype != logits.dtype:
        mask = mask.type(logits.dtype)
    logits = sequence_masking(logits, mask, '-inf', logits.ndim - 2)
    logits = sequence_masking(logits, mask, '-inf', logits.ndim - 1)
    # 排除下三角
    mask = torch.tril(torch.ones_like(logits), diagonal=-1)
    logits = logits - mask * 1e12
    return logits


class SinusoidalPositionEmbedding(nn.Module):
    """定义Sin-Cos位置Embedding
    """
    def __init__(
        self, output_dim, merge_mode='add', custom_position_ids=False):
        super(SinusoidalPositionEmbedding, self).__init__()
        self.output_dim = output_dim
        self.merge_mode = merge_mode
        self.custom_position_ids = custom_position_ids

    def forward(self, inputs):
        if self.custom_position_ids:
            seq_len = inputs.shape[1]
            inputs,position_ids = inputs
            position_ids = position_ids.type(torch.float)
        else:
            input_shape = inputs.shape
            batch_size, seq_len = input_shape[0], input_shape[1]
            position_ids = torch.arange(seq_len, dtype=torch.float)[None]
        indices = torch.arange(self.output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000.0, -2 * indices / self.output_dim)
        embeddings = torch.einsum('bn,d->bnd', position_ids, indices)
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = torch.reshape(embeddings, (-1, seq_len, self.output_dim))
        if self.merge_mode == 'add':
            return inputs + embeddings.to(inputs.device)
        elif self.merge_mode == 'mul':
            return inputs * (embeddings + 1.0).to(inputs.device)
        elif self.merge_mode == 'zero':
            return embeddings.to(inputs.device)


class Biaffine(nn.Module):
    def __init__(self, hidden_size, num_labels, bias_x=True, bias_y=True):
        super(Biaffine, self).__init__()
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.num_labels = num_labels
        self.weight = nn.Parameter(torch.Tensor(num_labels + self.bias_x, num_labels, num_labels + self.bias_y))
        self.start_layer = torch.nn.Sequential(torch.nn.Linear(in_features=hidden_size, out_features=num_labels),
                                               torch.nn.ReLU())
        self.end_layer = torch.nn.Sequential(torch.nn.Linear(in_features=hidden_size, out_features=num_labels),
                                             torch.nn.ReLU())

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, inputs, mask=None):
        hidden_size = inputs.shape[-1]
        # 位置编码
        inputs = SinusoidalPositionEmbedding(hidden_size, "zero")(inputs)
        start_logits = self.start_layer(inputs)
        end_logits = self.end_layer(inputs)
        if self.bias_x:
            start_logits = torch.cat((start_logits, torch.ones_like(start_logits[..., :1])), dim=-1)
        if self.bias_y:
            end_logits = torch.cat((end_logits, torch.ones_like(end_logits[..., :1])), dim=-1)
        # bxi,oij,byj->boxy
        # [batch_size, num_labels, seq_len, seq_len]
        logits = torch.einsum('bxi,ioj,byj->bxyo', start_logits, self.weight, end_logits)
        logits = logits.permute(0, 3, 1, 2)
        logits = add_mask_tril(logits, mask)
        return logits


class Models(BertPreTrainedModel):
    def __init__(self, config, num_labels, head_size, mode):
        super(Models, self).__init__(config)
        self.mode = mode
        self.head_size = head_size
        self.num_labels = num_labels

        self.bert = BertModel(config=config)
        self.dense = nn.Linear(config.hidden_size, self.num_labels * self.head_size * 2)
        if self.mode == "Efficient":
            self.head_dense = nn.Linear(config.hidden_size, self.head_size * 2)
            self.tial_dense = nn.Linear(config.hidden_size, self.num_labels * 2)
        if self.mode == "Biaffine":
            self.biaffine_model = Biaffine(config.hidden_size, self.num_labels)

    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        embeddings = embeddings.to("cuda")
        return embeddings

    def forward(self, input_ids, attention_mask, token_type_ids, entity_label_ids=None):
        context_outputs = self.bert(input_ids, attention_mask, token_type_ids)
        # last_hidden_state:(batch_size, seq_len, hidden_size)
        last_hidden_state = context_outputs.last_hidden_state
        if self.mode == "Biaffine":
            logits = self.biaffine_model(last_hidden_state, attention_mask)
            return logits
        batch_size = last_hidden_state.size()[0]
        seq_len = last_hidden_state.size()[1]
        if self.mode == "Global":
            # outputs:(batch_size, seq_len, num_labels*head_size*2)
            outputs = self.dense(last_hidden_state)
            outputs = torch.split(outputs, self.head_size * 2, dim=-1)
            # outputs:(batch_size, seq_len, num_labels, head_size*2)
            outputs = torch.stack(outputs, dim=-2)
            # pos_emb:(batch_size, seq_len, head_size)
            qw, kw = outputs[..., :self.head_size], outputs[..., self.head_size:]
            sinusoidal_position = self.sinusoidal_position_embedding(batch_size, seq_len, self.head_size)
            # cos_pos,sin_pos: (batch_size, seq_len, 1, head_size)
            cos_pos = sinusoidal_position[..., None, 1::2].repeat_interleave(2, dim=-1)
            sin_pos = sinusoidal_position[..., None, ::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
            qw2 = qw2.reshape(qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
            kw2 = kw2.reshape(kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos

            # logits:(batch_size, num_labels, seq_len, seq_len)
            logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)
            # padding mask
            pad_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.num_labels, seq_len, seq_len)
            logits = logits * pad_mask - (1 - pad_mask) * 1e12

            # 排除下三角
            mask = torch.tril(torch.ones_like(logits), -1)
            logits = logits - mask * 1e12

            return logits / self.head_size ** 0.5
        # qw,kw:(batch_size, seq_len, num_labels, head_size)
        if self.mode == "Efficient":
            outputs = self.head_dense(last_hidden_state)
            qw, kw = outputs[..., ::2], outputs[..., 1::2]
            # pos_emb:(batch_size, seq_len, head_size)
            sinusoidal_position = SinusoidalPositionEmbedding(self.head_size, "zero")(outputs)
            # cos_pos,sin_pos: (batch_size, seq_len, 1, head_size)
            cos_pos = sinusoidal_position[..., 1::2].repeat_interleave(2, dim=-1)
            sin_pos = sinusoidal_position[..., ::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], 3)
            qw2 = torch.reshape(qw2, qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], 3)
            kw2 = torch.reshape(kw2, kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos
            # 计算内积
            logits = torch.einsum('bmd,bnd->bmn', qw, kw) / self.head_size**0.5
            bias = torch.einsum('bnh->bhn', self.tial_dense(last_hidden_state)) / 2
            logits = logits[:, None] + bias[:, ::2, None] + bias[:, 1::2, :, None]
            # padding mask
            logits = add_mask_tril(logits, mask=attention_mask)
            return logits