#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time    : 2022/2/8 11:34
# @Author  : SinGaln

import copy
import math
import torch
from torch import nn
from .bert import AutoBertModel
import torch.nn.functional as F


class MultiHeadsAttention(nn.Module):
    def __init__(self, hidden, num_labels, attention_head_size):
        super(MultiHeadsAttention, self).__init__()
        self.hidden = hidden
        self.num_attention_heads = num_labels
        self.attention_head_size = attention_head_size
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(hidden, self.all_head_size)
        self.key = nn.Linear(hidden, self.all_head_size)
        self.value = nn.Linear(hidden, self.all_head_size)
        # 初始化
        nn.init.xavier_normal_(self.query.weight)
        nn.init.xavier_normal_(self.key.weight)
        nn.init.xavier_normal_(self.value.weight)

        self.dropout = nn.Dropout(0.5)
        self.out = nn.Linear(self.all_head_size, self.all_head_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, query, key, value):
        mixed_query_layer = self.query(query)
        mixed_key_layer = self.key(key)
        mixed_value_layer = self.value(value)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.out(context_layer)
        return hidden_states


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        # print("x", x.size())
        # print(self.weight.size())
        # print(self.bias.size())
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class StarTransformerLayer(nn.Module):
    def __init__(self, hidden, num_attention_heads, attention_head_size):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = attention_head_size
        self.multi_att_satellite = MultiHeadsAttention(hidden, num_attention_heads, attention_head_size)
        self.multi_att_relay = copy.deepcopy(self.multi_att_satellite)
        self.ln_satellite = LayerNorm(self.num_attention_heads * self.attention_head_size)
        self.ln_relay = copy.deepcopy(self.ln_satellite)

    def cycle_shift(self, hidden_state: torch.Tensor, forward=True):
        batch_size, seq_length, hidden_dim = hidden_state.size()
        if forward:
            temp = hidden_state[:, -1, :]
            for i in range(seq_length - 1):
                hidden_state[:, i + 1, :] = hidden_state[:, i, :]
            hidden_state[:, 0, :] = temp
        else:
            temp = hidden_state[:, 0, :]
            for i in range(1, seq_length):
                hidden_state[:, i - 1, :] = hidden_state[:, i, :]
            hidden_state[:, -1, :] = temp

        return hidden_state

    def forward(self, hidden_state: torch.Tensor, center: torch.Tensor):
        # 获取初始的hidden_states(也就是随机初始化的词向量表示)
        # hidden_states [batch_size, seq_length, hidden_dim]
        batch_size, seq_length, hidden_dim = hidden_state.size()
        # 当前节点的states
        current_states = hidden_state.clone()
        # center = F.avg_pool2d(current_states, (current_states.shape[1], 1)).squeeze(1)
        for _ in range(2):
            # 获取当前节点的前一个和后一个节点的states
            current_states_before, current_states_after = self.cycle_shift(current_states.clone(), False), self.cycle_shift(current_states.clone(), True)
            center_exp = center.unsqueeze(1).expand_as(current_states) # [batch_size, 1, hidden_dim]
            concats = torch.cat(
                [current_states_before.unsqueeze(-2), current_states.unsqueeze(-2), current_states_after.unsqueeze(-2), hidden_state.unsqueeze(-2), center_exp.unsqueeze(-2)],
                dim=-2)
            concats = concats.view(batch_size * seq_length, -1, hidden_dim)
            current_states = current_states.unsqueeze(-2).view(batch_size * seq_length, -1, hidden_dim)
            # 更新卫星节点
            current_states = self.ln_satellite(F.relu(self.multi_att_satellite(current_states, concats, concats))).squeeze(-2).view(batch_size, seq_length, -1)
            # 更新中间节点
            center = center.unsqueeze(1) # [batch_size, 1, 1, hidden_dim]
            center_concat = torch.cat([center, current_states], dim=1)
            center = self.ln_relay(F.relu(self.multi_att_relay(center, center_concat, center_concat))).squeeze(1)
        return current_states, center


class StarformerNER(nn.Module):
    def __init__(self, config, num_labels, attention_head_size, vocab_size, use_bert=False, use_starformer=True):
        super(StarformerNER, self).__init__()
        self.use_bert = use_bert
        self.num_labels = num_labels
        self.use_starformer = use_starformer
        self.attention_head_size = attention_head_size
        if use_bert:
            self.bert = AutoBertModel(config)
        else:
            self.embedding = nn.Embedding(vocab_size, self.num_labels * self.attention_head_size)
        self.staformer = StarTransformerLayer(hidden=192,
                                              num_attention_heads=num_labels, attention_head_size=attention_head_size)
        # self.encoders = nn.ModuleList([copy.deepcopy(self.staformer) for _ in range(5)])
        self.dense = nn.Linear(config.hidden_size, self.num_labels * self.attention_head_size)
        self.linear = nn.Linear(config.hidden_size, self.num_labels * self.attention_head_size * 2)

    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim, device):
        position_ids = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(-1)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float, device=device)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((batch_size, *([1]*len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        embeddings = embeddings.to(device)
        return embeddings

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, entity_label_ids=None, biaffine=True):
        # hidden_states
        global context_outputs
        if self.use_bert:
            hidden_states = self.bert.forward(input_ids=input_ids, attention_mask=attention_mask,
                                              token_type_ids=token_type_ids)
            if self.use_starformer:
                center = self.dense(hidden_states[0][:, -1, :])
                for encoder in self.encoders:
                    context_outputs, _ = encoder(hidden_states[0], center)
            else:
                context_outputs = hidden_states[0]
        else:
            hidden_states = self.embedding(input_ids)
            center = self.dense(hidden_states[:, -1, :])
            # for encoder in self.encoders:
            context_outputs, _ = self.staformer(hidden_states, center)
        batch_size, seq_len = context_outputs.size(0), context_outputs.size(1)
        outputs = self.linear(context_outputs)
        outputs = torch.split(outputs, self.attention_head_size * 2, dim=-1)
        outputs = torch.stack(outputs, dim=-2)
        qw, kw = outputs[..., :self.attention_head_size], outputs[..., self.attention_head_size:]

        sinusoidal_positions = self.sinusoidal_position_embedding(batch_size, seq_len, self.attention_head_size, outputs.device)

        cos_pos = sinusoidal_positions[..., None, 1::2].repeat_interleave(2, dim=-1)
        sin_pos = sinusoidal_positions[..., None, ::2].repeat_interleave(2, dim=-1)
        qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
        qw2 = qw2.reshape(qw.shape)
        qw = qw * cos_pos + qw2 * sin_pos
        kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
        kw2 = kw2.reshape(kw.shape)
        kw = kw * cos_pos + kw2 * sin_pos
        # logits:(batch_size, ent_type_size, seq_len, seq_len)
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)
        # padding mask
        pad_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.num_labels, seq_len, seq_len)
        logits = logits * pad_mask - (1 - pad_mask) * 1e12

        # 排除下三角
        mask = torch.tril(torch.ones_like(logits), -1)
        logits = logits - mask * 1e12

        return logits / self.attention_head_size ** 0.5
