#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time    : 2022/2/22 13:46
# @Author  : SinGaln
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from .bert import AutoBertModel

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

        self.dropout = nn.Dropout(0.2)
        self.out = nn.Linear(hidden, self.all_head_size)

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
        hidden_states = self.dropout(self.out(context_layer))
        hidden_states = hidden_states + query
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
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

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

    def forward(self, hidden_state: torch.Tensor):
        # 获取初始的hidden_states(也就是随机初始化的词向量表示)
        # hidden_states [batch_size, seq_length, hidden_dim]
        batch_size, seq_length, hidden_dim = hidden_state.size()
        # 当前节点的states
        current_states = hidden_state.clone()
        center = F.avg_pool2d(current_states, (current_states.shape[1], 1)).squeeze(1)
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
    def __init__(self, config, num_labels, head_size, mode, vocab_size):
        super(StarformerNER, self).__init__()
        self.mode = mode
        self.head_size = head_size
        self.num_labels = num_labels
        self.embed = nn.Embedding(vocab_size, self.num_labels * self.head_size)
        self.bert = AutoBertModel.from_pretrained("./AutoTinyBert")
        self.embed.weight.requires_grad = True # 设置梯度信息为True
        self.starformer = StarTransformerLayer(self.num_labels * self.head_size, self.num_labels, self.head_size)
        self.dense = nn.Linear(432, self.num_labels * self.head_size)
        self.head_dense = nn.Linear(self.num_labels * self.head_size, self.head_size * 2)
        self.tial_dense = nn.Linear(self.head_size * 2, self.num_labels * 2)

    # def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
    #     position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)
    #
    #     indices = torch.arange(0, output_dim // 2, dtype=torch.float)
    #     indices = torch.pow(10000, -2 * indices / output_dim)
    #     embeddings = position_ids * indices
    #     embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
    #     embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))
    #     embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
    #     embeddings = embeddings.to("cpu")
    #     return embeddings
    def sequence_masking(self, x, mask, value='-inf', axis=None):
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

    def add_mask_tril(self, logits, mask):
        if mask.dtype != logits.dtype:
            mask = mask.type(logits.dtype)
        logits = self.sequence_masking(logits, mask, '-inf', logits.ndim - 2)
        logits = self.sequence_masking(logits, mask, '-inf', logits.ndim - 1)
        # 排除下三角
        mask = torch.tril(torch.ones_like(logits), diagonal=-1)
        logits = logits - mask * 1e12
        return logits

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, entity_label_ids=None):
        # input_ids = self.embed(input_ids)
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).last_hidden_state
        # last_hidden_state:(batch_size, seq_len, hidden_size)
        # qw,kw:(batch_size, seq_len, num_labels, head_size)
        output = self.dense(output)
        context_outputs, _ = self.starformer(output)
        outputs = self.head_dense(context_outputs)
        qw, kw = outputs[..., ::2], outputs[..., 1::2]
        # pos_emb:(batch_size, seq_len, head_size)
        sinusoidal_position = SinusoidalPositionEmbedding(self.head_size, 'zero')(outputs)
        # cos_pos,sin_pos: (batch_size, seq_len, 1, head_size)
        cos_pos = sinusoidal_position[..., 1::2].repeat_interleave(2, dim=-1)
        sin_pos = sinusoidal_position[..., ::2].repeat_interleave(2, dim=-1)
        qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
        qw2 = qw2.reshape(qw.shape)
        qw = qw * cos_pos + qw2 * sin_pos
        kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
        kw2 = kw2.reshape(kw.shape)
        kw = kw * cos_pos + kw2 * sin_pos
        # 计算内积
        logits = torch.einsum('bmd,bnd->bmn', qw, kw) / self.head_size ** 0.5
        bias = torch.einsum('bnh->bhn', self.tial_dense(outputs)) / 2
        logits = logits[:, None] + bias[:, ::2, None] + bias[:, 1::2, :, None]
        logits = self.add_mask_tril(logits, mask=attention_mask)
        return logits