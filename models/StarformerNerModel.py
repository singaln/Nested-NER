#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time    : 2022/1/25 11:39
# @Author  : SinGaln

"""
star_transformer修改思路：
    (1)前向环和后向环Biaffine融合,经过线性层后引入位置向量
    (2)前向环和后向环双线性融合，通过multi-attention进行位置编码后输出
    (3)......
"""
import copy
import math
import torch
import torch.nn as nn
from .loss import loss_fun
from .bert import AutoBertModel
import torch.nn.functional as F


class MultiHeadsAttention(nn.Module):
    def __init__(self, config, num_labels, attention_head_size):
        super(MultiHeadsAttention, self).__init__()
        self.config = config
        self.num_attention_heads = num_labels
        self.attention_head_size = attention_head_size
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def sinusoidal_position_embeddings(self, inputs, attention_head_size):
        seq_len = inputs.size(1)
        position_ids = torch.arange(
            0, seq_len, dtype=torch.float32, device=inputs.device)

        indices = torch.arange(
            0, attention_head_size // 2, dtype=torch.float32, device=inputs.device)
        indices = torch.pow(10000.0, -2 * indices / attention_head_size)
        embeddings = torch.einsum('n,d->nd', position_ids, indices)  # [seq_len, attention_head_size // 2]
        embeddings = torch.stack([embeddings.sin(), embeddings.cos()], dim=-1)  # [seq_len, attention_head_size // 2, 2]
        embeddings = torch.reshape(embeddings, (seq_len, attention_head_size))  # [seq_len, attention_head_size]
        embeddings = embeddings[None, None, :, :]  # [1, 1, seq_len, attention_head_size]
        return embeddings

    def forward(self, inputs, attention_head_size, attention_mask=None):
        mixed_query_layer = self.query(inputs)  # [batch_size, seq_len, hidden_size]
        mixed_key_layer = self.key(inputs)  # [batch_size, seq_len, hidden_size]
        mixed_value_layer = self.value(inputs)  # [batch_size, seq_len, hidden_size]

        query_layer = self.transpose_for_scores(mixed_query_layer)  # [batch_size, num_heads, seq_len, heads_size]
        key_layer = self.transpose_for_scores(mixed_key_layer)  # [batch_size, num_heads, seq_len, heads_size]
        value_layer = self.transpose_for_scores(mixed_value_layer)  # [batch_size, num_heads, seq_len, heads_size]
        sinusoidal_positions = self.sinusoidal_position_embeddings(inputs, attention_head_size)
        # 计算cos
        cos_pos = torch.repeat_interleave(sinusoidal_positions[..., 1::2], 2, dim=-1)
        # 计算sin
        sin_pos = torch.repeat_interleave(sinusoidal_positions[..., ::2], 2, dim=-1)
        '''
            query_layer[..., 1::2]为按列取最后一维的偶数列  shape:[batch_size, num_heads, seq_len, head_dim / 2]
            query_layer[..., ::2]为按列取的最后一维的奇数列  shape:[batch_size, num_heads, seq_len, head_dim / 2]
            通过stack拼接后得到的为增加了一维，如下例所示：
            a = [[[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]]]
            b = [[[10, 20, 30],
                [40, 50, 60],
                [70, 80, 90]]]
            c = torch.stack(a,b,dim=0)
            tensor([[[[ 1,  2,  3],
                  [ 4,  5,  6],
                  [ 7,  8,  9]]],
                [[[10, 20, 30],
                  [40, 50, 60], 
                  [70, 80, 90]]]]) torch.Size([2, 1, 3, 3])
            d = torch.stack(a,b,dim=1)
            tensor([[[[ 1,  2,  3],
                  [ 4,  5,  6],
                  [ 7,  8,  9]],
                 [[10, 20, 30],
                  [40, 50, 60],
                  [70, 80, 90]]]]) torch.Size([1, 2, 3, 3])
            e = torch.stack(a,b,dim=-1)
            tensor([[[[ 1, 10],
                  [ 2, 20],
                  [ 3, 30]],
                 [[ 4, 40],
                  [ 5, 50],
                  [ 6, 60]],
                 [[ 7, 70],
                  [ 8, 80],
                  [ 9, 90]]]]) torch.Size([1, 3, 3, 2])
            通过以上例子就可以知道，这两个矩阵拼接后的维度增加了一维，并且是两个矩阵最后一维的元素进行拼接，如上述的e一样，
            所以torch.stack([-query_layer[..., 1::2], query_layer[..., ::2]],dim=-1) shape:[batch_size, num_heads, seq_len, head_size/2, 2]
            最后通过reshape把最后的两维进行合并得到qw2,kw2 shape:[batch_size, num_heads,seq_len, head_dim]
            '''
        qw2 = torch.stack([-query_layer[..., 1::2], query_layer[..., ::2]],
                          dim=-1).reshape_as(query_layer)  # [batch_size, num_heads, seq_len, head_dim]
        query_layer = query_layer * cos_pos + qw2 * sin_pos  # [batch_size, num_heads, seq_len, head_dim]
        kw2 = torch.stack([-key_layer[..., 1::2], key_layer[..., ::2]],
                          dim=-1).reshape_as(key_layer)  # [batch_size, num_heads, seq_len, head_dim]
        key_layer = key_layer * cos_pos + kw2 * sin_pos  # [batch_size, num_heads, seq_len, head_dim]
        attention_scores = torch.matmul(query_layer,
                                        key_layer.transpose(-1, -2))  # [batch_size, num_heads, seq_len, seq_len]
        attention_scores = attention_scores / math.sqrt(self.all_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # 对attention scores 按列进行归一化
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # dropout
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)  # [batch_size, num_heads, seq_len, head_dim]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # [batch_size, seq_len, num_heads, head_dim]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)  # [batch_size, seq_len, embedding_size]
        return context_layer, query_layer, key_layer


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


class Biaffine(nn.Module):
    def __init__(self, in_size, out_size, bias_x=True, bias_y=True):
        super().__init__()
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.out_size = out_size
        # [hidden_size+1, num_labels, hidden_size+1]
        self.U = torch.nn.Parameter(torch.randn(in_size + int(bias_x), out_size, in_size + int(bias_y)))

    def forward(self, x, y):
        if self.bias_x:
            # [batch_size, seq_len, hidden_size+1]
            x = torch.cat((x, torch.ones_like(x[..., :1])), dim=-1)
        if self.bias_y:
            # [batch_size, seq_len, hidden_size+1]
            y = torch.cat((y, torch.ones_like(y[..., :1])), dim=-1)
        bilinar_mapping = torch.einsum('bxi,ioj,byj->bxyo', x, self.U, y)
        return bilinar_mapping


class StarTransformerNER(nn.Module):
    def __init__(self, config, num_labels, attention_head_size, vocab_size, use_bert=False):
        super(StarTransformerNER, self).__init__()
        self.use_bert = use_bert
        self.num_labels = num_labels
        self.attention_head_size = attention_head_size
        if use_bert:
            self.bert = AutoBertModel(config)
        else:
            self.embedding = nn.Embedding(vocab_size, config.hidden_size)
        # 卫星多头注意力
        self.multi_att_satellite = MultiHeadsAttention(config, self.num_labels, self.attention_head_size)
        # 中继点多头注意力
        self.multi_att_relay = copy.deepcopy(self.multi_att_satellite)
        # 卫星layernorm
        self.ln_satellite = LayerNorm(self.num_labels * self.attention_head_size)
        # 中继layernorm
        self.ln_relay = copy.deepcopy(self.ln_satellite)
        # 双仿射融合
        self.biffine = Biaffine(self.num_labels * self.attention_head_size, self.num_labels)
        # 双线性融合
        self.bilinear_pool = nn.Bilinear(config.hidden_size, config.hidden_size, config.hidden_size)
        self.dense = nn.Linear(config.hidden_size, self.num_labels * 2)

    # Transformer环
    def cycle_shift(self, hidden: torch.Tensor, forward=True):
        batch, length, hideen_size = hidden.size()
        # 前向环
        if forward:
            temp = hidden[:, -1, :]
            for i in range(length - 1):
                hidden[:, i + 1, :] = hidden[:, i, :]
            hidden[:, 0, :] = temp
        # 后向环
        else:
            temp = hidden[:, 0, :]
            for i in range(1, length):
                hidden[:, i - 1, :] = hidden[:, i, :]
            hidden[:, -1, :] = temp
        return hidden

    def transpose(self, x):
        head_size = x.size()[-1]
        x_shape = x.transpose(-2, -3)
        x_shape = x_shape.size()[:2] + (self.num_labels * head_size,)
        return x.reshape(*x_shape)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, entity_labels_id=None, biaffine=True):
        if self.use_bert:
            hidden_states = self.bert.forward(input_ids=input_ids, attention_mask=attention_mask,
                                              token_type_ids=token_type_ids).last_hidden_state
        else:
            hidden_states = self.embedding(input_ids)
        hidden = hidden_states.clone()
        batch_size, length, hidden_dim = hidden.size()
        # 计算中继节点 (We initialize the state with H0 = E and s0 = average(E))
        center = F.avg_pool2d(hidden, (hidden.shape[1], 1)).unsqueeze(1)
        # 计算前向环以及后向环
        forward_hidden, backward_hidden = self.cycle_shift(hidden.clone(), True), self.cycle_shift(hidden.clone(),
                                                                                                   False)
        if biaffine:
            # 分别进行starformer
            forwards, _, _ = self.multi_att_satellite(forward_hidden, self.attention_head_size)
            backwards, _, _ = self.multi_att_satellite(backward_hidden, self.attention_head_size)
            forward_hidden = self.ln_satellite(forwards)
            backward_hidden = self.ln_satellite(backwards)
            # 使用双仿射对前向环和后向环进行融合(Biaffine)
            hidden_logits = self.biffine(forward_hidden, backward_hidden)  # [batch_size, seq_len, seq_len, num_labels]
        else:
            hidden = self.bilinear_pool(forward_hidden, backward_hidden)
            _, qw, kw = self.multi_att_satellite(hidden, self.attention_head_size)
            qw, kw = self.transpose(qw), self.transpose(kw)
            hidden_logits = torch.einsum('bmd,bnd->bmn', qw, kw) / self.num_labels ** 0.5
            bias = torch.einsum('bnh->bhn', self.dense(hidden)) / 2
            hidden_logits = hidden_logits[:, None] + bias[:, ::2, None] + bias[:, 1::2, :, None]
            padding_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.num_labels, length, length)
            hidden_logits = hidden_logits * padding_mask - (1 - padding_mask) * 1e12
            # 排除下三角
            mask = torch.tril(torch.ones_like(hidden_logits), -1)
            hidden_logits = hidden_logits - mask * 1e12  # [batch_size, num_labels, seq_len, seq_len]
        center = center.squeeze(1)
        content_layer, _, _ = self.multi_att_relay(center, self.attention_head_size)
        center_hidden = self.ln_relay(content_layer)
        # 计算模型的loss,提供loss计算函数loss_function

        if entity_labels_id is not None:
            entity_loss = loss_fun(entity_labels_id, hidden_logits)
            preds = torch.argmax(hidden_logits)
            return hidden_logits, center_hidden, entity_loss, preds


if __name__ == "__main__":
    from transformers import BertTokenizerFast, BertConfig

    tokenizer = BertTokenizerFast.from_pretrained("../AutoTinyBert")
    data = "彭小军认为，国内银行现在走的是台湾的发卡模式"
    out = tokenizer.encode_plus(data, return_tensors="pt")
    input_ids, token_type_ids, attention_mask = out["input_ids"], out["token_type_ids"], out["attention_mask"]
    config = BertConfig.from_pretrained("../AutoTinyBert")
    model = StarTransformerNER(config=config, num_labels=10, attention_head_size=64, use_bert=False)
    output, _, loss, pred = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                            biaffine=False)
    print(output.size())
