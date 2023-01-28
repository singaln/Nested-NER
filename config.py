#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: SinGaln
# @time: 2022/1/5 10:33

"""
各个模型的默认参数配置
"""


# 设置lstm训练参数
class TrainingConfig(object):
    batch_size = 16
    # 学习速率
    lr = 1e-3
    bert_lr = 2e-5
    epochs = 50
    logging_step = 5
    eval_batch_size = 16
    adam_epsilon = 1e-8
    warmup_steps = 0
    evaluate_step = 100
    temperature = 1


class LSTMConfig(object):
    emb_size = 128  # 词向量的维数
    hidden_size = 128  # lstm隐向量的维数
    num_layers = 1
    dropout = 0.2
    out_size = 128


class LANConfig(object):
    emb_size = 128  # 词向量的维数
    hidden_size = 100  # lstm隐向量的维数
    dropout = 0.5
    num_attention_head = 5
    num_of_lstm_layer = 2


class BertConfigs(object):
    dropout_rate = 0.2
    alpha = 0.5


class PredictConfig(object):
    batch_size = 64