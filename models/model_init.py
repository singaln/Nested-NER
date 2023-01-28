#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time    : 2022/2/7 11:29
# @Author  : SinGaln
"""
模型初始化
"""
import torch
from transformers import BertConfig
from utils import apex_device, get_device
# from .model import StarformerNER
from .NestedModel import Models
# from .starformer import StarformerNER
from config import LANConfig, LSTMConfig, TrainingConfig, BertConfigs

class ModelInit(object):
    def __init__(self, args, vocab_size, num_labels, id2tag, mode):
        self.lstm_config = LSTMConfig()
        self.lan_config = LANConfig()
        self.bert_config = BertConfigs()
        self.train_config = TrainingConfig()
        self.device = args.device

        self.id2tag = id2tag
        # starmodel 模型初始化
        if args.model_type == "bert" and args.entity_type == "nested":
            self.config = BertConfig.from_pretrained(args.autobert_path)
            # self.startmodel = StarformerNER(config=self.config, num_labels=num_labels, head_size=64,
            #                                 vocab_size=vocab_size, mode=mode)
            self.startmodel = Models(config=self.config, num_labels=num_labels, head_size=64, mode=mode)
            self.startmodel, self.model_device = get_device(self.startmodel, self.device)