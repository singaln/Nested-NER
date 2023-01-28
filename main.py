#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time    : 2022/2/7 12:00
# @Author  : SinGaln
"""
模型训练主函数
"""
import os
import time
import logging
import argparse
import warnings
from utils import write
from trainer import Trainer
from datetime import datetime
from utils import get_vocab
from transformers import BertTokenizerFast
# from loader import EntityProcess, get_vocab
from data_loader import EntDataset, load_data

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def main(args):
    print("------------ 开始时间 %s ------------" % (datetime.now()))
    project_path = os.path.join(args.data_path, args.project_name)
    # tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(args.autobert_path, do_lower_case=True)
    tag2id, id2tag = get_vocab(args)
    # train_data and val_data
    train_data = load_data(project_path + "/CMeEE_train.json", tag2id)
    dev_data = load_data(project_path + "/CMeEE_dev.json", tag2id)
    # ep = EntityProcess(args)
    # sentences, labels = ep._read_input_file(project_path + "/train.txt")
    # word2id, tag2id = get_vocab(sentences, labels)
    # write(project_path, word2id, tag2id)
    # id2tag = {value: key for key, value in tag2id.items()}
    # train_data = ep.get_example(sentences, labels, word2id, tag2id)
    # sentences_dev, labels_dev = ep._read_input_file(project_path + "/test.txt")
    # dev_data = ep.get_example(sentences_dev, labels_dev, word2id, tag2id)
    trainer = Trainer(args=args, vocab_size=len(tokenizer.get_vocab()), num_labels=len(id2tag), id2tag=id2tag, mode=args.mode)

    if args.do_train:
        start_time = time.time()
        trainer.train(train_data, dev_data, tokenizer)
        print("训练总用时 %s 秒" % (time.time() - start_time))
    print("------------ 结束时间 %s ------------" % (datetime.now()))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_type", type=str, default="bert", choices=["bert"],
                        help="Select the type of model required for training.")
    parser.add_argument("--mode", type=str, default="Global", choices=["Global", "Efficient", "Biaffine"],
                        help="Select the mode of model.")
    parser.add_argument("--entity_type", type=str, default="nested", choices=["flat", "nested"],
                        help="Select the type of entity required for training.")
    parser.add_argument("--project_name", type=str, default="car", required=True, help="The name of project.")
    parser.add_argument("--data_name", type=str, default="", help="The data name.")
    parser.add_argument("--crf", action='store_true', help="Whether to enable crf.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cuda:0","cuda:1","cpu", "cuda:0,1"], help="Select the device.")
    parser.add_argument("--apex", action='store_true', help="Whether to mixing precision.")

    # data path, model path, save path
    parser.add_argument("--data_path", type=str, default="./datasets", help="The path of dataset.")
    parser.add_argument("--autobert_path", type=str, default="./AutoTinyBert",
                        help="The path of pretrained model.")
    parser.add_argument("--save_path", type=str, default="save_model", help="The path of each model to save.")

    parser.add_argument("--do_train", action='store_true', help="training.")
    args = parser.parse_args()
    main(args)