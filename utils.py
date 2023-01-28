#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time    : 2022/2/7 9:52
# @Author  : SinGaln
"""
评估方法以及log信息
"""
import os
import json
import torch
import logging
import numpy as np

logger = logging.getLogger(__name__)

def get_vocab(args):
    labels = {}
    path = os.path.join(args.data_path, args.project_name, "CMeEE_train.json")
    with open(path, "r", encoding="utf-8") as f:
        dic_lst = json.load(f)
        tags = []
        for dic in dic_lst:
            lst = dic["entities"]
            for i in range(len(lst)):
                label = lst[i]["type"]
                if label not in tags:
                    tags.append(label)
        for i, tag in enumerate(tags):
            labels[tag] = i
    id2labels = {value:key for key, value in labels.items()}
    return labels, id2labels



def multilabel_categorical_crossentropy(y_true, y_pred):
    """
    详情见: https://kexue.fm/archives/7359
    y_true和y_pred的shape一致，y_true的元素非0即1，
    1表示对应的类为目标类，0表示对应的类为非目标类。
    """
    y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
    y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
    y_pred_pos = y_pred - (1 - y_true) * 1e12  # mask the pred outputs of neg classes
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

    return (neg_loss + pos_loss).mean()


def loss_fun(y_true, y_pred):
    """
    :param y_true: [batch_size, num_labels, seq_len, seq_len]
    :param y_pred: [batch_size, num_labels, seq_len, seq_len]
    :return: loss scores
    """
    bh = y_pred.shape[0] * y_pred.shape[1]
    y_pred = torch.reshape(y_pred, (bh, -1))
    y_true = torch.reshape(y_true, (bh, -1))
    loss = multilabel_categorical_crossentropy(y_true, y_pred)
    return loss


class Metrics(object):
    """
    f1: 2 * precision * recall / precision + recall
    precision_score: pred_true / all_pred
    recall_score:
    """

    def get_sample_f1(self, y_pred, y_true):
        y_pred = torch.gt(y_pred, 0).float()
        return 2 * torch.sum(y_true * y_pred) / torch.sum(y_true + y_pred)

    def get_sample_precision(self, y_pred, y_true):
        y_pred = torch.gt(y_pred, 0).float()
        return torch.sum(y_pred[y_true == 1]) / (y_pred.sum() + 1)

    def get_evaluate_fpr(self, y_pred, y_true):
        y_pred = y_pred.data.cpu().numpy()
        y_true = y_true.data.cpu().numpy()
        pred = []
        true = []
        for b, l, start, end in zip(*np.where(y_pred > 0)):
            pred.append((b, l, start, end))
        for b, l, start, end in zip(*np.where(y_true > 0)):
            true.append((b, l, start, end))
        R = set(pred)
        T = set(true)
        correct = len(R & T)
        preds = len(R)
        trues = len(T)
        if preds == 0 or trues == 0:
            return 0, 0, 0
        precision = correct / preds
        recall = correct / trues
        if precision == 0 or recall == 0:
            return 0, 0, 0
        f1 = 2 * precision * recall / (precision + recall)
        return f1, precision, recall


# 初始化log的输出
def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


def write(data_path, word2id, tag2id):
    with open(data_path + "/word2id.json", "w", encoding="utf-8") as f1, open(data_path + "/tag2id.json", "w",
                                                                              encoding="utf-8") as f2:
        token2id = json.dumps(word2id, ensure_ascii=False, indent=2)
        label2id = json.dumps(tag2id, ensure_ascii=False, indent=2)
        f1.write(token2id)
        f2.write(label2id)


def get_device(model, device):
    """
    :param model: 初始化模型
    :param device: 传入的device信息
    :return:
    """
    if device == "cuda:0,1":
        model = torch.nn.parallel.DataParallel(model, device_ids=[0, 1])
        model = model.cuda()
        return model, "cuda"
    elif device == "cuda:0" or device == "cpu" or device == "cuda:1":
        model = model.to(device)
        return model, device


def apex_device(args, model, device):
    """
    :param args: 传入的参数
    :param model: 模型
    :param device: 传入的device信息
    :return: 使用混合精度的模型和模型的device信息
    """
    global model_device
    if args.apex and (device == "cuda:0" or device == "cuda:1"):
        model, model_device = get_device(model, device)
    elif args.apex and device == "cuda:0,1":
        model, model_device = model, "cuda"
    elif args.apex and device == "cpu":
        raise Exception("CPU devices are incompatible with mixing precision. Select CUDA as the device.")
    elif not args.apex:
        model, model_device = get_device(model, device)
    return model, model_device


def save_bert_model(args, model, save_path):
    """
    :param args: 传入的参数
    :param model: bert模型
    :param save_path: 保存路径
    :return:
    """
    # Save model checkpoint (Overwrite)
    save_path = os.path.join(save_path, "finetune_bert_model")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(save_path)

    # Save training arguments together with the trained model
    torch.save(args, os.path.join(save_path, 'training_args.bin'))
    logger.info("Saving model checkpoint to %s", save_path)


def save_model(args, model, save_path):
    """
    :param args: 传入的参数
    :param model: 模型
    :param save_path:保存路径
    :return:
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if args.model_type == "lstm" and not args.crf and not args.lan:
        torch.save(model, save_path + "/" + args.model_type + ".pt")
    elif args.model_type == "lstm" and args.crf:
        torch.save(model, save_path + "/" + args.model_type + "_crf.pt")
    elif args.model_type == "lstm" and args.lan:
        torch.save(model, save_path + "/" + args.model_type + "_lan.pt")
    elif args.model_type == "bert" and args.distilled and not args.finetune:
        torch.save(model, save_path + "/" + args.model_type + "_distilled.pt")
    logger.info("Saving model checkpoint to %s", save_path)
