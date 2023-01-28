#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time    : 2022/2/7 9:12
# @Author  : SinGaln

"""
数据加载模块：Flat和Nested实体识别数据处理,主要将数据从BIO标记形式转换为idx标记的形式，将数据标签转换为矩阵的形式。
"""
import os
import torch
import logging
import argparse
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


def get_vocab(data_lst, label_lst):
    """
    :param data_lst: 读取到train中的所有数据
    :return: word2id, tag2id
    """
    word2id, tag2id = {}, {}
    sent_lst, lab_lst = [], []
    word_special = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}
    for word_sent in data_lst:
        word_lst = word_sent.split()
        for word in word_lst:
            if word not in sent_lst:
                sent_lst.append(word)
    for tag_sent in label_lst:
        tag_lst = tag_sent.split()
        for tag in tag_lst:
            if tag not in lab_lst:
                lab_lst.append(tag)

    word2id.update({w: idx + 4 for idx, w in enumerate(sent_lst)})
    tag2id.update({t: idx + 3 for idx, t in enumerate(lab_lst)})

    word_special.update(word2id)
    new_tag2id = {}
    lab = []
    for tag, value in tag2id.items():
        if tag == "O":
            continue
        else:
            new_tag = tag.split("-")[1]
            if new_tag not in lab:
                lab.append(new_tag)
    for i, v in enumerate(lab):
        new_tag2id[v] = i
    return word_special, new_tag2id


class EntityProcess(object):
    """
    数据格式转换及数据处理
    """

    def __init__(self, args):
        self.args = args

    def _read_input_file(self, input_files):
        with open(input_files, "r", encoding="utf-8") as f:
            contents = []
            labels = []
            sentences = f.read().split("\n\n")
            for sentence in sentences:
                lines = sentence.split("\n")
                new_line = []
                new_label = []
                for line in lines:
                    if line:
                        new_line.append(line.split("\t")[0].strip())
                        new_label.append(line.split("\t")[1].strip())
                assert len(new_line) == len(new_label), "Error of text length and label length."
                contents.append(" ".join(new_line))
                labels.append(" ".join(new_label))
            return contents, labels

    def transfer_idx(self, texts, labels, word2id, tag2id):
        """
        :param texts: a b c d e f g
        :param labels: O B-loc I-loc I-loc O O O
        :return: text [a b c d e f g,(1,4,0)],.....
        """
        label_lst = []
        dic = {}
        text_lst = []
        for word in texts:
            text_lst.append(word2id[word] if word in word2id else word2id["[UNK]"])
        if len(text_lst) > 510:
            text_lst = text_lst[:510]
        text_lst.insert(0, word2id["[CLS]"])
        text_lst.append(word2id["[SEP]"])
        label_lst.append(text_lst)
        for idx, (t, l) in enumerate(zip(texts, labels)):
            dic[idx] = (t, l)

        for k in list(dic.keys()):
            if dic[k][1] == "O":
                del dic[k]

        idx_dic = {}
        for i, key in enumerate(list(dic.keys())):
            if dic[key][1].startswith("B"):
                idx_dic[i] = key

        each_key = []
        if idx_dic:
            for i in range(len(list(idx_dic.keys())) - 1):
                start1 = list(idx_dic.keys())[i]
                start2 = list(idx_dic.keys())[i + 1]
                each_key.append(list(dic.keys())[start1:start2])
            each_key.append(list(dic.keys())[list(idx_dic.keys())[-1]:])
        for each in each_key:
            lst = ()
            lst += (each[0] + 1, each[-1] + 1,)
            idx1 = []
            for e in each:
                idx1.append(tag2id[dic[e][1].split("-")[1]])
            idx2 = list(set(idx1))[0]
            lst += (idx2,)
            label_lst.append(lst)
        return label_lst

    def get_example(self, contents, labels, word2id, tag2id):
        all_examples = []
        for sentence, label in zip(contents, labels):
            sent = sentence.split()
            tag = label.split()
            data = self.transfer_idx(sent, tag, word2id, tag2id)
            all_examples.append(data)
        return all_examples


class EntityDataset(Dataset):
    def __init__(self, examples, tag2id):
        self.tag2id = tag2id
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def sequence_padding(self, inputs, length=None, value=0, seq_dims=1, mode='post'):
        """Numpy函数，将序列padding到同一长度
        """
        if length is None:
            length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
        elif not hasattr(length, '__getitem__'):
            length = [length]

        slices = [np.s_[:length[i]] for i in range(seq_dims)]
        slices = tuple(slices) if len(slices) > 1 else slices[0]
        pad_width = [(0, 0) for _ in np.shape(inputs[0])]

        outputs = []
        for x in inputs:
            x = x[slices]
            for i in range(seq_dims):
                if mode == 'post':
                    pad_width[i] = (0, length[i] - np.shape(x)[i])
                elif mode == 'pre':
                    pad_width[i] = (length[i] - np.shape(x)[i], 0)
                else:
                    raise ValueError('"mode" argument must be "post" or "pre".')
            x = np.pad(x, pad_width, 'constant', constant_values=value)
            outputs.append(x)

        return np.array(outputs)

    def collect_fn(self, data):
        if len(data[0]) > 1:
            data.sort(key=lambda x: len(x[0]), reverse=True)
            data_length = [len(sequence[0]) for sequence in data]
            max_length = data_length[0]
            attention_mask = []
            token_type_ids = []
            for i in data_length:
                attention_mask.append([1] * i)
                token_type_ids.append([0] * i)
            lines = [i[0] for i in data]
            span_mapping = []
            span_mapping.append((0, 0))
            for i in range(max_length - 1):
                span_mapping.append((i, i + 1))
            span_mapping.append((0, 0))
            start_mapping = {j[0]: i for i, j in enumerate(span_mapping) if j != (0, 0)}
            end_mapping = {j[-1] - 1: i for i, j in enumerate(span_mapping) if j != (0, 0)}
            labels = np.zeros((len(self.tag2id), max_length, max_length))
            batch_labels = []
            for item in data:
                for start, end, label in item[1:]:
                    if start in start_mapping and end in end_mapping:
                        start = start_mapping[start]
                        end = end_mapping[end]
                        labels[label, start, end] = 1
                batch_labels.append(labels[:, :len(item[0]), :len(item[0])])
            input_ids = pad_sequence([torch.from_numpy(np.array(line)) for line in lines], batch_first=True,
                                     padding_value=0)
            attention_mask = pad_sequence([torch.from_numpy(np.array(mask)) for mask in attention_mask],
                                          batch_first=True,
                                          padding_value=0)
            token_type_ids = pad_sequence([torch.from_numpy(np.array(token_type)) for token_type in token_type_ids],
                                          batch_first=True, padding_value=0)
            labels = torch.tensor(self.sequence_padding(batch_labels, seq_dims=3)).long()
            return input_ids.type(torch.long), attention_mask.type(torch.long), token_type_ids.type(torch.long), labels

    def __getitem__(self, item):
        return self.examples[item]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default="./datasets/car/", help="The path of data sets.")

    args = parser.parse_args()
    ep = EntityProcess(args=args)
    train_path = os.path.join(args.data_path, "train.txt")
    contents, labels = ep._read_input_file(train_path)
    word2id, tag2id = get_vocab(contents, labels)
    train_data = ep.get_example(contents, labels, word2id, tag2id)
    entity_label_lst = list(tag2id.keys())
    ner_train_data = EntityDataset(examples=train_data, tag2id=tag2id)
    data_loader = DataLoader(ner_train_data, batch_size=32, collate_fn=ner_train_data.collect_fn)
    # for data in data_loader:
    #     print(data)
