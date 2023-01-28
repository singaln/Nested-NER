#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time    : 2022/2/7 9:50
# @Author  : SinGaln
"""
训练函数：主要定义模型的初始化、优化函数、评估函数以及训练过程等
"""

import time
import torch
from tqdm import tqdm
from utils import Metrics
from visdom import Visdom
from loader import EntityDataset
from data_loader import EntDataset
from torch.utils.data import DataLoader
from models.model_init import ModelInit
from train_steps import starformer_steps
from utils import loss_fun, save_model, get_vocab

# vis = Visdom()

metrics = Metrics()


class Trainer(object):
    def __init__(self, args, vocab_size, num_labels, id2tag, mode):
        self.args = args
        self.tag2id = {tag: idx for idx, tag in id2tag.items()}
        self.init_model = ModelInit(args, vocab_size, num_labels, id2tag, mode)

    def train(self, train_data, dev_data, tokenizer):
        global model, scheduler
        tag2id, id2tag = get_vocab(self.args)
        # ner_train_data = EntityDataset(examples=train_data, tag2id=self.tag2id)
        # ner_dev_data = EntityDataset(examples=dev_data, tag2id=self.tag2id)
        # collate_fn = ner_train_data.collect_fn
        # train_dataloader = DataLoader(train_data, batch_size=self.init_model.train_config.batch_size,
        #                               collate_fn=collate_fn)
        # dev_dataloader = DataLoader(ner_dev_data, batch_size=self.init_model.train_config.eval_batch_size,
        #                             collate_fn=collate_fn)
        ner_train = EntDataset(train_data, tokenizer=tokenizer, ent2id=tag2id)
        train_dataloader = DataLoader(ner_train, batch_size=self.init_model.train_config.batch_size, collate_fn=ner_train.collate, shuffle=True)
        ner_evl = EntDataset(dev_data, tokenizer=tokenizer, ent2id=tag2id)
        dev_dataloader = DataLoader(ner_evl, batch_size=self.init_model.train_config.eval_batch_size, collate_fn=ner_evl.collate, shuffle=False)
        # 多模型
        print("***==========================***Running training***==========================***")
        print("  Num examples = ", len(train_dataloader))
        print("  Num Epochs = ", self.init_model.train_config.epochs)

        global_step = 0
        best_result = 0

        for epoch in range(0, int(self.init_model.train_config.epochs)):
            total_loss, total_f1 = 0., 0.
            start_time = time.time()
            with tqdm(train_dataloader, desc="Training") as p:
                for step, batch in enumerate(train_dataloader):
                    if self.args.model_type == "bert" and self.args.entity_type == "nested":
                        model = self.init_model.startmodel
                        loss, logits, labels = starformer_steps(args=self.args, batch=batch,
                                                train_dataloader=ner_train,
                                                model=model, lr=self.init_model.train_config.bert_lr,
                                                adam_epsilon=self.init_model.train_config.adam_epsilon,
                                                warmup_steps=self.init_model.train_config.warmup_steps,
                                                epochs=self.init_model.train_config.epochs,
                                                device=self.init_model.model_device, batch_size=self.init_model.train_config.batch_size)
                        total_loss += loss.item()
                    global_step += 1
                    sample_f1 = metrics.get_sample_f1(logits, labels)
                    total_loss += loss.item()
                    total_f1 += sample_f1.item()
                    avg_loss = total_loss / (step + 1)
                    avg_f1 = total_f1 / (step + 1)
                    if step % 10 == 0:
                        p.set_postfix({"loss": "{}".format(avg_loss), "f1_scores": "{}".format(avg_f1)})
                        p.update()
            scheduler.step()
            use_time = time.time() - start_time
            print("epoch {}/{}, use time: {}".format(epoch, self.init_model.train_config.epochs, use_time))
            dev_loss, result_f1, result_precision, result_recall = self.evaluate(dev_dataloader, model, self.init_model.id2tag)
            print("precision: %.4f, recall: %.4f, f1-score: %.4f" % (result_precision, result_recall, result_f1))
            if best_result <= result_f1:
                best_result = result_f1
                if self.args.model_type == "bert":
                    print("Saving model checkpoint to %s" % self.args.save_path)
                    # save_bert_model(self.args, model, self.args.save_path)
                else:
                    print("Saving model checkpoint to %s" % self.args.save_path)
                    save_model(self.args, model, self.args.save_path)
            # vis.line([total_loss], [epoch], win="train_loss", update='append')
            # vis.line([result_precision], [epoch], win="train_precision", update='append')
            # vis.line([result_recall], [epoch], win="train_recall", update='append')
            # vis.line([result_f1], [epoch], win="train_f1", update='append')
            # vis.line([result_acc], [epoch], win="train_acc", update='append')

    def evaluate(self, dev_dataset, model, all_labels):
        print("***==========================****Evaluating****==========================***")
        model.eval()
        total_f1, total_precision, total_recall = 0., 0., 0.
        with torch.no_grad():
            for step, batch in enumerate(tqdm(dev_dataset)):

                if self.args.model_type == "bert" and self.args.entity_type == "nested":
                    _, input_ids, attention_mask, token_type_ids, labels = batch
                    input_ids, attention_mask, token_type_ids, labels = input_ids.to(
                        self.init_model.model_device), attention_mask.to(
                        self.init_model.model_device), token_type_ids.to(self.init_model.model_device), labels.to(
                        self.init_model.model_device)
                    logits = model(input_ids=input_ids, attention_mask=attention_mask,
                                   token_type_ids=token_type_ids, entity_label_ids=labels)
                loss = loss_fun(labels, logits)
                f1, precision, recall = metrics.get_evaluate_fpr(logits, labels)
                total_f1 += f1
                total_precision += precision
                total_recall += recall
            avg_f1 = total_f1 / (len(dev_dataset))
            avg_precision = total_precision / (len(dev_dataset))
            avg_recall = total_recall / (len(dev_dataset))
        print("Dev Loss: {:.6f}".format(loss))

        return loss.item(), avg_f1, avg_precision, avg_recall
