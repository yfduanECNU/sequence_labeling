#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
NER model using Bert+bilstm+crf
======
A class for BaseModel for all basic deep learning models.
配置文件：cmed.dl.base_model.norm.json
forward(X, X_lengths)：
    X：padding后的输入数据。2维列表，size = [batch_size, seq_len]
    X_lengths: 输入数据的真实长度。1维列表，size = [batch_size]
    返回值：tag_scores。2维张量，shape = [batch_size*seq_len, tags_size]
"""
import argparse
import datetime
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from transformers.optimization import AdamW
from transformers import BertTokenizer, BertConfig, BertModel
from torchcrf import CRF

sys.path.insert(0, '.')
sys.path.insert(0, '..')
from sequence_labeling.data_loader import DataLoader
from sequence_labeling.data_processor import DataProcessor
from sequence_labeling.utils.evaluate import Evaluator
from sequence_labeling.utils.evaluate_2 import Metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)


class Bert_BiLSTM_CRF(nn.Module):

    def __init__(self, **config):
        super(Bert_BiLSTM_CRF, self).__init__()
        self.config = config
        self.model_name = config['model_name']
        self.data_root = config['data_root']
        self.model_save_path = config['model_save_name']

        self.embedding_dim = config['embedding_dim']
        self.hidden_dim = config['hidden_dim']
        self.layers = config['num_layers']
        self.bidirectional = True if config['bidirectional'] == 'True' else False
        self.n_directions = 2 if self.bidirectional else 1
        self.dropout_p = config['dropout_rate']
        self.pad_token = config['pad_token']

        self.num_epochs = config['num_epochs']
        self.batch_size = config['batch_size']
        self.learning_rate = config['learning_rate']

        self.tag_index_dict = DataProcessor(**self.config).load_tags()
        self.tags_size = len(self.tag_index_dict)
        self.vocab = DataProcessor(**self.config).load_vocab()
        self.vocab_size = len(self.vocab)
        self.padding_idx = self.vocab[self.pad_token]
        self.padding_idx_tags = self.tag_index_dict[self.pad_token]

        # Bert模型
        self.maxLength = 510
        self.checkpoint = config['checkpoint']
        self.tokenizer = BertTokenizer.from_pretrained(self.checkpoint)
        model_config = BertConfig.from_pretrained(self.checkpoint)
        model_config.output_hidden_states = True
        model_config.output_attentions = True
        self.word_embeddings = BertModel.from_pretrained(self.checkpoint, config=model_config)

        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim, num_layers=self.layers,
                            bidirectional=self.bidirectional, batch_first=True)

        # 将模型输出映射到标签空间
        self.output_to_tag = nn.Linear(self.hidden_dim * self.n_directions, self.tags_size)

        # CRF模型
        self.crf = CRF(self.tags_size, batch_first=True)

        # print('完成类{}的初始化'.format(self.__class__.__name__))

    def forward(self, x, y=None):
        # BertModel embedding
        batch = self.tokenizer(x, padding=True, truncation=True,
                               return_tensors="pt", add_special_tokens=False).to(device)
        embedded = self.word_embeddings(**batch).last_hidden_state
        output, _ = self.lstm(embedded)  # [batch_size, seq_len, hidden_dim]
        logits = self.output_to_tag(output)  # [batch_size*seq_len, tags_size]

        outputs = (logits,)
        if y is not None:
            loss = self.crf(logits, y) * (-1)
            outputs = (loss,) + outputs
        # contain: (loss), scores
        return outputs

    def run_train(self, model, data_path):
        print('Running {} model. Training...'.format(self.model_name))
        run_mode = 'train'
        model.to(device)
        model.train()

        # model.named_parameters(): [word_embeddings, lstm, output_to_tag, crf]
        embedding_optimizer = list(model.word_embeddings.named_parameters())
        lstm_optimizer = list(model.lstm.named_parameters())
        classifier_optimizer = list(model.output_to_tag.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        weight_decay = 0.01
        optimizer_grouped_parameters = [
            {'params': [p for n, p in embedding_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in embedding_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
            {'params': [p for n, p in lstm_optimizer if not any(nd in n for nd in no_decay)],
             'lr': self.learning_rate * 5, 'weight_decay': weight_decay},
            {'params': [p for n, p in lstm_optimizer if any(nd in n for nd in no_decay)],
             'lr': self.learning_rate * 5, 'weight_decay': 0.0},
            {'params': [p for n, p in classifier_optimizer if not any(nd in n for nd in no_decay)],
             'lr': self.learning_rate * 5, 'weight_decay': weight_decay},
            {'params': [p for n, p in classifier_optimizer if any(nd in n for nd in no_decay)],
             'lr': self.learning_rate * 5, 'weight_decay': 0.0},
            {'params': model.crf.parameters(), 'lr': self.learning_rate * 5}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate, correct_bias=False)

        train_losses = []  # 记录每一batch的loss
        valid_losses = []
        valid_f1_scores = []
        best_valid_f1 = float('-inf')
        torch.backends.cudnn.enabled = False
        for epoch in range(self.num_epochs):
            train_loss = 0
            train_data_num = 0  # 统计数据量
            for seq_list, tag_list in DataLoader(**self.config).data_generator(data_path=data_path,
                                                                               run_mode=run_mode):
                train_data_num += len(seq_list)
                # 生成batch_y
                batch_output = self.tag_to_index(tag_list)
                y = self.pad_taglist(batch_output)
                batch_y = torch.tensor(y).long().to(device)

                loss = model(seq_list, batch_y)[0]

                model.zero_grad()
                loss.backward()
                # 梯度裁剪
                nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=5)
                # 计算梯度
                optimizer.step()

                train_loss += loss.item()

            train_losses.append(train_loss / train_data_num)  # 记录评价结果

            # 模型验证
            avg_eval_loss, eval_f1 = self.evaluate(model, data_path)
            print("Done Epoch{}. Eval_Loss={}".format(epoch + 1, avg_eval_loss))
            valid_losses.append(avg_eval_loss)
            valid_f1_scores.append(eval_f1)

            # 保存参数最优的模型
            if best_valid_f1 < eval_f1:
                best_valid_f1 = eval_f1
                model_save_path = '{}{}_{}'.format(self.data_root, self.model_name, self.model_save_path)
                torch.save(model, '{}'.format(model_save_path))

        # self.prtplot(train_losses)
        # self.prtplot(valid_losses)

        return train_losses, valid_losses, valid_f1_scores

    def evaluate(self, model, data_path):
        # print('Running {} model. Evaluating...'.format(self.model_name))
        run_mode = 'eval'

        avg_loss, y_true, y_predict = self.eval_process(model, run_mode, data_path)
        f1_score = Evaluator().f1score(y_true, y_predict)

        return avg_loss, f1_score

    def test(self, data_path):
        print('Running {} model. Testing data in folder: {}'.format(self.model_name, data_path))
        run_mode = 'test'
        best_model_path = '{}{}_{}'.format(self.data_root, self.model_name, self.model_save_path)
        model = torch.load(best_model_path)

        avg_loss, y_true, y_predict = self.eval_process(model, run_mode, data_path)

        # 输出评价结果
        print(Evaluator().classifyreport(y_true, y_predict))
        f1_score = Evaluator().f1score(y_true, y_predict)

        # 输出混淆矩阵
        metrix = Metrics(y_true, y_predict)
        metrix.report_scores()
        metrix.report_confusion_matrix()

        return f1_score

    def eval_process(self, model, run_mode, data_path):
        model.to(device)
        model.eval()
        with torch.no_grad():
            total_loss = 0
            data_num = 0
            all_y_predict = []
            all_y_true = []
            for seq_list, tag_list in DataLoader(**self.config).data_generator(data_path=data_path,
                                                                               run_mode=run_mode):
                data_num += len(seq_list)
                # 生成batch_y
                batch_output = self.tag_to_index(tag_list)
                y = self.pad_taglist(batch_output)
                batch_y = torch.tensor(y).long().to(device)

                loss = model(seq_list, batch_y)[0]
                total_loss += loss.item()

                y_predict = model(seq_list)[0]
                y_predict = model.crf.decode(y_predict)
                y_predict = np.array(y_predict).flatten()
                y_predict = self.index_to_tag(y_predict, self.list_to_length(tag_list))
                all_y_predict = all_y_predict + y_predict

                y_true = y.flatten()
                y_true = self.index_to_tag(y_true, self.list_to_length(tag_list))
                all_y_true = all_y_true + y_true

            return total_loss / data_num, all_y_true, all_y_predict

    # 根据列表内容，生成描述其成员长度的list
    def list_to_length(self, input):
        seq_list = []
        for line in input:
            seq_list.append(line.strip().split())
        lenth_list = [len(seq) for seq in seq_list]
        return lenth_list

    def index_to_tag(self, y, y_len):
        index_tag_dict = dict(zip(self.tag_index_dict.values(), self.tag_index_dict.keys()))
        y_tagseq = []
        seq_len = int(len(y) / len(y_len))
        for i in range(len(y_len)):
            temp_list = y[i * seq_len:(i + 1) * seq_len]
            for count in range(seq_len):
                if count < y_len[i]:
                    y_tagseq.append(index_tag_dict[temp_list[count]])
        return y_tagseq

    def tag_to_index(self, output):
        batch_output = []
        for line in output:  # 依据Tag词典，将output转化为向量
            new_line = []
            for tag in line.strip().split():
                new_line.append(
                    int(self.tag_index_dict[tag]) if tag in self.tag_index_dict else print(
                        "There is a wrong Tag in {}!".format(line)))
            batch_output.append(new_line)

        return batch_output

    # 补齐每批tag列表数据
    def pad_taglist(self, taglist):
        seq_lengths = [len(st) for st in taglist]
        # create an empty matrix with padding tokens
        pad_token = self.padding_idx_tags
        longest_sent = max(seq_lengths)
        batch_size = len(taglist)

        # 不添加[CLS], [SEP]
        padded_seq = np.ones((batch_size, longest_sent)) * pad_token  # !!!

        # 添加[CLS], [SEP]
        # padded_seq = np.ones((batch_size, longest_sent+2)) * pad_token
        # copy over the actual sequences
        for i, x_len in enumerate(seq_lengths):
            sequence = taglist[i]

            # 不添加[CLS], [SEP]
            padded_seq[i, 0:x_len] = sequence[:x_len]  # !!!

            # 添加[CLS], [SEP]
            # padded_seq[i, 0] = self.tag_index_dict['[CLS]']
            # padded_seq[i, 1:x_len+1] = sequence[:x_len]
            # padded_seq[i, x_len+1] = self.tag_index_dict['[SEP]']
        return padded_seq


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='Process some description.')
    parser.add_argument('--phase', default='test', help='the function name.')

    args = parser.parse_args()

    if args.phase == 'test':
        print('This is a test process.')
    else:
        print("There is no {} function. Please check your command.".format(args.phase))
    end_time = datetime.datetime.now()
    print('{} takes {} seconds.'.format(args.phase, (end_time - start_time).seconds))

    print('Done Base_Model!')
