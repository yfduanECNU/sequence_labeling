#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
model: Vote
======
平行结构的模型
配置文件：cmed.dl.vote.norm.json

配置文件中"checkpoint"建议使用:
hfl/chinese-bert-wwm-ext
hfl/chinese-roberta-wwm-ext
hfl/chinese-macbert-base
"""

import argparse
import datetime
import sys
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertConfig, BertModel

sys.path.insert(0, '.')
sys.path.insert(0, '..')
from sequence_labeling.data_loader import DataLoader
from sequence_labeling.data_processor import DataProcessor
from sequence_labeling.utils.evaluate import Evaluator
from sequence_labeling.utils.evaluate_2 import Metrics
from .crf import CRF

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)


class Init_Class(nn.Module):
    def __init__(self, **config):
        super(Init_Class, self).__init__()
        self.config = config
        self.model_name = config['model_name']
        self.data_root = config['data_root']
        self.model_save_suffix = config['model_save_name']

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

        # self.crf = CRF(self.tags_size, self.tag_index_dict)
        # self.criterion = self.crf.neg_log_likelihood
        self.criterion = torch.nn.NLLLoss()

        self.optimizer = torch.optim.Adam

        # 标签和词典操作
        self.tag_index_dict = DataProcessor(**self.config).load_tags()
        self.tags_size = len(self.tag_index_dict)
        self.vocab = DataProcessor(**self.config).load_vocab()
        self.vocab_size = len(self.vocab)
        self.padding_idx = self.vocab[self.pad_token]
        self.padding_idx_tags = self.tag_index_dict[self.pad_token]

        # Bert模型
        self.checkpoint = config['checkpoint']
        self.tokenizer = BertTokenizer.from_pretrained(self.checkpoint)
        model_config = BertConfig.from_pretrained(self.checkpoint)
        model_config.output_hidden_states = True
        model_config.output_attentions = True
        self.bert_model = BertModel.from_pretrained(self.checkpoint, config=model_config)
        self.concat_to_hidden = nn.Linear(model_config.hidden_size * 4, model_config.hidden_size)

        # 基础模型
        self.dropout = nn.Dropout(p=self.dropout_p)
        self.relu = nn.ReLU()
        # MLP
        self.fc = nn.Sequential(nn.Linear(self.embedding_dim, self.hidden_dim), self.relu, self.dropout)
        self.linear_output_to_tag = nn.Linear(self.hidden_dim, self.tags_size)

        # LSTM
        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim, num_layers=self.layers,
                            bidirectional=self.bidirectional, batch_first=True)
        self.rnn_output_to_tag = nn.Linear(self.hidden_dim * self.n_directions, self.tags_size)

        # CNN
        self.window_sizes = config['window_sizes']
        self.out_channels = config['out_channels']

        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=self.embedding_dim,
                                    out_channels=self.out_channels,
                                    kernel_size=h, padding=(int((h - 1) / 2))),
                          # nn.BatchNorm1d(num_features=self.out_channels),
                          nn.ReLU())
            for h in self.window_sizes
        ])
        self.cnn_output_to_tag = nn.Linear(in_features=self.out_channels * len(self.window_sizes),
                                           out_features=self.tags_size)

        # Multiheads Self-Attention
        self.nums_head = config['nums_head']
        self.input_dim = self.embedding_dim
        self.dim_k = self.embedding_dim
        self.dim_v = self.embedding_dim
        assert self.dim_k % self.nums_head == 0
        assert self.dim_v % self.nums_head == 0
        # 定义WQ、WK、WV矩阵
        self.q = nn.Linear(self.input_dim, self.dim_k)
        self.k = nn.Linear(self.input_dim, self.dim_k)
        self.v = nn.Linear(self.input_dim, self.dim_v)
        self._norm_fact = 1 / math.sqrt(self.dim_k)

        self.att_output_to_tag = nn.Linear(self.dim_v, self.tags_size)

        # print('完成类{}的初始化'.format(self.__class__.__name__))

    def _lstm_init_hidden(self, batch_size):
        hidden = (torch.zeros(self.layers * self.n_directions, batch_size, self.hidden_dim).to(device),
                  torch.zeros(self.layers * self.n_directions, batch_size, self.hidden_dim).to(device))
        return hidden

    def _init_hidden(self, batch_size):
        return torch.zeros(self.layers * self.n_directions, batch_size, self.hidden_dim).to(device)

    # 根据列表内容，生成描述其成员长度的list
    def list_to_length(self, input):
        seq_list = []
        for line in input:
            seq_list.append(line.strip().split())
        lenth_list = [len(seq) for seq in seq_list]
        return lenth_list

    # 将index转换为token
    # def index_to_tag(self, y):
    #     index_tag_dict = dict(zip(self.tag_index_dict.values(), self.tag_index_dict.keys()))
    #     y_tagseq = []
    #     for i in range(len(y)):
    #         y_tagseq.append(index_tag_dict[y[i]])
    #     return y_tagseq

    # 将index转换为token，并依据y_len(原始数据长度)除去[PAD]
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

    def index_to_vocab(self, x):
        index_vocab_dict = dict(zip(self.vocab.values(), self.vocab.keys()))
        x_vocabseq = []
        for i in range(len(x)):
            x_vocabseq.append(index_vocab_dict[x[i]])
        return x_vocabseq

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

    def del_special_token(self, batch_list, batch_tensor):
        batch_size, seq_len, embed_dim = batch_tensor.shape
        seq_lengths = [int(len(st) / 2) for st in batch_list]

        # 按句取出，除去句中表示[CLS], [SEP]的向量
        tmp = []
        for i in range(batch_size):
            # 不含PAD的序列（一批中最长的序列）
            if seq_lengths[i] + 2 == seq_len:
                newt = batch_tensor[i][1:-1, :]
                tmp.append(newt)
            # 对于含有PAD的序列将其分为两部分处理，即[CLS]seq[SEP],[PAD]部分
            else:
                part1 = batch_tensor[i][1:seq_lengths[i] + 1, :]
                part2 = batch_tensor[i][seq_lengths[i] + 2:, :]
                newt = torch.cat((part1, part2), 0)
                tmp.append(newt)

        # 重新拼接为表示batch的张量
        new_batch_tensor = tmp[0].unsqueeze(0)
        if len(tmp) > 1:
            for i in range(len(tmp) - 1):
                new_batch_tensor = torch.cat((new_batch_tensor, tmp[i + 1].unsqueeze(0)), 0)

        return new_batch_tensor

    def get_token_embedding(self, batch, method):
        self.bert_model.to(device)
        # self.bert_model.eval()
        #
        # with torch.no_grad():
        if method == 0:
            # 使用“last_hidden_state”
            # last_hidden_state.shape = (batch_size, sequence_length, hidden_size)
            embedded = self.bert_model(**batch).last_hidden_state

        # 使用hidden_states。hidden_states是一个元组，第一个元素是embedding，其余元素是各层的输出，
        # 每个元素的形状是(batch_size, sequence_length, hidden_size)。
        elif method == 1:
            # 第1种方式：使用“initial embedding outputs”
            embedded = self.bert_model(**batch).hidden_states[0]

        elif method == 2:
            # 第2种方式：使用隐藏层输出sum last 4 layers
            hidden_states = self.bert_model(**batch).hidden_states

            layers = len(hidden_states)
            batch_size, seq_len, embed_dim = hidden_states[-1].size()
            # print(layers, batch_size, seq_len, embed_dim)

            batch_embedding = []
            for batch_i in range(batch_size):
                token_embeddings = []
                for token_i in range(seq_len):
                    hidden_layers = []
                    for layer_i in range(layers):
                        vec = hidden_states[layer_i][batch_i][token_i]
                        hidden_layers.append(vec)
                    token_embeddings.append(hidden_layers)
                # 连接最后四层 [number_of_tokens, 3072]
                # concatenated_last_4_layers = [torch.cat((layer[-1], layer[-2], layer[-3], layer[-4]), 0) for layer in
                #                               token_embeddings]
                # batch_embedding.append(torch.stack((concatenated_last_4_layers)))

                # 对最后四层求和 [number_of_tokens, 768]
                summed_last_4_layers = [torch.sum(torch.stack(layer)[-4:], 0) for layer in token_embeddings]
                batch_embedding.append(torch.stack((summed_last_4_layers)))

            embedded = torch.stack((batch_embedding))
        else:
            # 第3种方式：使用隐藏层输出concat last 4 layers
            hidden_states = self.bert_model(**batch).hidden_states

            layers = len(hidden_states)
            batch_size, seq_len, embed_dim = hidden_states[-1].size()
            # print(layers, batch_size, seq_len, embed_dim)

            batch_embedding = []
            for batch_i in range(batch_size):
                token_embeddings = []
                for token_i in range(seq_len):
                    hidden_layers = []
                    for layer_i in range(layers):
                        vec = hidden_states[layer_i][batch_i][token_i]
                        hidden_layers.append(vec)
                    token_embeddings.append(hidden_layers)
                # 连接最后四层 [number_of_tokens, 3072]
                concatenated_last_4_layers = [torch.cat((layer[-1], layer[-2], layer[-3], layer[-4]), 0) for layer
                                              in
                                              token_embeddings]
                batch_embedding.append(torch.stack((concatenated_last_4_layers)))

            embedded = self.concat_to_hidden(torch.stack((batch_embedding)))

        return embedded

    def prtplot(self, y_axis_values):
        x_axis_values = np.arange(1, len(y_axis_values) + 1, 1)
        y_axis_values = np.array(y_axis_values)
        plt.plot(x_axis_values, y_axis_values)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid()
        plt.show()


class MLP(Init_Class):
    def __init__(self, **config):
        super().__init__(**config)

    def forward(self, seq_list):
        # BertModel embedding
        batch = self.tokenizer(seq_list, padding=True, truncation=True, return_tensors="pt").to(device)
        # get_token_embedding()的第2个参数用于选择embedding的方式
        # 0: 使用“last_hidden_state”
        # 1: 使用“initial embedding outputs”，即hidden_states[0]
        # 2: 使用隐藏层后4层输出的和
        # 3: 使用隐藏层后4层输出的concat
        embedded = self.get_token_embedding(batch, 0)
        # 从embedded中删除表示[CLS],[SEP]的向量
        embedded = self.del_special_token(seq_list, embedded)

        # MLP
        mlp_output = self.fc(embedded)
        mlp_output = self.linear_output_to_tag(mlp_output)
        mlp_output = mlp_output.reshape(-1, mlp_output.shape[2])
        mlp_tag_scores = F.log_softmax(mlp_output, dim=1)  # [batch_size*seq_len, tags_size]

        return mlp_tag_scores


class BiLSTM(Init_Class):
    def __init__(self, **config):
        super().__init__(**config)

    def forward(self, seq_list):
        # BertModel embedding
        batch = self.tokenizer(seq_list, padding=True, truncation=True, return_tensors="pt").to(device)
        embedded = self.get_token_embedding(batch, 0)
        embedded = self.del_special_token(seq_list, embedded)

        # BiLSTM
        batch_size = len(seq_list)
        hidden = self._lstm_init_hidden(batch_size)
        bilstm_output, _ = self.lstm(embedded, hidden)

        bilstm_output = self.rnn_output_to_tag(bilstm_output)
        bilstm_output = bilstm_output.reshape(-1, bilstm_output.shape[2])
        bilstm_tag_scores = F.log_softmax(bilstm_output, dim=1)  # [batch_size*seq_len, tags_size]

        return bilstm_tag_scores


class CNN(Init_Class):
    def __init__(self, **config):
        super().__init__(**config)

    def forward(self, seq_list):
        # BertModel embedding
        batch = self.tokenizer(seq_list, padding=True, truncation=True, return_tensors="pt").to(device)
        embedded = self.get_token_embedding(batch, 0)
        embedded = self.del_special_token(seq_list, embedded)

        # CNN
        # [batch_size, seq_len, embedding_dim]  -> [batch_size, embedding_dim, seq_len]
        embedded = embedded.permute(0, 2, 1)
        cnn_output = [conv(embedded) for conv in self.convs]  # out[i]: [batch_size, self.out_channels, 1]
        cnn_output = torch.cat(cnn_output, dim=1)  # 对应第⼆个维度（⾏）拼接起来，⽐如说5*2*1,5*3*1的拼接变成5*5*1
        cnn_output = cnn_output.transpose(1, 2)
        cnn_output = self.cnn_output_to_tag(cnn_output)

        cnn_output = cnn_output.reshape(-1, cnn_output.shape[2])
        cnn_tag_scores = F.log_softmax(cnn_output, dim=1)  # [batch_size*seq_len, tags_size]

        return cnn_tag_scores


class Self_Attn_MH(Init_Class):
    def __init__(self, **config):
        super().__init__(**config)

    def forward(self, seq_list):
        # BertModel embedding
        batch = self.tokenizer(seq_list, padding=True, truncation=True, return_tensors="pt").to(device)
        embedded = self.get_token_embedding(batch, 0)
        embedded = self.del_special_token(seq_list, embedded)

        # MultiHeads self-attention
        Q = self.q(embedded).reshape(-1, embedded.shape[0], embedded.shape[1], self.dim_k // self.nums_head)
        K = self.k(embedded).reshape(-1, embedded.shape[0], embedded.shape[1], self.dim_k // self.nums_head)
        V = self.v(embedded).reshape(-1, embedded.shape[0], embedded.shape[1], self.dim_v // self.nums_head)

        atten = nn.Softmax(dim=-1)(torch.matmul(Q, K.permute(0, 1, 3, 2)))  # Q * K.T() # batch_size * seq_len * seq_len

        att_output = torch.matmul(atten, V).reshape(embedded.shape[0], embedded.shape[1],
                                                    -1)  # Q * K.T() * V # batch_size * seq_len * dim_v
        att_output = self.att_output_to_tag(att_output)
        att_output = att_output.reshape(-1, att_output.shape[2])  # 降维为[batch_size*seq_len, tags_size]
        att_tag_scores = F.log_softmax(att_output, dim=1)  # [batch_size*seq_len, tags_size]

        return att_tag_scores


class Vote(Init_Class):
    def __init__(self, **config):
        super().__init__(**config)
        self.model_mlp = MLP(**config)
        self.model_bilstm = BiLSTM(**config)
        self.model_cnn = CNN(**config)
        self.model_self_attn = Self_Attn_MH(**config)

    def run_train(self, model, data_path):
        print('Running {} model. Training...'.format(self.model_name))
        run_mode = 'train'

        self.model_mlp.to(device)
        self.model_bilstm.to(device)
        self.model_cnn.to(device)
        self.model_self_attn.to(device)

        mlp_optimizer = self.optimizer(self.model_mlp.parameters(), lr=self.learning_rate)
        bilstm_optimizer = self.optimizer(self.model_bilstm.parameters(), lr=self.learning_rate)
        cnn_optimizer = self.optimizer(self.model_cnn.parameters(), lr=self.learning_rate)
        self_attn_optimizer = self.optimizer(self.model_self_attn.parameters(), lr=self.learning_rate)

        self.model_mlp.train()
        self.model_bilstm.train()
        self.model_cnn.train()
        self.model_self_attn.train()

        train_losses = []  # 记录每一batch的loss
        valid_losses = []
        valid_f1_scores = []
        best_valid_loss = float('inf')
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
                batch_y = batch_y.view(-1)

                # 运行models
                mlp_tag_scores = self.model_mlp(seq_list)
                mlp_loss = self.criterion(mlp_tag_scores, batch_y)
                self.model_mlp.zero_grad()
                mlp_loss.backward()
                mlp_optimizer.step()

                bilstm_tag_scores = self.model_bilstm(seq_list)
                bilstm_loss = self.criterion(bilstm_tag_scores, batch_y)
                self.model_bilstm.zero_grad()
                bilstm_loss.backward()
                bilstm_optimizer.step()

                cnn_tag_scores = self.model_cnn(seq_list)
                cnn_loss = self.criterion(cnn_tag_scores, batch_y)
                self.model_cnn.zero_grad()
                cnn_loss.backward()
                cnn_optimizer.step()

                self_attn_tag_scores = self.model_self_attn(seq_list)
                self_attn_loss = self.criterion(self_attn_tag_scores, batch_y)
                self.model_self_attn.zero_grad()
                self_attn_loss.backward()
                self_attn_optimizer.step()

                train_loss += (mlp_loss.item() + bilstm_loss.item() + cnn_loss.item() + self_attn_loss.item())
                # print(train_data_num)

            train_losses.append(train_loss / train_data_num)  # 记录评价结果

            # 模型验证
            avg_eval_loss, eval_f1 = self.evaluate(model, data_path)
            print("Done Epoch{}. Eval_Loss={}".format(epoch + 1, avg_eval_loss))
            valid_losses.append(avg_eval_loss)
            valid_f1_scores.append(eval_f1)

            # 保存参数最优的模型
            if eval_f1 > best_valid_f1:
                best_valid_f1 = eval_f1
                torch.save(self.model_mlp, './model/vote_mlp.ckpt')
                torch.save(self.model_bilstm, './model/vote_bilstm.ckpt')
                torch.save(self.model_cnn, './model/vote_cnn.ckpt')
                torch.save(self.model_self_attn, './model/vote_self_attn.ckpt')

            # if avg_eval_loss < best_valid_loss:
            #     best_valid_loss = avg_eval_loss
            #     model_save_path = '{}{}_{}'.format(self.data_root, self.model_name, self.model_save_path)
            #     torch.save(model, '{}'.format(model_save_path))

        # self.prtplot(train_losses)
        # self.prtplot(valid_losses)

        return train_losses, valid_losses, valid_f1_scores

    def evaluate(self, model, data_path):
        # print('Running {} model. Evaluating...'.format(self.model_name))
        run_mode = 'eval'

        avg_loss, y_true, y_predict = self.eval_process(run_mode, data_path)
        f1_score = Evaluator().f1score(y_true, y_predict)

        return avg_loss, f1_score

    def test(self, data_path):
        print('Running {} model. Testing data in folder: {}'.format(self.model_name, data_path))
        run_mode = 'test'

        self.model_mlp = torch.load('./model/vote_mlp.ckpt')
        self.model_bilstm = torch.load('./model/vote_bilstm.ckpt')
        self.model_cnn = torch.load('./model/vote_cnn.ckpt')
        self.model_self_attn = torch.load('./model/vote_self_attn.ckpt')

        avg_loss, y_true, y_predict = self.eval_process(run_mode, data_path)

        # 输出评价结果
        print(Evaluator().classifyreport(y_true, y_predict))
        f1_score = Evaluator().f1score(y_true, y_predict)

        # 输出混淆矩阵
        metrix = Metrics(y_true, y_predict)
        metrix.report_scores()
        metrix.report_confusion_matrix()

        return f1_score

    def eval_process(self, run_mode, data_path):
        self.model_mlp.to(device)
        self.model_bilstm.to(device)
        self.model_cnn.to(device)
        self.model_self_attn.to(device)

        self.model_mlp.eval()
        self.model_bilstm.eval()
        self.model_cnn.eval()
        self.model_self_attn.eval()

        with torch.no_grad():
            total_loss = 0
            data_num = 0
            all_y_predict = []
            all_mlp_predict = []
            all_bilstm_predict = []
            all_cnn_predict = []
            all_self_attn_predict = []
            all_y_true = []
            for seq_list, tag_list in DataLoader(**self.config).data_generator(data_path=data_path,
                                                                               run_mode=run_mode):
                data_num += len(seq_list)
                # 生成batch_y
                batch_output = self.tag_to_index(tag_list)
                y = self.pad_taglist(batch_output)
                batch_y = torch.tensor(y).long().to(device)
                batch_y = batch_y.view(-1)

                # 运行models
                mlp_tag_scores = self.model_mlp(seq_list)
                mlp_loss = self.criterion(mlp_tag_scores, batch_y)

                bilstm_tag_scores = self.model_bilstm(seq_list)
                bilstm_loss = self.criterion(bilstm_tag_scores, batch_y)

                cnn_tag_scores = self.model_cnn(seq_list)
                cnn_loss = self.criterion(cnn_tag_scores, batch_y)

                self_attn_tag_scores = self.model_self_attn(seq_list)
                self_attn_loss = self.criterion(self_attn_tag_scores, batch_y)

                total_loss += (mlp_loss.item() + bilstm_loss.item() + cnn_loss.item() + self_attn_loss.item())

                # 返回每一行中最大值的索引
                mlp_predict = torch.max(mlp_tag_scores, dim=1)[1]  # 1维张量
                mlp_y_predict = list(mlp_predict.cpu().numpy())
                mlp_y_predict = self.index_to_tag(mlp_y_predict, self.list_to_length(tag_list))
                all_mlp_predict = all_mlp_predict + mlp_y_predict

                bilstm_predict = torch.max(bilstm_tag_scores, dim=1)[1]  # 1维张量
                bilstm_y_predict = list(bilstm_predict.cpu().numpy())
                bilstm_y_predict = self.index_to_tag(bilstm_y_predict, self.list_to_length(tag_list))
                all_bilstm_predict = all_bilstm_predict + bilstm_y_predict

                cnn_predict = torch.max(cnn_tag_scores, dim=1)[1]  # 1维张量
                cnn_y_predict = list(cnn_predict.cpu().numpy())
                cnn_y_predict = self.index_to_tag(cnn_y_predict, self.list_to_length(tag_list))
                all_cnn_predict = all_cnn_predict + cnn_y_predict

                self_attn_predict = torch.max(self_attn_tag_scores, dim=1)[1]  # 1维张量
                self_attn_y_predict = list(self_attn_predict.cpu().numpy())
                self_attn_y_predict = self.index_to_tag(self_attn_y_predict, self.list_to_length(tag_list))
                all_self_attn_predict = all_self_attn_predict + self_attn_y_predict

                y_true = y.flatten()
                y_true = self.index_to_tag(y_true, self.list_to_length(tag_list))
                all_y_true = all_y_true + y_true
                # print(list(set(all_y_true)))

            # 投票表决
            mlp_len = len(all_mlp_predict)
            bilstm_len = len(all_bilstm_predict)
            cnn_len = len(all_cnn_predict)
            self_attn_len = len(all_self_attn_predict)
            if mlp_len == bilstm_len and bilstm_len == cnn_len and cnn_len == self_attn_len:
                for i in range(mlp_len):
                    labels = []
                    labels.append(all_mlp_predict[i])
                    labels.append(all_bilstm_predict[i])
                    labels.append(all_cnn_predict[i])
                    labels.append(all_self_attn_predict[i])

                    unique_labels = set(labels)  # set集合去重
                    label_num = {}
                    for label in unique_labels:
                        label_num[label] = labels.count(label)

                    # 对字典按value排序，返回值为list
                    label_num_sorted = sorted(label_num.items(), key=lambda x: x[1], reverse=True)

                    if len(label_num_sorted) == 1 or len(label_num_sorted) == 3 or (
                            len(label_num_sorted) == 2 and label_num_sorted[0][1] > label_num_sorted[1][1]):
                        all_y_predict.append(label_num_sorted[0][0])
                    else:
                        all_y_predict.append(all_bilstm_predict[i])
                # print(list(set(all_y_predict)))

            return total_loss / data_num, all_y_true, all_y_predict


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
