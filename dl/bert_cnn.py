#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
model: Bert_CNN
======
A class for CNN using Bert embedding.
配置文件：cmed.dl.bert_cnn.norm.json
"""
import argparse
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

from sequence_labeling.dl.bert_mlp import Bert_MLP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)


class Bert_CNN(Bert_MLP):
    def __init__(self, **config):
        super().__init__(**config)

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
        self.output_to_tag = nn.Linear(in_features=self.out_channels * len(self.window_sizes),
                                       out_features=self.tags_size)

    def forward(self, seq_list):
        # BertModel embedding
        batch = self.tokenizer(seq_list, padding=True, truncation=True, return_tensors="pt").to(device)
        embedded = self.get_token_embedding(batch, 0)
        embedded = self.del_special_token(seq_list, embedded)  # 剔除[CLS], [SEP]标识

        # [batch_size, seq_len, embedding_dim]  -> [batch_size, embedding_dim, seq_len]
        embedded = embedded.permute(0, 2, 1)

        out = [conv(embedded) for conv in self.convs]  # out[i]: [batch_size, self.out_channels, 1]
        out = torch.cat(out, dim=1)  # 对应第⼆个维度（⾏）拼接起来，⽐如说5*2*1,5*3*1的拼接变成5*5*1
        out = out.transpose(1, 2)
        out = self.output_to_tag(out)

        output = out.reshape(-1, out.shape[2])
        tag_scores = F.log_softmax(output, dim=1)  # [batch_size*seq_len, tags_size]

        return tag_scores


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
