# -*- coding: utf-8 -*-
# @Auther   : liou

# coding:utf8
import numpy as np
import pickle
import sys
import codecs
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as D
from torch.autograd import Variable
from BiLSTM_ATT import BiLSTM_ATT

# with open('./data/engdata_train.pkl', 'rb') as inp:
with open('./data/people_relation_train.pkl', 'rb') as inp:
    word2id = pickle.load(inp)
    id2word = pickle.load(inp)
    relation2id = pickle.load(inp)


EMBEDDING_SIZE = len(word2id) + 1
EMBEDDING_DIM = 100

POS_SIZE = 82  # 不同数据集这里可能会报错。
POS_DIM = 25

HIDDEN_DIM = 200

TAG_SIZE = len(relation2id)

BATCH = 128
EPOCHS = 100

config = {}
config['EMBEDDING_SIZE'] = EMBEDDING_SIZE
config['EMBEDDING_DIM'] = EMBEDDING_DIM
config['POS_SIZE'] = POS_SIZE
config['POS_DIM'] = POS_DIM
config['HIDDEN_DIM'] = HIDDEN_DIM
config['TAG_SIZE'] = TAG_SIZE
config['BATCH'] = BATCH
config["pretrained"] = False

class trainer (object) :
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = 0.0005
        self.config = config


        self.embedding_pre = []
        if config["pretrained"] == True:
            self.get_pre_()

        self.model = BiLSTM_ATT(config, self.embedding_pre).to(self.device)
        # model = torch.load('model/model_epoch20.pkl')
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        self.criterion = nn.CrossEntropyLoss(size_average=True)

    def get_pre_ (self) :
        print("use pretrained embedding")
        word2vec = {}
        with codecs.open('vec.txt', 'r', 'utf-8') as input_data:
            for line in input_data.readlines():
                word2vec[line.split()[0]] = list(map(eval, line.split()[1:]))

        unknow_pre = []
        unknow_pre.extend([1] * 100)
        self.embedding_pre.append(unknow_pre)  # wordvec id 0
        for word in word2id:
            if word in word2vec:
                self.embedding_pre.append(word2vec[word])
            else:
                self.embedding_pre.append(unknow_pre)

        embedding_pre = np.asarray(self.embedding_pre)
        print(embedding_pre.shape)

    def get_data (self) :

        with open('./data/people_relation_train.pkl', 'rb') as inp:
            word2id = pickle.load(inp)
            id2word = pickle.load(inp)
            relation2id = pickle.load(inp)
            train = pickle.load(inp)
            labels = pickle.load(inp)
            position1 = pickle.load(inp)
            position2 = pickle.load(inp)

        # with open('./data/engdata_test.pkl', 'rb') as inp:
        with open('./data/people_relation_test.pkl', 'rb') as inp:
            test = pickle.load(inp)
            labels_t = pickle.load(inp)
            position1_t = pickle.load(inp)
            position2_t = pickle.load(inp)

        print("train len", len(train))
        print("test len", len(test))
        print("word2id len", len(word2id))
        train = torch.LongTensor(train[:len(train) - len(train) % BATCH])
        position1 = torch.LongTensor(position1[:len(train) - len(train) % BATCH])
        position2 = torch.LongTensor(position2[:len(train) - len(train) % BATCH])
        labels = torch.LongTensor(labels[:len(train) - len(train) % BATCH])
        train_datasets = D.TensorDataset(train, position1, position2, labels)
        train_dataloader = D.DataLoader(train_datasets, BATCH, True, num_workers=2)

        test = torch.LongTensor(test[:len(test) - len(test) % BATCH])
        position1_t = torch.LongTensor(position1_t[:len(test) - len(test) % BATCH])
        position2_t = torch.LongTensor(position2_t[:len(test) - len(test) % BATCH])
        labels_t = torch.LongTensor(labels_t[:len(test) - len(test) % BATCH])
        test_datasets = D.TensorDataset(test, position1_t, position2_t, labels_t)
        test_dataloader = D.DataLoader(test_datasets, BATCH, True, num_workers=2)

        return train_dataloader, test_dataloader


    def train (self) :
        train_dataloader, test_dataloader = self.get_data()

        for epoch in range(EPOCHS):
            print("epoch:", epoch)
            acc = 0
            total = 0

            for sentence, pos1, pos2, tag in train_dataloader:
                sentence = Variable(sentence).to(self.device)
                pos1 = Variable(pos1).to(self.device)
                pos2 = Variable(pos2).to(self.device)
                y = self.model(sentence, pos1, pos2)
                tags = Variable(tag).to(self.device)
                loss = self.criterion(y, tags).to(self.device)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                y = np.argmax(y.cpu().data.numpy(), axis=1)

                for y1, y2 in zip(y, tag):
                    if y1 == y2:
                        acc += 1
                    total += 1

            print("train:",100*float(acc)/total,"%")

            acc_t = 0
            total_t = 0
            count_predict = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            count_total = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            count_right = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            for sentence, pos1, pos2, tag in test_dataloader:
                sentence = Variable(sentence).to(self.device)
                pos1 = Variable(pos1).to(self.device)
                pos2 = Variable(pos2).to(self.device)
                y = self.model(sentence, pos1, pos2)
                y = np.argmax(y.cpu().data.numpy(), axis=1)
                for y1, y2 in zip(y, tag):
                    count_predict[y1] += 1
                    count_total[y2] += 1
                    if y1 == y2:
                        count_right[y1] += 1

            precision = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            recall = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            for i in range(len(count_predict)):
                if count_predict[i] != 0:
                    precision[i] = float(count_right[i]) / count_predict[i]

                if count_total[i] != 0:
                    recall[i] = float(count_right[i]) / count_total[i]

            precision = sum(precision) / len(relation2id)
            recall = sum(recall) / len(relation2id)
            print("准确率：", precision)
            print("召回率：", recall)
            print("f：", (2 * precision * recall) / (precision + recall))

            if epoch % 20 == 0:
                model_name = "./model/model_epoch" + str(epoch) + ".pkl"
                torch.save(self.model, model_name)
                print(model_name,"has been saved")

        torch.save(self.model, "./model/model_01.pkl")
        print("model has been saved")

if __name__ == '__main__':
    T = trainer()
    T.train()