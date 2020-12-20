# -*- coding: utf-8 -*-
# @Auther   : liou

from os.path import join
from codecs import open

def build_corpus (split, make_vocab = True, data_dir = './data') :
    assert split in ['train', 'test','dev']

    relation2id = {}
    id2relation = {}
    with open(join(data_dir, "/relation/relation2id.txt"), 'r', 'utf-8') as input_data:
        for line in input_data.readlines():
            relation2id[line.split()[0]] = int(line.split()[1])
            id2relation[line.split()[1]] = int(line.split()[0])
        input_data.close()

    word_lists = []
    tag_lists = []
    pos1_lists = []
    pos2_lists = []
    count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    total_data = 0
    with open (join(data_dir, split + ".txt"), 'r', encoding='utf-8') as f:
        for line in f:
            if line != '\n' :
                lines = line.split()
                sentence = []
                index1 = line[3].index(line[0])
                position1 = []
                index2 = line[3].index(line[1])
                position2 = []
                for i, word in enumerate(line[3]):
                    sentence.append(word)
                    position1.append(i - index1)
                    position2.append(i - index2)
                    i += 1
                word_lists.append(sentence)
                tag_lists.append(line[2])
                pos1_lists.append(position1)
                pos2_lists.append(position2)
            total_data += 1
            count[relation2id[line[2]]] += 1
    for i in range (12) :
        print ("tag:", id2relation[i], " num:", count[i])

    print("tot:", total_data, len(word_lists))

    if make_vocab :
        word2id = build_corpus(word_lists)
        return word_lists, tag_lists, pos1_lists, pos2_lists, word2id, relation2id
    else :
        return word_lists, tag_lists, pos1_lists, pos2_lists

def build_map (lists) :
    maps = {}
    for list in lists :
        for e in list:
            if e not in maps :
                maps[e] = len (maps)
    return maps
