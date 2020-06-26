# -*- coding: utf-8 -*-
# @Auther   : liou

import torch
import torch.nn as nn
import torch.nn.functional as F

def extend_maps (word2id, tag2id) :
    word2id['<unk>'] = len (word2id)
    word2id['<pad>'] = len (word2id)
    tag2id['<unk>'] = len (tag2id)
    tag2id['<pad>'] = len (tag2id)

    return word2id, tag2id