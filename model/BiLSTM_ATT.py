#coding:utf8
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(1)

class BiLSTM_ATT(nn.Module):
    def __init__(self,config,embedding_pre):
        super(BiLSTM_ATT,self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.batch = config['BATCH']
        
        self.embedding_size = config['EMBEDDING_SIZE']
        self.embedding_dim = config['EMBEDDING_DIM']
        
        self.hidden_dim = config['HIDDEN_DIM']
        self.tag_size = config['TAG_SIZE']
        
        self.pos_size = config['POS_SIZE']
        self.pos_dim = config['POS_DIM']
        
        self.pretrained = config['pretrained']
        if self.pretrained:
            #self.word_embeds.weight.data.copy_(torch.from_numpy(embedding_pre))
            self.word_embeds = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_pre),freeze=False)
        else:
            self.word_embeds = nn.Embedding(self.embedding_size,self.embedding_dim)
        
        self.pos1_embeds = nn.Embedding(self.pos_size,self.pos_dim)
        self.pos2_embeds = nn.Embedding(self.pos_size,self.pos_dim)
        self.relation_embeds = nn.Embedding(self.tag_size,self.hidden_dim)
        
        self.lstm = nn.LSTM(input_size=self.embedding_dim+self.pos_dim*2,hidden_size=self.hidden_dim//2,num_layers=1, bidirectional=True).to(self.device)
        self.hidden2tag = nn.Linear(self.hidden_dim,self.tag_size)
        
        self.dropout_emb=nn.Dropout(p=0.3)
        self.dropout_lstm=nn.Dropout(p=0.5)
        self.dropout_att=nn.Dropout(p=0.5)
        
        self.hidden = self.init_hidden()
        
        self.att_weight = nn.Parameter(torch.randn(self.batch,1,self.hidden_dim)).to(self.device)
        self.relation_bias = nn.Parameter(torch.randn(self.batch,self.tag_size,1)).to(self.device)
        
    def init_hidden(self):
        return torch.randn(2, self.batch, self.hidden_dim // 2)
        
    def init_hidden_lstm(self):
        return (torch.randn(2, self.batch, self.hidden_dim // 2).to(self.device),
                torch.randn(2, self.batch, self.hidden_dim // 2).to(self.device))
                
    def attention(self,H):
        """
        :param H: [128, 200, 50]: [batch_size, hidden_size, sentence_lengths]
        :return:  [128, 200, 1]: [batch_size, hidden_size, 1]
        """
        M = F.tanh(H)   # M: torch.Size([128, 200, 50])

        a = F.softmax(torch.bmm(self.att_weight,M),2)   # a: torch.Size([128, 1, 50])

        a = torch.transpose(a,1,2)      # a: torch.Size([128, 50, 1])

        return torch.bmm(H,a)
        
    
                
    def forward(self,sentence,pos1,pos2):

        self.hidden = self.init_hidden_lstm()   # sentence: torch.Size([128, 50])

        embeds = torch.cat((self.word_embeds(sentence),self.pos1_embeds(pos1),self.pos2_embeds(pos2)),2)
        
        embeds = torch.transpose(embeds,0,1).to(self.device)    # embeds: torch.Size([50, 128, 150])

        embeds = self.dropout_emb (embeds)

        lstm_out, self.hidden = self.lstm(embeds, self.hidden)  # lstm out: torch.Size([50, 128, 200])
        
        lstm_out = torch.transpose(lstm_out,0,1)
        lstm_out = torch.transpose(lstm_out,1,2)    # lstm out: torch.Size([128, 200, 50])

        lstm_out = self.dropout_lstm(lstm_out)
        att_out = F.tanh(self.attention(lstm_out))  # att_out: torch.Size([128, 200, 1])
        #att_out = self.dropout_att(att_out)
        
        relation = torch.tensor([i for i in range(self.tag_size)],dtype = torch.long).repeat(self.batch, 1).to(self.device) # 循环了128遍的1234。。。12

        relation = self.relation_embeds(relation)   # hidden_size, relation: torch.Size([128, 12, 200])

        res = torch.add(torch.bmm(relation,att_out), self.relation_bias)     # res: torch.Size([128, 12, 1])

        res = F.softmax(res,1)      # res: torch.Size([128, 12, 1])


        return res.view(self.batch,-1)
