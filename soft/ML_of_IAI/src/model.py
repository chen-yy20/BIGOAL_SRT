import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TextCnn(nn.Module):
    def __init__(self,args):
        self.args = args
        super(TextCnn, self).__init__()
        self.embed_dim =50
        class_num = 2
        Ci = 1
        Co = args.kernel_num
        # 已经有词向量故不需要嵌入
        kernel_sizes = [int(fsz) for fsz in args.kernel_sizes.split(',')]
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (f, self.embed_dim), padding = (2, 0)) for f in kernel_sizes])
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(Co * len(kernel_sizes), class_num)

    def forward(self, xs):
        final = []
        for x in xs:
            # x的格式为 (token_num, embed_dim)
            x = x.unsqueeze(0)  # (N, Ci, token_num, embed_dim)
            x = x.unsqueeze(0) 
            x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, token_num) * len(kernel_sizes)]
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co) * len(kernel_sizes)]
            # kernel_size_num*kernel_num的向量
            x = torch.cat(x, 1) # (N, Co * len(kernel_sizes))
            x = self.dropout(x)  # (N, Co * len(kernel_sizes))
            logit = self.fc(x)  # (N, class_num)
            logit = logit.squeeze(0)
            final.append(logit)
        # final 最后是一个长度为64的列表batch
        return final

class TextLSTM(nn.Module):
    def __init__(self, args):
        super(TextLSTM, self).__init__()
        self.args = args
        self.hidden_dim = args.hidden_dim  # 隐藏层节点数
        self.num_layers = args.num_layers  # 神经元层数
        self.n_class = 2  # 类别数
        self.bidirectional = args.bidirectional  # 控制是否为双向LSTM
        # LSTM
        self.encoder = nn.LSTM(input_size=50, hidden_size=self.hidden_dim,
                               num_layers=self.num_layers, bidirectional=self.bidirectional,
                                dropout=args.dropout)
        if self.bidirectional:
            self.decoder1 = nn.Linear(self.hidden_dim * 4, self.hidden_dim)
            self.decoder2 = nn.Linear(self.hidden_dim, self.n_class)
        else:
            self.decoder1 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
            self.decoder2 = nn.Linear(self.hidden_dim, self.n_class)

    def forward(self, input):
        # input follows the form: [batch_size, seq_len, embed_dim]
        states, hidden = self.encoder(input.permute([1, 0, 2]))
        # states.shape= torch.Size([seq_len, batch_size, 200]) 输出的是一个h0和一个c0
        encoding = torch.cat([states[0], states[-1]], dim=1)
        # encoding.shape= torch.Size([batch_size, 400])
        outputs = self.decoder1(encoding)
        # outputs = F.softmax(outputs, dim=1)
        outputs = self.decoder2(outputs)
        return outputs

# 四层全连接层，就是来搞笑的
class MLP(nn.Module):
    def __init__(self,args):
        super(MLP, self).__init__()
        self.args = args
        self.n_class = 2
        self.fc1 = nn.Linear(119*50,512)
        self.fc2 = nn.Linear(512,128)
        self.fc3 = nn.Linear(128,32)
        self.fc4 = nn.Linear(32,self.n_class)
    def forward(self,input):
        # shape[batch_size,119,50]
        x = input.view(self.args.batch_size,119*50)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x





    