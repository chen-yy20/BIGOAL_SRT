# Src from https://www.zhihu.com/question/298810062

import torch
import torch.nn as nn
import math

class BertSelfAttention(nn.Module):
    def __init__(self):
        self.all_head_size = 768
        # Feature dimension of the word vector
        self.hidden_size = 768
        self.query = nn.Linear(self.hidden_size,self.all_head_size)
        self.key = nn.Linear(self.hidden_size,self.all_head_size)
        self.value = nn.Linear(self.hidden_size,self.all_head_size)
    def forward(self,hidden_states):
        Q = self.query(hidden_states)
        K = self.key(hidden_states)
        V = self.value(hidden_states)

        attention_scores = torch.matmul(Q,K.transpose(-1,-2))
        attention_scores = attention_scores/math.sqrt(self.all_head_size)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        out = torch.matmul(attention_probs,V)
        return out
