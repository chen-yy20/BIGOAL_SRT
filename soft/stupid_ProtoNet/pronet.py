#encoding=utf-8
# source code from : https://blog.csdn.net/hei653779919/article/details/106595614

import torch
import torch.nn as nn
import torch.nn.functional as F
import jieba
import random
import torch.optim as optim



def createData():
    text_list_pos = ["电影内容很好","电影题材很好","演员演技很好","故事很感人","电影特效很好"]
    text_list_neg = ["电影内容垃圾","电影是真的垃圾","表演太僵硬了","故事又臭又长","电影太让人失望了"]
    test_pos = ["电影","很","好"]
    test_neg = ["电影","垃圾"]
    words_pos = [[item for item in jieba.cut(text)] for text in text_list_pos]
    words_neg = [[item for item in jieba.cut(text)] for text in text_list_neg]
    words_all = []
    for item in words_pos:
        for key in item:
            words_all.append(key)
    for item in words_neg:
        for key in item:
            words_all.append(key)
    vocab = list(set(words_all))
    word2idx = {w:c for c,w in enumerate(vocab)}
    idx_words_pos = [[word2idx[item] for item in text] for text in words_pos]
    idx_words_neg = [[word2idx[item] for item in text] for text in words_neg]
    idx_test_pos = [word2idx[item] for item in test_pos]
    idx_test_neg = [word2idx[item] for item in test_neg]
    return vocab,word2idx,idx_words_pos,idx_words_neg,idx_test_pos,idx_test_neg
def createOneHot(vocab,idx_words_pos,idx_words_neg,idx_test_pos,idx_test_neg):
    input_dim = len(vocab)
    features_pos = torch.zeros(size=[len(idx_words_pos),input_dim])
    features_neg = torch.zeros(size=[len(idx_words_neg), input_dim])
    for i in range(len(idx_words_pos)):
        for j in idx_words_pos[i]:
            features_pos[i,j] = 1.0

    for i in range(len(idx_words_neg)):
        for j in idx_words_neg[i]:
            features_neg[i,j] = 1.0
    features = torch.cat([features_pos,features_neg],dim=0)
    labels = [1,1,1,1,1,0,0,0,0,0]
    labels = torch.LongTensor(labels)
    test_x_pos = torch.zeros(size=[1,input_dim])
    test_x_neg = torch.zeros(size=[1,input_dim])
    for item in idx_test_pos:
        test_x_pos[0,item] = 1.0
    for item in idx_test_neg:
        test_x_neg[0,item] = 1.0
    test_x = torch.cat([test_x_pos,test_x_neg],dim=0)
    test_labels = torch.LongTensor([1,0])
    return features,labels,test_x,test_labels
def randomGenerate(features):
    N = features.shape[0]
    half_n = N // 2
    support_input = torch.zeros(size=[6, features.shape[1]])
    query_input = torch.zeros(size=[4,features.shape[1]])
    postive_list = list(range(0,half_n))
    negtive_list = list(range(half_n,N))
    support_list_pos = random.sample(postive_list,3)
    support_list_neg = random.sample(negtive_list,3)
    query_list_pos = [item for item in postive_list if item not in support_list_pos]
    query_list_neg = [item for item in negtive_list if item not in support_list_neg]
    index = 0
    for item in support_list_pos:
        support_input[index,:] = features[item,:]
        index += 1
    for item in support_list_neg:
        support_input[index,:] = features[item,:]
        index += 1
    index = 0
    for item in query_list_pos:
        query_input[index,:] = features[item,:]
        index += 1
    for item in query_list_neg:
        query_input[index,:] = features[item,:]
        index += 1
    query_label = torch.LongTensor([1,1,0,0])
    return support_input,query_input,query_label




class fewModel(nn.Module):
    def __init__(self,input_dim,hidden_dim,num_class):
        super(fewModel,self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_class = num_class
        # 线性层进行编码
        self.linear = nn.Linear(input_dim,hidden_dim)


    def embedding(self,features):
        result = self.linear(features)
        return result

    def forward(self,support_input,query_input):

        support_embedding = self.embedding(support_input)
        query_embedding = self.embedding(query_input)
        support_size = support_embedding.shape[0]
        every_class_num  = support_size // self.num_class
        class_meta_dict = {}
        for i in range(0,self.num_class):
            class_meta_dict[i] = torch.sum(support_embedding[i*every_class_num:(i+1)*every_class_num,:],dim=0) / every_class_num
        class_meta_information = torch.zeros(size=[len(class_meta_dict),support_embedding.shape[1]])
        for key,item in class_meta_dict.items():
            class_meta_information[key,:] = class_meta_dict[key]
        N_query = query_embedding.shape[0]
        result = torch.zeros(size=[N_query,self.num_class])
        for i in range(0,N_query):
            temp_value = query_embedding[i].repeat(self.num_class,1)
            cosine_value = torch.cosine_similarity(class_meta_information,temp_value,dim=1)
            result[i] = cosine_value
        result = F.log_softmax(result,dim=1)
        return result

hidden_dim = 4
n_class = 2
lr = 0.01
epochs = 1000
vocab,word2idx,idx_words_pos,idx_words_neg,idx_test_pos,idx_test_neg = createData()
features,labels,test_x,test_labels = createOneHot(vocab,idx_words_pos,idx_words_neg,idx_test_pos,idx_test_neg)

model = fewModel(features.shape[1],hidden_dim,n_class)
optimer = optim.Adam(model.parameters(),lr=lr,weight_decay=5e-4)

def train(epoch,support_input,query_input,query_label):
    optimer.zero_grad()
    output = model(support_input,query_input)
    loss = F.nll_loss(output,query_label)
    loss.backward()
    optimer.step()
    print("Epoch: {:04d}".format(epoch),"loss:{:.4f}".format(loss))

if __name__ == '__main__':
    for i in range(epochs):
        support_input, query_input, query_label = randomGenerate(features)
        train(i,support_input,query_input,query_label)


 
