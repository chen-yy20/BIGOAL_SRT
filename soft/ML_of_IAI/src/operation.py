import os
import sys
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import random
import jieba
from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format("./Dataset/wiki_word2vec_50.bin",binary=True)

def get_transformed_data(dataset,args,datalabel):
    data_vec = []
    for batch in dataset:
        transformed_batch = []
        for data_list_raw in batch:
            data_list_vec = []
            for word in data_list_raw:
                try:
                    # print(word, model[word], model[word].shape)
                    data_list_vec.append(model[word])
                except:
                    # 以全0的词向量进行填充
                    data_list_vec.append([0 for i in range(50)])
                    #print(word+" not exist.")
            # print(data_list_vec)
            if args.mode == 'cnn':
                data_list_vec = torch.Tensor(data_list_vec)
            transformed_batch.append(data_list_vec)
        if args.mode != 'cnn':
            transformed_batch = torch.Tensor(transformed_batch)
        data_vec.append(transformed_batch)
    # for i in range(len(dataset[0])):
    #     sentence = ''.join(dataset[0][i])
    #     print(sentence,datalabel[0][i])

    print("Vec_dataset generated.")
    return data_vec


def get_data(mode,args):
    data_raw = []
    data_label = []
    f = open("./Dataset/"+ mode +".txt",encoding='utf-8')
    lines = f.readlines()
    random.shuffle(lines)
    i = 0
    k = 0
    batch = []
    batch_label = []
    max_kenel_size = int(max(args.kernel_sizes.split(',')))
    print(mode+"set_len:"+str(len(lines)))
    if args.test:
        args.batch_size = len(lines)
    for line in lines:
        line = line.replace('\t',' ')
        line = line.strip("\n")
        word_list = line.split(" ")
        if args.mode != 'cnn':
            # 这一步是为了统一词向量
            if len(word_list)>300:
                # 单独处理那个678长度的玩意：跳过
                continue
            while len(word_list)<120:
                word_list.append('')
        while len(word_list)<max_kenel_size+1:
            word_list.append('')
        # print(word_list)
        batch.append(word_list[1:])
        batch_label.append(torch.tensor(int(word_list[0])))
        # if word_list[0] == '0':
        #     batch_label.append(torch.Tensor([1,0]))
        # elif word_list[0] == '1':
        #     batch_label.append(torch.Tensor([0,1]))
        i+=1
        if(i==args.batch_size):
            data_raw.append(batch)
            data_label.append(batch_label)
            i = 0
            batch = []
            batch_label = []
            k+=1
        # if(k==3):
        #     break
    # print(train_data_raw[0],train_data_label[0])
    print("Train data and label loaded.")
    # print("=============================================================")
    # for i in range(len(data_label[0])):
    #     sentence = ''.join(data_raw[0][i])
    #     print(sentence,data_label[0][i])
    return get_transformed_data(data_raw,args,data_label),data_label

# dataset:[k,64,n,dim] n为可变长度，k为batch数据总量


def train(train_iter, dev_iter, model, args):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    steps = 0
    best_acc = 0
    lowest_loss = 10
    smaller_step = 0
    validation_data_vec, validation_label = get_data('validation',args)
    for epoch in range(1, args.epochs+1):
        for batch in range(len(train_iter)):
            feature = train_iter[batch]
            target = dev_iter[batch]
            # feature.t_(), target.sub_(1)  # batch first, index align
            optimizer.zero_grad()
            try:
                logit = model(feature)

                loss = 0
                for j in range(len(logit)):
                    loss += F.cross_entropy(logit[j], target[j])
                loss /= len(logit)
                loss.backward()
                optimizer.step()
                steps += 1
                if steps % 10 == 0:
                    correct = 0
                    # 计算此batch的准确率
                    for i in range(len(train_iter[batch])):
                        if (torch.max(logit[i], 0)[1] == target[i]):
                            correct += 1
                    accuracy = correct*100/len(train_iter[batch])
                    print('\rBatch[{}] - loss: {:.6f}  acc: {:.3f}%({}/{})'.format(steps,
                                                                                    loss.data,
                                                                                    accuracy,
                                                                                    correct,
                                                                                    len(train_iter[batch])))
            # 存在句子比卷积核小的情况
            except RuntimeError:
                print("Runtime Error")
                steps -= 1
                continue
        # 使用验证集
        print("---------------------------epoch"+str(epoch)+"--------------------")
        acc,loss_data = test(validation_data_vec,validation_label,model, args)
        print("acc:{:.3f},best_acc:{:.3f},loss:{:.6f},lowest_loss:{:.6f}".format(acc,best_acc,loss_data.data.float(),lowest_loss))
        if acc > best_acc or loss_data.data.float() < lowest_loss: 
            print("update.")         
            smaller_step = 0
            if acc >= best_acc:
                best_acc = acc
            if loss_data.data.float() < lowest_loss:
                lowest_loss = loss_data.data.float()        
        else:
            smaller_step += 1
            if smaller_step == 2:
                save(model,'./models/'+args.mode,'_best-acc_{:.3f}'.format(best_acc),steps,args.batch_size)
                return
    save(model, './models/'+args.mode, '_final-acc_{:.3f}'.format(best_acc), steps,args.batch_size)


def test(data_iter, label_iter, model, args):
    P,N,TP,FP = 0,0,0,0
    model.eval()
    correct, avg_loss, size = 0, 0, 0 
    for batch in range(len(data_iter)):
        feature, target = data_iter[batch], label_iter[batch]
        logit = model(feature)
        loss = 0
        for j in range(len(logit)):
            loss += F.cross_entropy(logit[j], target[j],size_average=False)
            size += 1
        avg_loss += loss.data
        # 计算此batch的准确率
        for i in range(len(data_iter[batch])):
            predict = torch.max(logit[i], 0)[1]
            if (predict == target[i]):
                correct += 1
            if target[i] == 1:
                P += 1
                if predict == 1:
                    TP += 1
            elif target[i] == 0:
                N += 1
                if predict == 1:
                    FP += 1
        accuracy = correct/len(data_iter[batch])

    avg_loss /= size
    accuracy = 100.0 * float(correct) / size
    precision = TP/(TP+FP)
    recall = TP/P
    F_score = 2/(1/precision+1/recall)
    print('Evaluation - loss: {:.6f}  acc: {:.3f}% ({}/{}) \n'.format(avg_loss,
                                                                        accuracy,
                                                                        correct,
                                                                        size))

    print('precision:{:.3f}, recall:{:.3f} => F_score:{:.3f}'.format(precision,recall,F_score))
    return accuracy,avg_loss


def predict(nnmodel,args):
    nnmodel.eval()
    sentence = input("输入你的句子：")
    word_list = jieba.cut(sentence,cut_all=True)
    # print("/".join(word_list))
    vec_list = []
    for word in word_list:
        try:
            vec_list.append(model[word])
        except:
            # 以全0的词向量进行填充
            vec_list.append([0 for i in range(50)])
    if args.mode == 'cnn':
        x = torch.Tensor(vec_list)
        xs = [x]
    if args.mode != 'cnn':
        x = [vec_list]
        xs = torch.Tensor(x)
    output = nnmodel(xs)
    print(output[0])
    predicted = torch.max(output[0], 0)[1]
    if predicted.int() == 0:
        print("正面:"+sentence)
    else:
        print("负面:"+sentence)



def save(model, save_dir, save_prefix, steps,batch_size):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, '{}_steps_{}_batch_{}.pt'.format(save_prefix, steps,batch_size))
    torch.save(model, save_path)
    print("模型已保存")
