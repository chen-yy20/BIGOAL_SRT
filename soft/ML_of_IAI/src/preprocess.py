import torch
from gensim.models import KeyedVectors


model = KeyedVectors.load_word2vec_format("./Dataset/wiki_word2vec_50.bin",binary=True)
word_vectors = torch.randn([59300,50])


lines = []
f = open("./Dataset/test.txt",encoding='utf-8')
lines += f.readlines()
f = open("./Dataset/train.txt",encoding='utf-8')
lines += f.readlines()
f = open("./Dataset/validation.txt",encoding='utf-8')
lines += f.readlines()

## 给所有数据sentence的长度进行排序，确定padding的量

# len_lines = [len(i.split(" "))-1 for i in lines]
# list = sorted(len_lines,reverse=True)
# print(list)

# 结论：使用119的向量即可
# [678, 118, 118, 118, 116, 115, 113, 113, 113, 112, 112, 112, 111, 111, 110, 1

## 计算词表中一共有多少个词，方便进行embedding
## 词表长度： 59289

# word_list = []
# cnt =0
# for line in lines:
#     each_sentence_list = line[1:].split(" ")
#     for word in each_sentence_list:
#         word = word.strip('\n').strip('\t')
#         word = word.replace(' ','')
#         if word not in word_list:
#             # print(word)
#             cnt+=1
#             print(cnt)
#             word_list.append(word)
# print("词表长度："+str(len(word_list)))

# # 保存问题列表
file = open('./Dataset/word_list.txt',encoding = 'utf_8')
# for fp in word_list:
#     file.write(fp)
#     file.write('\n')
# file.close()
# print('保存词表成功！')

## load word_list

word_list = []
for line in file.readlines():
    line = line.strip('\n')
    word_list.append(line)
file.close()
print('读取文档成功！')

## 已有的进行词嵌入，没有的随机生成
for i in range(len(word_list)):
    word = word_list[i]
    if word in model:
        vector = model[word]
        word_vectors[i] = torch.from_numpy(vector)

print(word_vectors.shape)




