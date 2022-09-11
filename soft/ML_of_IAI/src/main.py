import torch
import torch.nn as nn
import torch.nn.functional as F

from model import TextCnn, TextLSTM, MLP
from operation import *
import argparse

parser = argparse.ArgumentParser(description='TextCNN & LSTM text classifier')

parser.add_argument('-lr', type=float, default=0.003, help='学习率')
parser.add_argument('-batch_size', type=int, default=128)
parser.add_argument('-epochs', type=int, default=30)
parser.add_argument('-dropout', type=float, default=0.5,help = '沉默某节点值的概率')
parser.add_argument('-snapshot', type=str, default='models\cnn\_best-acc_82.958_steps_1092_batch_128.pt', help='已有模型的路径')
# --------operation---------
parser.add_argument('-mode',type=str,default='rnn',help='选择使用cnn还是rnn还是MLP')
parser.add_argument('-train', action='store_true', default=True, help='训练一个新的模型')
parser.add_argument('-test', action='store_true', default=False, help='使用测试集测试模型，结合-snapshot来加载已有模型')
parser.add_argument('-predict', action='store_true', default=False, help='使用已有模型来预测在console的输入')
# --------CNN-args----------
parser.add_argument('-kernel-num', type=int, default=256, help='卷积核的个数')
parser.add_argument('-kernel-sizes', type=str, default='2,3,4,5,6', help='不同卷积核大小')
# --------RNN-args-----------
parser.add_argument('-hidden-dim',type=int,default=1024,help="LSTM隐含层的节点数")
parser.add_argument('-num-layers',type = int, default=3, help="lstm隐藏层层数")
parser.add_argument('-bidirectional',type=bool,default=True,help='LSTM是否为双向网络')

args = parser.parse_args()



if args.train:
    train_data_vec,train_data_label = get_data('train',args)
    if args.mode == 'cnn':
        model = TextCnn(args)
    if args.mode == 'rnn':
        model = TextLSTM(args)
    if args.mode == 'MLP':
        model = MLP(args)
    train(train_data_vec,train_data_label,model,args)
if args.test:
    model = torch.load(args.snapshot)
    print("Model loaded!")
    test_data_vec,test_data_label = get_data('test',args)
    test(test_data_vec,test_data_label,model,args)
if args.predict:
    model = torch.load(args.snapshot)
    print("Model loaded!")
    predict(model,args)





