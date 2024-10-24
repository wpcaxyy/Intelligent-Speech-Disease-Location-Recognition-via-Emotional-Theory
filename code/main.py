import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import math
from datetime import datetime
from lsl import LabelStructureLoss

from tqdm import tqdm
from earlystopping import EarlyStopping
from Encoder import *
import random

from sklearn import manifold
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

criterion = nn.CrossEntropyLoss()
criterion_lsl = LabelStructureLoss(4)

def set_seed(seed):
    if seed == 0:
        print(' random seed')
        torch.backends.cudnn.benchmark = True
    else:
        print('manual seed:', seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train(batch_size,model, device, train_set, optimizer, epoch):
    # 训练模型
    model.train()
    # Iterations代表完成一个epoch需要的batch数
    total_iters = math.ceil(len(train_set)/batch_size)
    # 进度条控件
    pbar = tqdm(range(total_iters), unit='batch')
    # 有序序列
    all_idx = []
    for i in range(len(train_set)):
        all_idx.append(i)
    # 随机
    all_idx = np.random.permutation(all_idx)

    loss_accum = 0
    for pos in pbar:
        # 选择一个batch数量的数据索引
        selected_idx = all_idx[pos*batch_size:(pos+1)*batch_size]
        # 根据索引得到此次batch里选中的样本
        batch = torch.stack([train_set.audio[idx] for idx in selected_idx]).permute(0,2,1)  #(32,40,219)

        # 得到输出值
        output = model(batch)
        # 得到标签值
        labels = torch.LongTensor(np.array([train_set.labels[idx].cpu().detach().numpy() for idx in selected_idx])).to(device)
        labels = torch.topk(labels, 1)[1].squeeze(1)

        #batchformer
        labels = torch.cat([labels,labels],dim=0)

        # loss = criterion(output, labels)
        # loss = criterion_lsl(output, labels)
        loss = criterion(output, labels) + criterion_lsl(output, labels)

        # 反向传播
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 得到此batch的loss
        loss = loss.detach().cpu().numpy()
        loss_accum += loss

        # report
        pbar.set_description('epoch: %d' % (epoch))

    average_loss = loss_accum / total_iters
    print("training loss: %f" % (average_loss))

    # 返回此次epoch的平均损失
    return average_loss


###pass data to model with minibatch during testing to avoid memory overflow (does not perform backpropagation)
def pass_data_iteratively(model, data, minibatch_size = 32):
    model.eval()
    output = []

    idx = np.arange(len(data))

    if minibatch_size == 1:
        for i in range(len(data)):
            sampled_idx = np.repeat(idx[i],32)
            batch = torch.stack([data.audio[j] for j in sampled_idx]).permute(0,2,1)  #(32,40,219)
            output.append(model(batch).detach()[0].unsqueeze(dim=0))
        return torch.cat(output, 0)

    for i in range(0, len(data), minibatch_size):
        sampled_idx = idx[i:i+minibatch_size]
        if len(sampled_idx) == 0:
            continue
        batch = torch.stack([data.audio[j] for j in sampled_idx]).permute(0,2,1)  #(32,40,219)
        output.append(model(batch).detach())
    return torch.cat(output, 0)

def tsne(type,pt,test):

    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)

    def draw(tsne,x,labels,name):
        x = tsne.fit_transform(x.cpu().detach())

        # 坐标缩放到[0,1]区间
        x_min, x_max = np.min(x,axis=0), np.max(x,axis=0)
        x = (x - x_min) / (x_max - x_min)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        # 降维后的坐标为（X[i, 0], X[i, 1]）
        plt.scatter(x[:,0], x[:,1], c=labels ,marker='o',cmap='rainbow')

        plt.title('t-sne')
        plt.savefig(f'/home/b3432/Code/experiment/heyinru/speech-batchformer/modules/{name}.png')

    model = NewModel().to(0)
    model.load_state_dict(torch.load(f'/home/b3432/Code/experiment/heyinru/speech-batchformer/checkpoint/checkpoint_{type}/{pt}.pt'))
    test_data = torch.load(f'/home/b3432/Code/experiment/heyinru/speech-batchformer/10_fold_split/data_all/features_librosa_7/IEMOCAP_test_na_{test}.dt')

    pres = []
    sufs = []
    idx = np.arange(len(test_data))
    for i in range(0, len(test_data), 32):
        sampled_idx = idx[i:i+32]
        if len(sampled_idx) == 0:
            continue
        batch = torch.stack([test_data.audio[j] for j in sampled_idx]).permute(0,2,1)
        pre,suf,score = model(batch)
        pres.append(pre)
        sufs.append(suf)
    pres = torch.cat(pres,0)
    sufs = torch.cat(sufs,0)

    labels = test_data.labels.long()
    labels = torch.topk(labels, 1)[1].squeeze(1)
    labels = labels.cpu().detach().numpy()

    draw(tsne,pres,labels,'pre')
    draw(tsne,sufs,labels,'suf')



def test(model, device, train_data, test_data,type=0):
    # 评估模式，关闭梯度计算
    model.eval()

    # type = 0  不测试训练集
    # type = 1  测试训练集和测试集

    # 计算测试集上的准确率
    output = pass_data_iteratively(model, test_data)
    pred = output.max(1, keepdim=True)[1]
    labels = test_data.labels.long().to(device)
    labels = torch.topk(labels, 1)[1].squeeze(1)
    new_labels = labels.view_as(pred)
    correct = pred.eq(new_labels).sum().cpu().item()
    acc_test = correct / float(len(test_data))
    
    # 计算UA
    UA = 0
    num_class = output.shape[1]
    for i in range(num_class):
        eq = torch.where(new_labels==i,i,-1)
        top = pred.eq(eq).sum().cpu().item()
        eq = torch.where(new_labels==i,1,0)
        bottom = eq.sum().cpu().item()
        UA += (top / bottom)
    UA /= num_class

    if type == 1:
        # 模型输出值
        output = pass_data_iteratively(model, train_data)
        # softmax预测值
        pred_ = output.max(1, keepdim=True)[1]
        # 标签值
        labels = train_data.labels.long().to(device)
        labels = torch.topk(labels, 1)[1].squeeze(1)
        # 预测正确的个数
        correct = pred_.eq(labels.view_as(pred_)).sum().cpu().item()
        # 准确率
        acc_train = correct / float(len(train_data))

        print("accuracy train: %f test: %f" % (acc_train, acc_test))
        print('unweight accuracy test : %f' % UA)
        return acc_train, acc_test, UA
        
    return acc_test, UA

def main():
    # Training settings
    # Note: Hyper-parameters need to be tuned in order to obtain results reported in the paper.
    # 参数设置
    device = 0
    fold = 10
    lr = 0.0001
    patience = 300
    epochs = 300
    batch_size = 32
    data_set = 'IEMOCAP'
    model_type = 'celsl'
    # 特征保存路径
    SAVE_PATH = '/home/b3432/Code/experiment/heyinru/speech-batchformer/10_fold_split/data_all/features_librosa_7/'

    # 设置随机种子和gpu设备
    set_seed(2019)

    device = torch.device("cuda:" + str(device)) if torch.cuda.is_available() else torch.device("cpu")

    acc_train_sum = 0
    acc_test_sum = 0
    uacc_test_sum = 0

    # Cross_Validation
    for i in range(fold):

        model = NewModel().to(device)

        # 加载数据集
        train_save_path = SAVE_PATH + f'{data_set}_train_na_{i}.dt'
        test_save_path = SAVE_PATH + f'{data_set}_test_na_{i}.dt'
        # IEMOCAP-Dataset类
        train_data = torch.load(train_save_path)
        test_data = torch.load(test_save_path)

        # 优化器用Adam
        optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay=1e-6)

        # 用StepLR机制调整学习率，Learning rate Decay
        g = math.exp(-0.15)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=g)

        # 设置Keras的Earlystop
        early_stopping = EarlyStopping(patience=patience, verbose=True)

        # 每个epoch训练
        for epoch in range(1, epochs + 1):

            # 训练模型获得平均损失
            avg_loss = train(batch_size, model, device, train_data, optimizer, epoch)
            scheduler.step()

            # 第一个epoch之后在验证集上求loss
            if epoch>1:
                # Validation check
                acc_test,UA = test(model, device, train_data, test_data)           
                #Check early stopping
                early_stopping(acc_test, UA, model,f'/home/b3432/Code/experiment/heyinru/speech-batchformer/checkpoint/checkpoint_{model_type}/checkpoint.pt')

                if early_stopping.early_stop:
                    print("Early stopping")
                    break

        # 此折训练结果
        # 读取本地checkpoint
        model.load_state_dict(torch.load(f'/home/b3432/Code/experiment/heyinru/speech-batchformer/checkpoint/checkpoint_{model_type}/checkpoint.pt'))
        # 此次最好的模型去验算
        acc_train, acc_test, uacc_test = test(model, device, train_data, test_data,1)
        acc_train_sum += acc_train
        acc_test_sum += acc_test
        uacc_test_sum += uacc_test
        # 写日志
        date = datetime.now()
        # os.system(f'cp /home/b3432/Code/experiment/heyinru/speech-batchformer/modules/checkpoint.pt /home/b3432/Code/experiment/heyinru/speech-batchformer/checkpoint/checkpoint_{model_type}/')
        os.system(f'mv /home/b3432/Code/experiment/heyinru/speech-batchformer/checkpoint/checkpoint_{model_type}/checkpoint.pt /home/b3432/Code/experiment/heyinru/speech-batchformer/checkpoint/checkpoint_{model_type}/{date.month}_{date.day}_{date.hour}_{date.minute}_{date.second}.pt')
        log = open(f'/home/b3432/Code/experiment/heyinru/speech-batchformer/checkpoint/checkpoint_{model_type}/log.txt', 'a')
        log.write(f'{date.month}_{date.day}_{date.hour}_{date.minute}_{date.second}.pt\t{acc_train}\t{acc_test}\t{uacc_test}\n')
        log.close()

    average_train_acc = acc_train_sum / fold
    average_test_acc = acc_test_sum / fold
    average_test_uacc = uacc_test_sum / fold

    print('Average train acc: %f, Average test acc: %f, Average test uacc: %f' % (average_train_acc, average_test_acc, average_test_uacc))

    log = open(f'/home/b3432/Code/experiment/heyinru/speech-batchformer/checkpoint/checkpoint_{model_type}/log.txt', 'a')
    log.write(f'{average_train_acc}\t{average_test_acc}\t{average_test_uacc}\n')
    log.close()


if __name__ == '__main__':

    main()

    # tsne('celsl','2_10_7_56_21','3')
