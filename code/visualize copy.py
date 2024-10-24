import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import itertools
import torch
import numpy as np
from Encoder import *
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from sklearn import manifold
import pandas as pd
params={
    'axes.labelsize': '50',         #轴上字
    'xtick.labelsize':'30',        #轴图例
    'ytick.labelsize':'30',           #轴图例
    'lines.linewidth':2 ,              #线宽
    'font.size': 33,
    'figure.figsize': '28, 20',    # set figure size
}
pylab.rcParams.update(params)            #set figure parameter


# 绘制混淆矩阵
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    if normalize:
        cm = 100 * cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('C->E')
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig('conf.png')

def pass_data_iteratively(model, data, minibatch_size = 32):
    model.eval()
    output = []

    idx = np.arange(len(data))

    for i in range(0, len(data), minibatch_size):
        sampled_idx = idx[i:i+minibatch_size]
        if len(sampled_idx) == 0:
            continue
        batch = torch.stack([data.audio[j] for j in sampled_idx]).permute(0,2,1)  #(32,40,219)
        output.append(model(batch).detach())
    return torch.cat(output, 0)

def confusion_matrix(preds, labels, conf_matrix):
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix

def conf():
    classes = ['angry','sad','happy','fear','neutral']

    def getconf(pt,train,test):
        model = CCModel(5).to(0)
        model.load_state_dict(torch.load(pt))
        model.eval()
        train_data = torch.load(train)
        test_data = torch.load(test)
        output1 = pass_data_iteratively(model, train_data)
        output2 = pass_data_iteratively(model, test_data)
        output = torch.cat([output1,output2],dim=0)
        pred = output.max(1, keepdim=True)[1]
        labels1 = train_data.labels.long().to(0)
        labels2 = test_data.labels.long().to(0)
        labels = torch.cat([labels1,labels2],dim=0)
        labels = torch.topk(labels, 1)[1].squeeze(1)
        conf_matrix = torch.zeros(5, 5)
        conf_matrix = confusion_matrix(pred, labels=labels, conf_matrix=conf_matrix)
        conf_matrix = conf_matrix.cpu().numpy()
        return conf_matrix

    conf_matrix = getconf('/home/b3432/Code/experiment/heyinru/speech-batchformer/checkpoint/checkpoint_CC_wobatch/CASIA_EMODB/5_5_0_18_45.pt',
                          '/home/b3432/Code/experiment/heyinru/speech-batchformer/5_fold_split_CC/data_all_EMODB/EMODB_train_na_2.dt',
                          '/home/b3432/Code/experiment/heyinru/speech-batchformer/5_fold_split_CC/data_all_EMODB/EMODB_test_na_2.dt')
    axes = plt.subplot(231)  
    cm = 100 * conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    axes.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes.set_xlabel('CASIA -> EMODB')
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    fmt = '.1f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        axes.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
########################################################################################
    conf_matrix = getconf('/home/b3432/Code/experiment/heyinru/speech-batchformer/checkpoint/checkpoint_CC_wobatch/CASIA_SAVEE/5_4_23_31_14.pt',
                          '/home/b3432/Code/experiment/heyinru/speech-batchformer/5_fold_split_CC/data_all_SAVEE/SAVEE_train_na_0.dt',
                          '/home/b3432/Code/experiment/heyinru/speech-batchformer/5_fold_split_CC/data_all_SAVEE/SAVEE_test_na_0.dt')
    axes = plt.subplot(232)  
    cm = 100 * conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    axes.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes.set_xlabel('CASIA -> SAVEE')
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    fmt = '.1f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        axes.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
#######################################################################################
    conf_matrix = getconf('/home/b3432/Code/experiment/heyinru/speech-batchformer/checkpoint/checkpoint_CC_wobatch/EMODB_CASIA/5_4_23_16_17.pt',
                          '/home/b3432/Code/experiment/heyinru/speech-batchformer/5_fold_split_CC/data_all_CASIA/CASIA_train_na_0.dt',
                          '/home/b3432/Code/experiment/heyinru/speech-batchformer/5_fold_split_CC/data_all_CASIA/CASIA_test_na_0.dt')
    axes = plt.subplot(233)  
    cm = 100 * conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    axes.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes.set_xlabel('EMODB -> CASIA')
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    fmt = '.1f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        axes.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
########################################################################################
    conf_matrix = getconf('/home/b3432/Code/experiment/heyinru/speech-batchformer/checkpoint/checkpoint_CC_wobatch/EMODB_SAVEE/5_4_23_40_27.pt',
                          '/home/b3432/Code/experiment/heyinru/speech-batchformer/5_fold_split_CC/data_all_SAVEE/SAVEE_train_na_2.dt',
                          '/home/b3432/Code/experiment/heyinru/speech-batchformer/5_fold_split_CC/data_all_SAVEE/SAVEE_test_na_2.dt')
    axes = plt.subplot(234)  
    cm = 100 * conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    axes.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes.set_xlabel('EMODB -> SAVEE')
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    fmt = '.1f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        axes.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
########################################################################################
    conf_matrix = getconf('/home/b3432/Code/experiment/heyinru/speech-batchformer/checkpoint/checkpoint_CC_wobatch/SAVEE_CASIA/5_4_23_32_36.pt',
                          '/home/b3432/Code/experiment/heyinru/speech-batchformer/5_fold_split_CC/data_all_CASIA/CASIA_train_na_3.dt',
                          '/home/b3432/Code/experiment/heyinru/speech-batchformer/5_fold_split_CC/data_all_CASIA/CASIA_test_na_3.dt')
    axes = plt.subplot(235)
    cm = 100 * conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    axes.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes.set_xlabel('SAVEE -> CASIA')
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    fmt = '.1f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        axes.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
########################################################################################
    conf_matrix = getconf('/home/b3432/Code/experiment/heyinru/speech-batchformer/checkpoint/checkpoint_CC_wobatch/SAVEE_EMODB/5_4_23_21_22.pt',
                          '/home/b3432/Code/experiment/heyinru/speech-batchformer/5_fold_split_CC/data_all_EMODB/EMODB_train_na_0.dt',
                          '/home/b3432/Code/experiment/heyinru/speech-batchformer/5_fold_split_CC/data_all_EMODB/EMODB_test_na_0.dt')
    axes = plt.subplot(236)  
    cm = 100 * conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    axes.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes.set_xlabel('SAVEE -> EMODB')
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    fmt = '.1f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        axes.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()

    plt.savefig('conf.png')

def drawMFCC(data,idx):
    fig, ax = plt.subplots()
    mfcc_data = np.swapaxes(data, 0, 1)
    cax = ax.imshow(mfcc_data, interpolation='nearest', cmap=cm.coolwarm, origin='lower')
    ax.set_title('MFCC')
    plt.savefig(f'fig/{idx}.png')
    #plt.show()

def pass_data_iteratively_test(model, data1,data2, minibatch_size = 32):
    model.eval()
    output = []

    idx = np.arange(len(data1))
    for i in range(0, len(data1), minibatch_size):
        sampled_idx = idx[i:i+minibatch_size]
        if len(sampled_idx) == 0:
            continue
        batch = torch.stack([data1.audio[j] for j in sampled_idx]).permute(0,2,1)  #(32,40,219)
        features,score = model(batch)
        output.append(features.detach())
    idx = np.arange(len(data2))
    for i in range(0, len(data2), minibatch_size):
        sampled_idx = idx[i:i+minibatch_size]
        if len(sampled_idx) == 0:
            continue
        batch = torch.stack([data2.audio[j] for j in sampled_idx]).permute(0,2,1)  #(32,40,219)
        features,score = model(batch)
        output.append(features.detach())

    return torch.cat(output, 0)

def tsne():

    tsne = manifold.TSNE(n_components=2, init='pca', random_state=1999)
    maker=['o','v','^','s','p','*','<','>','D','d','h','H']#设置散点形状
    model = CCModel(5).to(0)
    ################################################################################################################################################
    model.load_state_dict(torch.load('/home/b3432/Code/experiment/heyinru/speech-batchformer/checkpoint/checkpoint_CC_CASIA_EMODB/4_25_17_25_2.pt'))
    model.eval()
    data1 = torch.load('/home/b3432/Code/experiment/heyinru/speech-batchformer/5_fold_split_CC/data_all_EMODB/EMODB_train_na_1.dt')
    data2 = torch.load('/home/b3432/Code/experiment/heyinru/speech-batchformer/5_fold_split_CC/data_all_EMODB/EMODB_test_na_1.dt')
    features = pass_data_iteratively_test(model, data1,data2)
    labels1 = data1.labels.long()
    labels1 = torch.topk(labels1, 1)[1].squeeze(1)
    # labels1 = labels1.cpu().detach().numpy()
    labels2 = data2.labels.long()
    labels2 = torch.topk(labels2, 1)[1].squeeze(1)
    # labels2 = labels2.cpu().detach().numpy()
    labels = torch.cat([labels1,labels2],dim=0)
    labels = labels.cpu().detach().numpy()
    #print(len(labels))
    x = tsne.fit_transform(features.cpu().detach())
    # 坐标缩放到[0,1]区间
    x_min, x_max = np.min(x,axis=0), np.max(x,axis=0)
    x = (x - x_min) / (x_max - x_min)
    True_labels=labels.reshape((-1,1))
    S_data=np.hstack((x,True_labels))
    S_data=pd.DataFrame({'x':S_data[:,0],'y':S_data[:,1],'label':S_data[:,2]})
    axex = plt.subplot(231)
    for index in range(5):
        X= S_data.loc[S_data['label'] == index]['x']
        Y=S_data.loc[S_data['label'] == index]['y']
        axex.scatter(X,Y,cmap='rainbow', s=50, marker=maker[index])
    axex.legend(['angry','sad','happy','fear','neutral'],loc="upper right")
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.title('C->E')
################################################################################################################################################
    model.load_state_dict(torch.load('/home/b3432/Code/experiment/heyinru/speech-batchformer/checkpoint/checkpoint_CC_CASIA_SAVEE/4_25_17_26_15.pt'))
    model.eval()
    data1 = torch.load('/home/b3432/Code/experiment/heyinru/speech-batchformer/5_fold_split_CC/data_all_SAVEE/SAVEE_train_na_1.dt')
    data2 = torch.load('/home/b3432/Code/experiment/heyinru/speech-batchformer/5_fold_split_CC/data_all_SAVEE/SAVEE_test_na_1.dt')
    features = pass_data_iteratively_test(model, data1,data2)
    labels1 = data1.labels.long()
    labels1 = torch.topk(labels1, 1)[1].squeeze(1)
    # labels1 = labels1.cpu().detach().numpy()
    labels2 = data2.labels.long()
    labels2 = torch.topk(labels2, 1)[1].squeeze(1)
    # labels2 = labels2.cpu().detach().numpy()
    labels = torch.cat([labels1,labels2],dim=0)
    labels = labels.cpu().detach().numpy()
    #print(len(labels))
    x = tsne.fit_transform(features.cpu().detach())
    # 坐标缩放到[0,1]区间
    x_min, x_max = np.min(x,axis=0), np.max(x,axis=0)
    x = (x - x_min) / (x_max - x_min)
    True_labels=labels.reshape((-1,1))
    S_data=np.hstack((x,True_labels))
    S_data=pd.DataFrame({'x':S_data[:,0],'y':S_data[:,1],'label':S_data[:,2]})
    axex = plt.subplot(232)
    for index in range(5):
        X= S_data.loc[S_data['label'] == index]['x']
        Y=S_data.loc[S_data['label'] == index]['y']
        axex.scatter(X,Y,cmap='rainbow', s=50, marker=maker[index])
    axex.legend(['angry','sad','happy','fear','neutral'],loc="upper right")
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.title('C->S')
    ################################################################################################################################################
    model.load_state_dict(torch.load('/home/b3432/Code/experiment/heyinru/speech-batchformer/checkpoint/checkpoint_CC_EMODB_CASIA/4_25_17_4_27.pt'))
    model.eval()
    data1 = torch.load('/home/b3432/Code/experiment/heyinru/speech-batchformer/5_fold_split_CC/data_all_CASIA/CASIA_train_na_1.dt')
    data2 = torch.load('/home/b3432/Code/experiment/heyinru/speech-batchformer/5_fold_split_CC/data_all_CASIA/CASIA_test_na_1.dt')
    features = pass_data_iteratively_test(model, data1,data2)
    labels1 = data1.labels.long()
    labels1 = torch.topk(labels1, 1)[1].squeeze(1)
    # labels1 = labels1.cpu().detach().numpy()
    labels2 = data2.labels.long()
    labels2 = torch.topk(labels2, 1)[1].squeeze(1)
    # labels2 = labels2.cpu().detach().numpy()
    labels = torch.cat([labels1,labels2],dim=0)
    labels = labels.cpu().detach().numpy()
    #print(len(labels))
    x = tsne.fit_transform(features.cpu().detach())
    # 坐标缩放到[0,1]区间
    x_min, x_max = np.min(x,axis=0), np.max(x,axis=0)
    x = (x - x_min) / (x_max - x_min)
    True_labels=labels.reshape((-1,1))
    S_data=np.hstack((x,True_labels))
    S_data=pd.DataFrame({'x':S_data[:,0],'y':S_data[:,1],'label':S_data[:,2]})
    axex = plt.subplot(233)
    for index in range(5):
        X= S_data.loc[S_data['label'] == index]['x']
        Y=S_data.loc[S_data['label'] == index]['y']
        axex.scatter(X,Y,cmap='rainbow', s=50, marker=maker[index])
    axex.legend(['angry','sad','happy','fear','neutral'],loc="upper right")
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.title('E->C')
    ################################################################################################################################################
    model.load_state_dict(torch.load('/home/b3432/Code/experiment/heyinru/speech-batchformer/checkpoint/checkpoint_CC_EMODB_SAVEE/4_25_17_22_52.pt'))
    model.eval()
    data1 = torch.load('/home/b3432/Code/experiment/heyinru/speech-batchformer/5_fold_split_CC/data_all_SAVEE/SAVEE_train_na_3.dt')
    data2 = torch.load('/home/b3432/Code/experiment/heyinru/speech-batchformer/5_fold_split_CC/data_all_SAVEE/SAVEE_test_na_3.dt')
    features = pass_data_iteratively_test(model, data1,data2)
    labels1 = data1.labels.long()
    labels1 = torch.topk(labels1, 1)[1].squeeze(1)
    # labels1 = labels1.cpu().detach().numpy()
    labels2 = data2.labels.long()
    labels2 = torch.topk(labels2, 1)[1].squeeze(1)
    # labels2 = labels2.cpu().detach().numpy()
    labels = torch.cat([labels1,labels2],dim=0)
    labels = labels.cpu().detach().numpy()
    #print(len(labels))
    x = tsne.fit_transform(features.cpu().detach())
    # 坐标缩放到[0,1]区间
    x_min, x_max = np.min(x,axis=0), np.max(x,axis=0)
    x = (x - x_min) / (x_max - x_min)
    True_labels=labels.reshape((-1,1))
    S_data=np.hstack((x,True_labels))
    S_data=pd.DataFrame({'x':S_data[:,0],'y':S_data[:,1],'label':S_data[:,2]})
    axex = plt.subplot(234)
    for index in range(5):
        X= S_data.loc[S_data['label'] == index]['x']
        Y=S_data.loc[S_data['label'] == index]['y']
        axex.scatter(X,Y,cmap='rainbow', s=50, marker=maker[index])
    axex.legend(['angry','sad','happy','fear','neutral'],loc="upper right")
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.title('E->S')
    ################################################################################################################################################
    model.load_state_dict(torch.load('/home/b3432/Code/experiment/heyinru/speech-batchformer/checkpoint/checkpoint_CC_SAVEE_CASIA/4_25_17_27_23.pt'))
    model.eval()
    data1 = torch.load('/home/b3432/Code/experiment/heyinru/speech-batchformer/5_fold_split_CC/data_all_CASIA/CASIA_train_na_4.dt')
    data2 = torch.load('/home/b3432/Code/experiment/heyinru/speech-batchformer/5_fold_split_CC/data_all_CASIA/CASIA_test_na_4.dt')
    features = pass_data_iteratively_test(model, data1,data2)
    labels1 = data1.labels.long()
    labels1 = torch.topk(labels1, 1)[1].squeeze(1)
    # labels1 = labels1.cpu().detach().numpy()
    labels2 = data2.labels.long()
    labels2 = torch.topk(labels2, 1)[1].squeeze(1)
    # labels2 = labels2.cpu().detach().numpy()
    labels = torch.cat([labels1,labels2],dim=0)
    labels = labels.cpu().detach().numpy()
    #print(len(labels))
    x = tsne.fit_transform(features.cpu().detach())
    # 坐标缩放到[0,1]区间
    x_min, x_max = np.min(x,axis=0), np.max(x,axis=0)
    x = (x - x_min) / (x_max - x_min)
    True_labels=labels.reshape((-1,1))
    S_data=np.hstack((x,True_labels))
    S_data=pd.DataFrame({'x':S_data[:,0],'y':S_data[:,1],'label':S_data[:,2]})
    axex = plt.subplot(235)
    for index in range(5):
        X= S_data.loc[S_data['label'] == index]['x']
        Y=S_data.loc[S_data['label'] == index]['y']
        axex.scatter(X,Y,cmap='rainbow', s=50, marker=maker[index])
    axex.legend(['angry','sad','happy','fear','neutral'],loc="upper right")
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.title('S->C')
    ################################################################################################################################################
    model.load_state_dict(torch.load('/home/b3432/Code/experiment/heyinru/speech-batchformer/checkpoint/checkpoint_CC_SAVEE_EMODB/4_25_17_10_13.pt'))
    model.eval()
    data1 = torch.load('/home/b3432/Code/experiment/heyinru/speech-batchformer/5_fold_split_CC/data_all_EMODB/EMODB_train_na_2.dt')
    data2 = torch.load('/home/b3432/Code/experiment/heyinru/speech-batchformer/5_fold_split_CC/data_all_EMODB/EMODB_test_na_2.dt')
    features = pass_data_iteratively_test(model, data1,data2)
    labels1 = data1.labels.long()
    labels1 = torch.topk(labels1, 1)[1].squeeze(1)
    # labels1 = labels1.cpu().detach().numpy()
    labels2 = data2.labels.long()
    labels2 = torch.topk(labels2, 1)[1].squeeze(1)
    # labels2 = labels2.cpu().detach().numpy()
    labels = torch.cat([labels1,labels2],dim=0)
    labels = labels.cpu().detach().numpy()
    #print(len(labels))
    x = tsne.fit_transform(features.cpu().detach())
    # 坐标缩放到[0,1]区间
    x_min, x_max = np.min(x,axis=0), np.max(x,axis=0)
    x = (x - x_min) / (x_max - x_min)
    True_labels=labels.reshape((-1,1))
    S_data=np.hstack((x,True_labels))
    S_data=pd.DataFrame({'x':S_data[:,0],'y':S_data[:,1],'label':S_data[:,2]})
    axex = plt.subplot(236)
    for index in range(5):
        X= S_data.loc[S_data['label'] == index]['x']
        Y=S_data.loc[S_data['label'] == index]['y']
        axex.scatter(X,Y,cmap='rainbow', s=50, marker=maker[index])
    axex.legend(['angry','sad','happy','fear','neutral'],loc="upper right")
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.title('S->E')
    plt.subplots_adjust(wspace=0.05,hspace=0.05)
    plt.savefig('tSNE.png')


if __name__ == '__main__':

    # conf()
    tsne()