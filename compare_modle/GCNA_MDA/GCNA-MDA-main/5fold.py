import numpy as np
import pandas as pd
import random
import torch
from GCN import GCN_Net
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score,roc_curve, auc,precision_recall_curve
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors                     # 引入matplotlib的colors模块，用于颜色管理
from scipy.interpolate import interp1d                  # 引入scipy库的interp1d函数，用于插值计算
from Sensitivity import sensitivity                     # 从Sensitivity模块导入sensitivity函数，用于计算敏感性
import matplotlib
matplotlib.use('TkAgg')
colors = list(mcolors.TABLEAU_COLORS.keys())        # 获取TABLEAU_COLORS的颜色列表
from sklearn.metrics import (roc_curve, auc, precision_recall_curve, average_precision_score,
                             f1_score, accuracy_score, recall_score, precision_score, confusion_matrix)
import pickle

random.seed(123)
matplotlib.use('TkAgg')
aadj=pd.DataFrame(np.loadtxt("interaction.txt"))
SD=pd.DataFrame(np.loadtxt("SD.txt"))
SM=pd.DataFrame(np.loadtxt("SM.txt"))

adj_list=[]
# 遍历aadj的每一行和每一列，如果行列对应位置的值为1，则将[miRNA节点下标,疾病节点下标]添加到邻接列表
for index,row in aadj.iterrows():         # index为行索引值 row为对应行的行内容
    for i in range(len(aadj.iloc[0])):
        if row[i]==1:
            adj_list.append([index,i+495])#(miRNA节点下标,疾病节点下标)
feature_matrix=pd.DataFrame(0,index=range(878),columns=range(878))      # 初始化特征矩阵，尺寸为878x878，初始值全为0


# 填充特征矩阵的前495列和后面的列，分别使用SM和SD的数据
for index in range(len(SM)):
    feature_matrix.iloc[index,:495]=SM.iloc[index]
for index in range(len(SM),len(SM)+len(SD)):
    feature_matrix.iloc[index,495:]=SD.iloc[index-495]


# 使用StratifiedKFold进行分层抽样，分为5个fold，不进行shuffle
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=False)
label=[1]*len(adj_list)                                 # 初始化正样本标签为1，负样本标签为0，并将正负样本标签合并
label.extend([0]*len(adj_list))
random.shuffle(adj_list)                                # 打乱邻接列表的顺序
lenth=len(adj_list)                                      # 获取邻接列表的长度
i=0

# 循环生成负样本，直到负样本的数量与正样本相等
while i <lenth:
    neg_index=random.randint(495,877)
    # 如果生成的负样本不在邻接列表中，则添加到邻接列表，并将i加1  为每一个存在正样本对的的mirna找到一个负样本
    if [adj_list[i][0],neg_index] not in adj_list:
        adj_list.append([adj_list[i][0],neg_index])
        i+=1
    else:
        continue         # 如果生成的负样本已经存在于邻接列表中，则跳过并继续尝试生成新的负样本
adj_list=np.array(adj_list)
label=np.array(label)
fold=-1                                                         # 初始化fold变量，用于记录当前是第几个fold



sk_tprs = []
sk_aucs = []
sk_precisions = []
sk_recalls = []
sk_average_precisions = []
sk_fpr = []

metrics_summary = {
    'f1_scores': [],
    'accuracies': [],
    'recalls': [],
    'specificities': [],
    'precisions': []
}

for train_index, val_index in skf.split(adj_list,label):
    fold+=1
    # 根据索引获取当前fold的训练集和验证集的特征和标签
    train_x,val_x=list(adj_list[train_index]),list(adj_list[val_index])
    train_y,val_y=list(label[train_index]),list(label[val_index])

    adj=[]
    # 遍历训练集，将正样本的特征添加到adj中
    for i in range(len(train_x)):
        if train_y[i]==1:
            adj.append(train_x[i])


    adj=torch.tensor(adj).to(torch.long).T.cuda()
    train_x=torch.tensor(train_x).to(torch.long).T.cuda()
    train_y=torch.tensor(train_y).to(torch.float32).cuda()
    val_x=torch.tensor(val_x).to(torch.long).T.cuda()
    val_y=torch.tensor(val_y).to(torch.float32).cuda()
    feature=torch.tensor(feature_matrix.to_numpy()).to(torch.float32).cuda()

    # 实例化GCN模型，设置输入层、隐藏层和输出层的维度，并移动到GPU上
    model = GCN_Net(len(feature[0]), 128, 64).cuda()
    model.train()
    opt = torch.optim.Adam(params=model.parameters(), lr=0.001,weight_decay=1e-4)
    loss_fn = torch.nn.BCELoss().cuda()     # 设置损失函数为二元交叉熵损失，并移动到GPU上

    
    epoch = 2000

    # 开始训练模型，每一轮中计算预测值、损失、执行反向传播和参数更新
    for i in range(epoch):
        y_hat = model(feature, adj, train_x)                        # 使用模型进行预测
        loss = loss_fn(y_hat.to(torch.float32), train_y)            # 计算损失
        opt.zero_grad()                                             # 清除梯度
        loss.backward()                                             # 执行反向传播
        opt.step()                                                  # 更新参数
        if i % 100==0:
            print(loss)                                             # 每100轮打印一次损失值
    model.eval()

    # 使用AUROC、BinaryAccuracy、BinarySpecificity和BinaryPrecision评估模型性能
    from torchmetrics import AUROC
    from torchmetrics.classification import BinaryAccuracy, BinarySpecificity, BinaryPrecision
    Auc = AUROC(task="binary")                                       # 初始化AUROC评估器
    with torch.no_grad():                                            # 关闭梯度计算
        y_hat = model(feature, adj, val_x)                          # 使用模型进行预测
        Auc_value = Auc(y_hat.cpu(), val_y.cpu()).item()             # 计算AUC值
        print(f"AUC:{Auc_value}")
        y_hat=y_hat.cpu()
        val_y=val_y.cpu()                                           # 将预测值和真实标签移动到CPU上
        #_list=[i/100 for i in range(65,75)]
        #for i in _list:
        i=0.67                                                      # 设置一个阈值为0.67
        y_label = (y_hat >= i).int()

    metrics_summary['f1_scores'].append(f1_score(val_y, y_label))
    metrics_summary['accuracies'].append(accuracy_score(val_y, y_label))
    metrics_summary['recalls'].append(recall_score(val_y, y_label))
    metrics_summary['precisions'].append(precision_score(val_y, y_label))
    tn, fp, fn, tp = confusion_matrix(val_y, y_label).ravel()
    specificity = tn / (tn + fp)
    metrics_summary['specificities'].append(specificity)

    fpr_temp, tpr_temp, _ = roc_curve(val_y, y_label)
    roc_auc = auc(fpr_temp, tpr_temp)
    sk_fpr.append(fpr_temp)
    sk_tprs.append(tpr_temp)
    sk_aucs.append(roc_auc)

    # 计算Precision-Recall曲线和AUPR
    precision_temp, recall_temp, _ = precision_recall_curve(val_y, y_label)
    average_precision = average_precision_score(val_y, y_label)
    sk_precisions.append(precision_temp)
    sk_recalls.append(recall_temp)
    sk_average_precisions.append(average_precision)




print('############ avg score #############')
for metric, values in metrics_summary.items():
    print(f"{metric}: {np.mean(values):.2f} ± {np.std(values):.2f}")

#############################################################   保存画图的数据   ########################################################
print('mean AUC = ',np.mean(sk_aucs))
print('mean AUPR = ',np.mean(sk_average_precisions))

mean_fpr = np.linspace(0, 1, 100)
mean_recall  = np.linspace(0, 1, 100)
tprs = []
precisions = []
for fpr_temp, tpr_temp in zip(sk_fpr, sk_tprs):
    interp_tpr = np.interp(mean_fpr, fpr_temp, tpr_temp)
    interp_tpr[0] = 0.0  # 确保曲线从 0 开始
    tprs.append(interp_tpr)
mean_tpr  = np.mean(tprs, axis=0)

mean_tpr [-1] = 1.0  # 确保曲线以 1 结束

for recall_temp, precision_temp in zip(sk_recalls, sk_precisions):
    interp_precision = np.interp(mean_recall, recall_temp[::-1], precision_temp[::-1])
    precisions.append(interp_precision)

mean_precision = np.mean(precisions, axis=0)
sk_aucs = np.mean(sk_aucs)
sk_average_precisions = np.mean(sk_average_precisions)




roc_data = {
    'fpr': mean_fpr,
    'tprs': mean_tpr,
    'aucs': sk_aucs
}

# Precision-Recall曲线数据
pr_data = {
    'recalls': mean_recall,
    'precisions': mean_precision,
    'average_precisions': sk_average_precisions
}

# 保存ROC数据
with open('roc_data.pkl', 'wb') as f:
    pickle.dump(roc_data, f)
# 保存Precision-Recall数据
with open('pr_data.pkl', 'wb') as f:
    pickle.dump(pr_data, f)


# 加载ROC数据
with open('roc_data.pkl', 'rb') as f:
    roc_data = pickle.load(f)
# 加载Precision-Recall数据
with open('pr_data.pkl', 'rb') as f:
    pr_data = pickle.load(f)

# 绘制ROC曲线
fig2, axs2 = plt.subplots(1, 1, figsize=(5, 5))

axs2.plot(roc_data['fpr'], roc_data['tprs'], label=f" AUC = {roc_data['aucs']:.2f}")
axs2.plot([0, 1], [0, 1], 'k--', label='Random', color='r')
axs2.set_xlim([-0.05, 1.05])
axs2.set_ylim([0.0, 1.05])
axs2.set_xlabel('False Positive Rate')
axs2.set_ylabel('True Positive Rate')
axs2.set_title('ROC')
axs2.legend(loc="lower right")
plt.show()

# 绘制Precision-Recall曲线
fig3, axs3 = plt.subplots(1, 1, figsize=(5, 5))

axs3.plot(pr_data['recalls'], pr_data['precisions'], label=f" AUPR = {pr_data['average_precisions']:.2f}")
axs3.plot([0, 1], [1, 0], 'k--', label='Random', color='r')
axs3.set_xlim([-0.05, 1.05])
axs3.set_ylim([0.0, 1.05])
axs3.set_xlabel('Recall')
axs3.set_ylabel('Precision')
axs3.set_title('Precision-Recall Curvesin')
axs3.legend(loc="lower left")
plt.show()

print("模型指标和曲线数据已保存。")

