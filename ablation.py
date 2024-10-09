import torch
# import shap
import matplotlib
import pandas as pd
from utils import *
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import (roc_curve, auc, precision_recall_curve, average_precision_score,
                             f1_score, accuracy_score, recall_score, precision_score, confusion_matrix)
#from model import LinearEncoder, create_multi_dimensional_data, MultiDimensionalGCN,MicroDiseaseModel,MultiDimensionalGCN_6,MultiDimensionalGCN_3_add_3_Nor,MultiDimensionalGCN_6_Nor
from model import *
#torch.autograd.set_detect_anomaly(True)
# 检查CUDA是否可用
out = []  # 用于存储每一折的训练集和测试集索引

iter = 0
k_split = 5
set_seed(123)
lambda_mse = 4  #
lambda_l2 = 3e-2 #4e-2
lambda_constrate = 5
matplotlib.use('TkAgg')
criterion = torch.nn.MSELoss()
kf = KFold(n_splits=k_split, shuffle=True, random_state=123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']# 使用颜色编码定义颜色




precisions = []
precisions_6_MGCN = []
precisions_3_MGCN_3_GCN= []
precisions_6_GCN = []
precisions_6_spilt = []

sk_tprs = []
sk_aucs = []
sk_precisions = []
sk_recalls = []
sk_average_precisions = []
sk_fpr= []


sk_tprs_6_MGCN = []
sk_aucs_6_MGCN = []
sk_precisions_6_MGCN = []
sk_recalls_6_MGCN = []
sk_average_precisions_6_MGCN = []
sk_fpr_6_MGCN = []


sk_tprs_3_MGCN_3_GCN = []
sk_aucs_3_MGCN_3_GCN = []
sk_precisions_3_MGCN_3_GCN = []
sk_recalls_3_MGCN_3_GCN = []
sk_average_precisions_3_MGCN_3_GCN = []
sk_fpr_3_MGCN_3_GCN = []


sk_tprs_6_GCN = []
sk_aucs_6_GCN = []
sk_precisions_6_GCN = []
sk_recalls_6_GCN = []
sk_average_precisions_6_GCN = []
sk_fpr_6_GCN = []


sk_tprs_6_spilt = []
sk_aucs_6_spilt = []
sk_precisions_6_spilt = []
sk_recalls_6_spilt = []
sk_average_precisions_6_spilt = []
sk_fpr_6_spilt = []







fig, ax = plt.subplots()
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
fig, ax = plt.subplots()
metrics_summary = {
    'f1_scores': [],
    'accuracies': [],
    'recalls': [],
    'specificities': [],
    'precisions': []
}

sk_tprs = []
sk_aucs = []
sk_precisions = []
sk_recalls = []
sk_average_precisions = []
sk_fpr = []
test_label_score = {}

# 读取数据
#HMDAD
A = pd.read_excel('./dataset/HMDAD/adj_mat.xlsx')
disease_chemical = pd.read_csv('./dataset/HMDAD/化学-疾病/complete_disease_similarity_matrix.csv')
disease_gene = pd.read_csv('./dataset/HMDAD/基因-疾病/complete_disease_similarity_matrix.csv')
disease_symptoms = pd.read_csv('./dataset/HMDAD/疾病-症状/complete_disease_similarity_matrix.csv')
disease_Semantics = pd.read_csv('./dataset/HMDAD/疾病-语义/similarity_matrix_model2.csv', header=None)
disease_pathway = pd.read_csv('./dataset/HMDAD/疾病-通路/complete_disease_similarity_matrix.csv')

micro_cos = pd.read_csv('./dataset/HMDAD/基于关联矩阵的微生物功能/Cosine_Sim.csv')
micro_gip = pd.read_csv('./dataset/HMDAD/基于关联矩阵的微生物功能/GIP_Sim.csv')
micro_sem = pd.read_csv('./dataset/HMDAD/基于疾病语义的微生物功能/functional_similarity2_matrix.csv')
micro_fun1 = pd.read_csv('./dataset/HMDAD/微生物-功能/complete_microbe_associations_ds2_matrix.csv')
micro_fun2 = pd.read_csv('./dataset/HMDAD/微生物-功能/complete_microbe_similarities_ds2_matrix.csv')
A = A.iloc[1:, 1:]

disease_chemical = disease_chemical.iloc[:, 1:]
disease_gene = disease_gene.iloc[:, 1:]
disease_symptoms = disease_symptoms.iloc[:, 1:]
disease_pathway = disease_pathway.iloc[:, 1:]
micro_cos = micro_cos.iloc[:, 1:]
micro_gip = micro_gip.iloc[:, 1:]
micro_fun1 = micro_fun1.iloc[:, 1:]
micro_fun2 = micro_fun2.iloc[:, 1:]


lambda_l2 = (lambda_l2*39*292)/(A.shape[0]*A.shape[1])


len_micro = len(micro_cos)
len_disease = len(disease_chemical)
# 生成5个不同来源的微生物关联矩阵，每个矩阵的大小为292x292
microbiome_matrices = [micro_cos.values, micro_gip.values, micro_sem.values, micro_fun1.values, micro_fun2.values]
# 生成5个不同来源的疾病关联矩阵，每个矩阵的大小为39x39
disease_matrices = [disease_chemical.values, disease_gene.values, disease_symptoms.values, disease_Semantics.values,
                    disease_pathway.values]

non_zero_micro, non_zero_disease = non_zero_count(microbiome_matrices, disease_matrices)
non_zero_micro = non_zero_micro.to(device)
non_zero_disease = non_zero_disease.to(device)

sim_d, sim_m = calculate_combined_similarity(disease_matrices, microbiome_matrices)
dm = A.values
mm = sim_m
dd = sim_d

samples = get_all_the_samples_old(A.T.values)
samples = np.array(samples)


micro_low_features = A.T  # 微生物的低级特征是矩阵A的列向量
disease_low_features = A  # 疾病的低级特征是矩阵A的行向量

micro_low_features_tensor = torch.Tensor(micro_low_features.values).to(device)
disease_low_features_tensor = torch.Tensor(disease_low_features.values).to(device)

# 将微生物关联矩阵转换为多维图数据
micro_data, micro1, micro2, micro3, micro4, micro5, micro_edge_indices, micor_edge_weights = create_multi_dimensional_data(microbiome_matrices, device,micro_low_features_tensor)
# 将疾病关联矩阵转换为多维图数据
disease_data, dis1, dis2, dis3, dis4, dis5, dis_edge_indices, dis_edge_weights = create_multi_dimensional_data(disease_matrices, device,disease_low_features_tensor)




#############################################    3MGCN_3spilt   begin   ############################################################
# HMDAD
microbiome_gcn = MultiDimensionalGCN(in_channels=331, hidden_channels=128, out_channels=64, num_dimensions=5).to(device)
disease_gcn = MultiDimensionalGCN(in_channels=331, hidden_channels=128, out_channels=64, num_dimensions=5).to(device)

len_micro_out = 64
len_disease_out = 64

# 优化器
optimizer1 = torch.optim.Adam(microbiome_gcn.parameters(), lr=0.01)
optimizer2 = torch.optim.Adam(disease_gcn.parameters(), lr=0.01)
# optimizer3 = torch.optim.Adam(vae_model.parameters(), lr=0.01)

A_tensor = torch.tensor(A.values, dtype=torch.float32).to(device)
prob_matrix_avg = np.zeros((A.shape[1], A.shape[0]))



fold_data = []
for cl, (train_index, test_index) in enumerate(kf.split(samples)):  # 循环每一折的训练和测试集

    print('############ {} fold #############'.format(cl))  # 打印当前折数
    out.append([train_index, test_index])  # 将训练和测试集索引存入列表中
    iter = iter + 1  # 迭代次数加1
    train_samples = samples[train_index, :]  # 获取当前折的训练集样本
    test_samples = samples[test_index, :]  # 获取当前折的测试集样本
    train_n = train_samples.shape[0]  # 获取训练集样本数量
    test_n = test_samples.shape[0]  # 获取测试集样本数量 len_micro

    micro_input_dim = len_micro_out
    disease_input_dim = len_disease_out

    fold_data_dict = {
        'final_matrix_train': [],
        'final_label_train': [],
        'final_matrix_test': [],
        'final_label_test': []
    }

    microbiome_gcn.train()
    disease_gcn.train()
    # vae_model.train()
    for epoch in range(50):
        loss = 0
        optimizer1.zero_grad()
        optimizer2.zero_grad()

        micro_out = microbiome_gcn(micro1, micro2,micro3,micro4,micro5,micro_edge_indices,micor_edge_weights, non_zero_micro)
        disease_out = disease_gcn(dis1,dis2,dis3,dis4,dis5,dis_edge_indices,dis_edge_weights, non_zero_disease)

        micro_out=F.normalize(micro_out, p=2, dim=1)
        disease_out = F.normalize(disease_out, p=2, dim=1)

        train_list = np.array(train_samples)
        test_list = np.array(test_samples)
        train_list_tensor = torch.tensor(train_list, dtype=torch.float32).to(device)
        test_list_tensor = torch.tensor(test_list, dtype=torch.float32).to(device)
        pre_predicted_matrix = torch.mm(micro_out,disease_out.T)
        predicted_matrix = (pre_predicted_matrix - pre_predicted_matrix.min()) / (pre_predicted_matrix.max() - pre_predicted_matrix.min()).to(torch.float32)

        train_labels = train_list_tensor[:, 2]  # 实际标签
        indices = train_list_tensor[:, :2].long()  # 确保索引为整数类型
        #train_label = predicted_matrix[indices[:, 0], indices[:, 1]]  # 使用张量索引获取预测值
        train_label = predicted_matrix[indices[:, 0], indices[:, 1]]
        train_label = train_label.double()
        train_labels = train_labels.double()
        loss_l2 = lambda_l2 * torch.norm(predicted_matrix, p='fro')
        # 现在 train_label 和 train_labels 都是张量，并且可以在计算损失时保持梯度追踪
        matrix_diff_loss = torch.mean((predicted_matrix - A_tensor.T) ** 2)
        constrate_loss = 0

        loss = lambda_mse * criterion(train_label,train_labels).to(torch.float32)  + loss_l2

        loss.backward(retain_graph=True)
        optimizer1.step()
        optimizer2.step()
        print(f'Epoch {epoch + 1}/{50}, Loss: {loss.item()}')

    microbiome_gcn.eval()
    disease_gcn.eval()

    with torch.no_grad():
        pre_prob_matrix = torch.mm(micro_out, disease_out.T)
        prob_matrix = (pre_prob_matrix - pre_prob_matrix.min()) / (pre_prob_matrix.max() - pre_prob_matrix.min())
        prob_matrix_np = prob_matrix.cpu().numpy()  # Ensure the matrix is on CPU
        prob_matrix_avg += prob_matrix_np

        unique_test_list_tensor = test_list_tensor
        test_labels = unique_test_list_tensor[:, 2].cpu().numpy()  # Move to CPU if necessary
        indices = unique_test_list_tensor[:, :2].long()  # Ensure indices are long

        perdcit_score = prob_matrix[indices[:, 0], indices[:, 1]].cpu().numpy()  # Move to CPU if necessary
        perdcit_label = [1 if prob >= 0.33 else 0 for prob in perdcit_score]

        viz = metrics.RocCurveDisplay.from_predictions(test_labels, perdcit_score,
                                                       name='ROC fold {}'.format(cl),
                                                       color=colors[cl],
                                                       alpha=0.6, lw=2, ax=ax)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

        metrics_summary['f1_scores'].append(f1_score(test_labels, perdcit_label))
        metrics_summary['accuracies'].append(accuracy_score(test_labels, perdcit_label))
        metrics_summary['recalls'].append(recall_score(test_labels, perdcit_label))
        metrics_summary['precisions'].append(precision_score(test_labels, perdcit_label))

        tn, fp, fn, tp = confusion_matrix(test_labels, perdcit_label).ravel()
        specificity = tn / (tn + fp)
        metrics_summary['specificities'].append(specificity)

        fpr_temp, tpr_temp, _ = roc_curve(test_labels, perdcit_score)
        roc_auc = auc(fpr_temp, tpr_temp)
        sk_fpr.append(fpr_temp)
        sk_tprs.append(tpr_temp)
        sk_aucs.append(roc_auc)

        precision_temp, recall_temp, _ = precision_recall_curve(test_labels, perdcit_score)
        average_precision = average_precision_score(test_labels, perdcit_score)
        sk_precisions.append(precision_temp)
        sk_recalls.append(recall_temp)
        sk_average_precisions.append(average_precision)

        test_label_score[cl] = [test_labels, perdcit_score]
        torch.cuda.empty_cache()



prob_matrix_avg = prob_matrix_avg / k_split

mean_fpr = np.linspace(0, 1, 100)
mean_recall = np.linspace(0, 1, 100)
tprs = []
precisions = []
for fpr_temp, tpr_temp in zip(sk_fpr, sk_tprs):
    interp_tpr = np.interp(mean_fpr, fpr_temp, tpr_temp)
    interp_tpr[0] = 0.0  # 确保曲线从 0 开始
    tprs.append(interp_tpr)
mean_tpr = np.mean(tprs, axis=0)

mean_tpr[-1] = 1.0  # 确保曲线以 1 结束

for recall_temp, precision_temp in zip(sk_recalls, sk_precisions):
    interp_precision = np.interp(mean_recall, recall_temp[::-1], precision_temp[::-1])
    precisions.append(interp_precision)

mean_precision = np.mean(precisions, axis=0)

sk_aucs = np.mean(sk_aucs)
sk_average_precisions = np.mean(sk_average_precisions)
mean_fpr = np.linspace(0, 1, 100)
mean_recall = np.linspace(0, 1, 100)
tprs = []
precisions = []
for fpr_temp, tpr_temp in zip(sk_fpr, sk_tprs):
    interp_tpr = np.interp(mean_fpr, fpr_temp, tpr_temp)
    interp_tpr[0] = 0.0  # 确保曲线从 0 开始
    tprs.append(interp_tpr)
mean_tpr = np.mean(tprs, axis=0)

mean_tpr[-1] = 1.0  # 确保曲线以 1 结束

for recall_temp, precision_temp in zip(sk_recalls, sk_precisions):
    interp_precision = np.interp(mean_recall, recall_temp[::-1], precision_temp[::-1])
    precisions.append(interp_precision)

mean_precision = np.mean(precisions, axis=0)

sk_aucs = np.mean(sk_aucs)
sk_average_precisions= np.mean(sk_average_precisions)


#############################################    3MGCN_3spilt   end   ############################################################

prob_matrix_avg = np.zeros((A.shape[0], A.shape[1]))
iter_ = 0
out = []                    # 用于存储每一折的训练集和测试集索引
test_label_score = {}       # 存储测试标签和预测得分的字典

#############################################    6_MGCN  begin   ############################################################

microbiome_gcn = MultiDimensionalGCN_6(in_channels=331, hidden_channels=128, out_channels=64, num_dimensions=5).to(device)
disease_gcn = MultiDimensionalGCN_6(in_channels=331, hidden_channels=128, out_channels=64, num_dimensions=5).to(device)

len_micro_out = 64
len_disease_out = 64

# 优化器
optimizer1 = torch.optim.Adam(microbiome_gcn.parameters(), lr=0.01)
optimizer2 = torch.optim.Adam(disease_gcn.parameters(), lr=0.01)

A_tensor = torch.tensor(A.values, dtype=torch.float32).to(device)
prob_matrix_avg = np.zeros((A.shape[1], A.shape[0]))



fold_data = []
for cl, (train_index, test_index) in enumerate(kf.split(samples)):  # 循环每一折的训练和测试集

    print('############ {} fold #############'.format(cl))  # 打印当前折数
    out.append([train_index, test_index])  # 将训练和测试集索引存入列表中
    iter = iter + 1  # 迭代次数加1
    train_samples = samples[train_index, :]  # 获取当前折的训练集样本
    test_samples = samples[test_index, :]  # 获取当前折的测试集样本
    train_n = train_samples.shape[0]  # 获取训练集样本数量
    test_n = test_samples.shape[0]  # 获取测试集样本数量 len_micro

    micro_input_dim = len_micro_out
    disease_input_dim = len_disease_out

    fold_data_dict = {
        'final_matrix_train': [],
        'final_label_train': [],
        'final_matrix_test': [],
        'final_label_test': []
    }

    microbiome_gcn.train()
    disease_gcn.train()
    # vae_model.train()
    for epoch in range(50):
        loss = 0
        optimizer1.zero_grad()
        optimizer2.zero_grad()

        micro_out = microbiome_gcn(micro1, micro2,micro3,micro4,micro5,micro_edge_indices,micor_edge_weights, non_zero_micro)
        disease_out = disease_gcn(dis1,dis2,dis3,dis4,dis5,dis_edge_indices,dis_edge_weights, non_zero_disease)

        micro_out=F.normalize(micro_out, p=2, dim=1)
        disease_out = F.normalize(disease_out, p=2, dim=1)

        train_list = np.array(train_samples)
        test_list = np.array(test_samples)
        train_list_tensor = torch.tensor(train_list, dtype=torch.float32).to(device)
        test_list_tensor = torch.tensor(test_list, dtype=torch.float32).to(device)
        pre_predicted_matrix = torch.mm(micro_out,disease_out.T)
        predicted_matrix = (pre_predicted_matrix - pre_predicted_matrix.min()) / (pre_predicted_matrix.max() - pre_predicted_matrix.min()).to(torch.float32)

        train_labels = train_list_tensor[:, 2]  # 实际标签
        indices = train_list_tensor[:, :2].long()  # 确保索引为整数类型
        #train_label = predicted_matrix[indices[:, 0], indices[:, 1]]  # 使用张量索引获取预测值
        train_label = predicted_matrix[indices[:, 0], indices[:, 1]]
        train_label = train_label.double()
        train_labels = train_labels.double()
        loss_l2 = lambda_l2 * torch.norm(predicted_matrix, p='fro')
        # 现在 train_label 和 train_labels 都是张量，并且可以在计算损失时保持梯度追踪
        matrix_diff_loss = torch.mean((predicted_matrix - A_tensor.T) ** 2)
        constrate_loss = 0

        loss = lambda_mse * criterion(train_label,train_labels).to(torch.float32)  + loss_l2

        loss.backward(retain_graph=True)
        optimizer1.step()
        optimizer2.step()
        print(f'Epoch {epoch + 1}/{50}, Loss: {loss.item()}')

    microbiome_gcn.eval()
    disease_gcn.eval()

    with torch.no_grad():
        pre_prob_matrix = torch.mm(micro_out, disease_out.T)
        prob_matrix = (pre_prob_matrix - pre_prob_matrix.min()) / (pre_prob_matrix.max() - pre_prob_matrix.min())
        prob_matrix_np = prob_matrix.cpu().numpy()  # Ensure the matrix is on CPU
        prob_matrix_avg += prob_matrix_np

        unique_test_list_tensor = test_list_tensor
        test_labels = unique_test_list_tensor[:, 2].cpu().numpy()  # Move to CPU if necessary
        indices = unique_test_list_tensor[:, :2].long()  # Ensure indices are long

        perdcit_score = prob_matrix[indices[:, 0], indices[:, 1]].cpu().numpy()  # Move to CPU if necessary
        perdcit_label = [1 if prob >= 0.33 else 0 for prob in perdcit_score]

        viz = metrics.RocCurveDisplay.from_predictions(test_labels, perdcit_score,
                                                       name='ROC fold {}'.format(cl),
                                                       color=colors[cl],
                                                       alpha=0.6, lw=2, ax=ax)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)  # 对TPR进行插值
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)  # 将插值后的TPR添加到列表中
        aucs.append(viz.roc_auc)  # 将每一次交叉验证的ROC AUC值添加到aucs列表中。

        fpr_temp, tpr_temp, _ = roc_curve(test_labels, perdcit_score)
        roc_auc = auc(fpr_temp, tpr_temp)
        sk_fpr_6_MGCN.append(fpr_temp)
        sk_tprs_6_MGCN.append(tpr_temp)
        sk_aucs_6_MGCN.append(roc_auc)

        # 计算Precision-Recall曲线和AUPR
        precision_temp, recall_temp, _ = precision_recall_curve(test_labels, perdcit_score)
        average_precision = average_precision_score(test_labels, perdcit_score)
        sk_precisions_6_MGCN.append(precision_temp)
        sk_recalls_6_MGCN.append(recall_temp)
        sk_average_precisions_6_MGCN.append(average_precision)
        test_label_score[cl] = [test_labels, perdcit_score]
        torch.cuda.empty_cache()

prob_matrix_avg = prob_matrix_avg / k_split

mean_fpr_6_MGCN = np.linspace(0, 1, 100)
mean_recall_6_MGCN = np.linspace(0, 1, 100)
tprs = []
precisions = []
for fpr_temp, tpr_temp in zip(sk_fpr_6_MGCN , sk_tprs_6_MGCN ):
    interp_tpr = np.interp(mean_fpr_6_MGCN, fpr_temp, tpr_temp)
    interp_tpr[0] = 0.0  # 确保曲线从 0 开始
    tprs.append(interp_tpr)
mean_tpr_6_MGCN = np.mean(tprs, axis=0)

mean_tpr_6_MGCN[-1] = 1.0  # 确保曲线以 1 结束

for recall_temp, precision_temp in zip(sk_recalls_6_MGCN , sk_precisions_6_MGCN ):
    interp_precision = np.interp(mean_recall_6_MGCN, recall_temp[::-1], precision_temp[::-1])
    precisions.append(interp_precision)
mean_precision_6_MGCN = np.mean(precisions, axis=0)
sk_aucs_6_MGCN = np.mean(sk_aucs_6_MGCN )
sk_average_precisions_6_MGCN = np.mean(sk_average_precisions_6_MGCN)

#############################################    6_MGCN  end   ############################################################



prob_matrix_avg = np.zeros((A.shape[0], A.shape[1]))
iter_ = 0
out = []                    # 用于存储每一折的训练集和测试集索引
test_label_score = {}       # 存储测试标签和预测得分的字典

#############################################    _3_MGCN_3_GCN  begin   ############################################################

microbiome_gcn = MultiDimensionalGCN_3_add_3_Nor(in_channels=331, hidden_channels=128, out_channels=64, num_dimensions=5).to(device)
disease_gcn = MultiDimensionalGCN_3_add_3_Nor(in_channels=331, hidden_channels=128, out_channels=64, num_dimensions=5).to(device)

len_micro_out = 64
len_disease_out = 64

# 优化器
optimizer1 = torch.optim.Adam(microbiome_gcn.parameters(), lr=0.01)
optimizer2 = torch.optim.Adam(disease_gcn.parameters(), lr=0.01)

A_tensor = torch.tensor(A.values, dtype=torch.float32).to(device)
prob_matrix_avg = np.zeros((A.shape[1], A.shape[0]))



fold_data = []
for cl, (train_index, test_index) in enumerate(kf.split(samples)):  # 循环每一折的训练和测试集

    print('############ {} fold #############'.format(cl))  # 打印当前折数
    out.append([train_index, test_index])  # 将训练和测试集索引存入列表中
    iter = iter + 1  # 迭代次数加1
    train_samples = samples[train_index, :]  # 获取当前折的训练集样本
    test_samples = samples[test_index, :]  # 获取当前折的测试集样本
    train_n = train_samples.shape[0]  # 获取训练集样本数量
    test_n = test_samples.shape[0]  # 获取测试集样本数量 len_micro

    micro_input_dim = len_micro_out
    disease_input_dim = len_disease_out

    fold_data_dict = {
        'final_matrix_train': [],
        'final_label_train': [],
        'final_matrix_test': [],
        'final_label_test': []
    }

    microbiome_gcn.train()
    disease_gcn.train()
    # vae_model.train()
    for epoch in range(50):
        loss = 0
        optimizer1.zero_grad()
        optimizer2.zero_grad()

        micro_out = microbiome_gcn(micro1, micro2,micro3,micro4,micro5,micro_edge_indices,micor_edge_weights, non_zero_micro)
        disease_out = disease_gcn(dis1,dis2,dis3,dis4,dis5,dis_edge_indices,dis_edge_weights, non_zero_disease)

        micro_out=F.normalize(micro_out, p=2, dim=1)
        disease_out = F.normalize(disease_out, p=2, dim=1)

        train_list = np.array(train_samples)
        test_list = np.array(test_samples)
        train_list_tensor = torch.tensor(train_list, dtype=torch.float32).to(device)
        test_list_tensor = torch.tensor(test_list, dtype=torch.float32).to(device)
        pre_predicted_matrix = torch.mm(micro_out,disease_out.T)
        predicted_matrix = (pre_predicted_matrix - pre_predicted_matrix.min()) / (pre_predicted_matrix.max() - pre_predicted_matrix.min()).to(torch.float32)

        train_labels = train_list_tensor[:, 2]  # 实际标签
        indices = train_list_tensor[:, :2].long()  # 确保索引为整数类型
        #train_label = predicted_matrix[indices[:, 0], indices[:, 1]]  # 使用张量索引获取预测值
        train_label = predicted_matrix[indices[:, 0], indices[:, 1]]
        train_label = train_label.double()
        train_labels = train_labels.double()
        loss_l2 = lambda_l2 * torch.norm(predicted_matrix, p='fro')
        # 现在 train_label 和 train_labels 都是张量，并且可以在计算损失时保持梯度追踪
        matrix_diff_loss = torch.mean((predicted_matrix - A_tensor.T) ** 2)
        constrate_loss = 0

        loss = lambda_mse * criterion(train_label,train_labels).to(torch.float32)  + loss_l2

        loss.backward(retain_graph=True)
        optimizer1.step()
        optimizer2.step()
        print(f'Epoch {epoch + 1}/{50}, Loss: {loss.item()}')

    microbiome_gcn.eval()
    disease_gcn.eval()

    with torch.no_grad():
        pre_prob_matrix = torch.mm(micro_out, disease_out.T)
        prob_matrix = (pre_prob_matrix - pre_prob_matrix.min()) / (pre_prob_matrix.max() - pre_prob_matrix.min())
        prob_matrix_np = prob_matrix.cpu().numpy()  # Ensure the matrix is on CPU
        prob_matrix_avg += prob_matrix_np

        unique_test_list_tensor = test_list_tensor
        test_labels = unique_test_list_tensor[:, 2].cpu().numpy()  # Move to CPU if necessary
        indices = unique_test_list_tensor[:, :2].long()  # Ensure indices are long

        perdcit_score = prob_matrix[indices[:, 0], indices[:, 1]].cpu().numpy()  # Move to CPU if necessary
        perdcit_label = [1 if prob >= 0.33 else 0 for prob in perdcit_score]

        viz = metrics.RocCurveDisplay.from_predictions(test_labels, perdcit_score,
                                                       name='ROC fold {}'.format(cl),
                                                       color=colors[cl],
                                                       alpha=0.6, lw=2, ax=ax)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)  # 对TPR进行插值
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)  # 将插值后的TPR添加到列表中
        aucs.append(viz.roc_auc)  # 将每一次交叉验证的ROC AUC值添加到aucs列表中。

        fpr_temp, tpr_temp, _ = roc_curve(test_labels, perdcit_score)
        roc_auc = auc(fpr_temp, tpr_temp)
        sk_fpr_3_MGCN_3_GCN.append(fpr_temp)
        sk_tprs_3_MGCN_3_GCN.append(tpr_temp)
        sk_aucs_3_MGCN_3_GCN.append(roc_auc)

        # 计算Precision-Recall曲线和AUPR
        precision_temp, recall_temp, _ = precision_recall_curve(test_labels, perdcit_score)
        average_precision = average_precision_score(test_labels, perdcit_score)
        sk_precisions_3_MGCN_3_GCN.append(precision_temp)
        sk_recalls_3_MGCN_3_GCN.append(recall_temp)
        sk_average_precisions_3_MGCN_3_GCN.append(average_precision)
        test_label_score[cl] = [test_labels, perdcit_score]
        torch.cuda.empty_cache()

prob_matrix_avg = prob_matrix_avg / k_split

mean_fpr_3_MGCN_3_GCN = np.linspace(0, 1, 100)
mean_recall_3_MGCN_3_GCN = np.linspace(0, 1, 100)
tprs = []
precisions = []
for fpr_temp, tpr_temp in zip(sk_fpr_3_MGCN_3_GCN , sk_tprs_3_MGCN_3_GCN ):
    interp_tpr = np.interp(mean_fpr_3_MGCN_3_GCN, fpr_temp, tpr_temp)
    interp_tpr[0] = 0.0  # 确保曲线从 0 开始
    tprs.append(interp_tpr)
mean_tpr_3_MGCN_3_GCN = np.mean(tprs, axis=0)

mean_tpr_3_MGCN_3_GCN[-1] = 1.0  # 确保曲线以 1 结束

for recall_temp, precision_temp in zip(sk_recalls_3_MGCN_3_GCN , sk_precisions_3_MGCN_3_GCN ):
    interp_precision = np.interp(mean_recall_3_MGCN_3_GCN, recall_temp[::-1], precision_temp[::-1])
    precisions.append(interp_precision)
mean_precision_3_MGCN_3_GCN = np.mean(precisions, axis=0)
sk_aucs_3_MGCN_3_GCN = np.mean(sk_aucs_3_MGCN_3_GCN)
sk_average_precisions_3_MGCN_3_GCN = np.mean(sk_average_precisions_3_MGCN_3_GCN)
#############################################    _3_MGCN_3_GCN  end   ############################################################

prob_matrix_avg = np.zeros((A.shape[0], A.shape[1]))
iter_ = 0
out = []                    # 用于存储每一折的训练集和测试集索引
test_label_score = {}       # 存储测试标签和预测得分的字典

#############################################    _6_GCN  begin   ############################################################

microbiome_gcn = MultiDimensionalGCN_6_Nor(in_channels=331, hidden_channels=128, out_channels=64, num_dimensions=5).to(device)
disease_gcn = MultiDimensionalGCN_6_Nor(in_channels=331, hidden_channels=128, out_channels=64, num_dimensions=5).to(device)

len_micro_out = 64
len_disease_out = 64

# 优化器
optimizer1 = torch.optim.Adam(microbiome_gcn.parameters(), lr=0.01)
optimizer2 = torch.optim.Adam(disease_gcn.parameters(), lr=0.01)

A_tensor = torch.tensor(A.values, dtype=torch.float32).to(device)
prob_matrix_avg = np.zeros((A.shape[1], A.shape[0]))



fold_data = []
for cl, (train_index, test_index) in enumerate(kf.split(samples)):  # 循环每一折的训练和测试集

    print('############ {} fold #############'.format(cl))  # 打印当前折数
    out.append([train_index, test_index])  # 将训练和测试集索引存入列表中
    iter = iter + 1  # 迭代次数加1
    train_samples = samples[train_index, :]  # 获取当前折的训练集样本
    test_samples = samples[test_index, :]  # 获取当前折的测试集样本
    train_n = train_samples.shape[0]  # 获取训练集样本数量
    test_n = test_samples.shape[0]  # 获取测试集样本数量 len_micro

    micro_input_dim = len_micro_out
    disease_input_dim = len_disease_out

    fold_data_dict = {
        'final_matrix_train': [],
        'final_label_train': [],
        'final_matrix_test': [],
        'final_label_test': []
    }

    microbiome_gcn.train()
    disease_gcn.train()
    # vae_model.train()
    for epoch in range(50):
        loss = 0
        optimizer1.zero_grad()
        optimizer2.zero_grad()

        micro_out = microbiome_gcn(micro1, micro2,micro3,micro4,micro5,micro_edge_indices,micor_edge_weights, non_zero_micro)
        disease_out = disease_gcn(dis1,dis2,dis3,dis4,dis5,dis_edge_indices,dis_edge_weights, non_zero_disease)

        micro_out=F.normalize(micro_out, p=2, dim=1)
        disease_out = F.normalize(disease_out, p=2, dim=1)

        train_list = np.array(train_samples)
        test_list = np.array(test_samples)
        train_list_tensor = torch.tensor(train_list, dtype=torch.float32).to(device)
        test_list_tensor = torch.tensor(test_list, dtype=torch.float32).to(device)
        pre_predicted_matrix = torch.mm(micro_out,disease_out.T)
        predicted_matrix = (pre_predicted_matrix - pre_predicted_matrix.min()) / (pre_predicted_matrix.max() - pre_predicted_matrix.min()).to(torch.float32)

        train_labels = train_list_tensor[:, 2]  # 实际标签
        indices = train_list_tensor[:, :2].long()  # 确保索引为整数类型
        #train_label = predicted_matrix[indices[:, 0], indices[:, 1]]  # 使用张量索引获取预测值
        train_label = predicted_matrix[indices[:, 0], indices[:, 1]]
        train_label = train_label.double()
        train_labels = train_labels.double()
        loss_l2 = lambda_l2 * torch.norm(predicted_matrix, p='fro')
        # 现在 train_label 和 train_labels 都是张量，并且可以在计算损失时保持梯度追踪
        matrix_diff_loss = torch.mean((predicted_matrix - A_tensor.T) ** 2)
        constrate_loss = 0

        loss = lambda_mse * criterion(train_label,train_labels).to(torch.float32)  + loss_l2

        loss.backward(retain_graph=True)
        optimizer1.step()
        optimizer2.step()
        print(f'Epoch {epoch + 1}/{50}, Loss: {loss.item()}')

    microbiome_gcn.eval()
    disease_gcn.eval()

    with torch.no_grad():
        pre_prob_matrix = torch.mm(micro_out, disease_out.T)
        prob_matrix = (pre_prob_matrix - pre_prob_matrix.min()) / (pre_prob_matrix.max() - pre_prob_matrix.min())
        prob_matrix_np = prob_matrix.cpu().numpy()  # Ensure the matrix is on CPU
        prob_matrix_avg += prob_matrix_np

        unique_test_list_tensor = test_list_tensor
        test_labels = unique_test_list_tensor[:, 2].cpu().numpy()  # Move to CPU if necessary
        indices = unique_test_list_tensor[:, :2].long()  # Ensure indices are long

        perdcit_score = prob_matrix[indices[:, 0], indices[:, 1]].cpu().numpy()  # Move to CPU if necessary
        perdcit_label = [1 if prob >= 0.33 else 0 for prob in perdcit_score]

        viz = metrics.RocCurveDisplay.from_predictions(test_labels, perdcit_score,
                                                       name='ROC fold {}'.format(cl),
                                                       color=colors[cl],
                                                       alpha=0.6, lw=2, ax=ax)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)  # 对TPR进行插值
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)  # 将插值后的TPR添加到列表中
        aucs.append(viz.roc_auc)  # 将每一次交叉验证的ROC AUC值添加到aucs列表中。

        fpr_temp, tpr_temp, _ = roc_curve(test_labels, perdcit_score)
        roc_auc = auc(fpr_temp, tpr_temp)
        sk_fpr_6_GCN .append(fpr_temp)
        sk_tprs_6_GCN .append(tpr_temp)
        sk_aucs_6_GCN .append(roc_auc)

        # 计算Precision-Recall曲线和AUPR
        precision_temp, recall_temp, _ = precision_recall_curve(test_labels, perdcit_score)
        average_precision = average_precision_score(test_labels, perdcit_score)
        sk_precisions_6_GCN .append(precision_temp)
        sk_recalls_6_GCN .append(recall_temp)
        sk_average_precisions_6_GCN .append(average_precision)
        test_label_score[cl] = [test_labels, perdcit_score]
        torch.cuda.empty_cache()

prob_matrix_avg = prob_matrix_avg / k_split

mean_fpr_6_GCN  = np.linspace(0, 1, 100)
mean_recall_6_GCN  = np.linspace(0, 1, 100)
tprs = []
precisions = []
for fpr_temp, tpr_temp in zip(sk_fpr_6_GCN  , sk_tprs_6_GCN  ):
    interp_tpr = np.interp(mean_fpr_6_GCN , fpr_temp, tpr_temp)
    interp_tpr[0] = 0.0  # 确保曲线从 0 开始
    tprs.append(interp_tpr)
mean_tpr_6_GCN  = np.mean(tprs, axis=0)

mean_tpr_6_GCN [-1] = 1.0  # 确保曲线以 1 结束

for recall_temp, precision_temp in zip(sk_recalls_6_GCN  , sk_precisions_6_GCN  ):
    interp_precision = np.interp(mean_recall_6_GCN , recall_temp[::-1], precision_temp[::-1])
    precisions.append(interp_precision)
mean_precision_6_GCN  = np.mean(precisions, axis=0)
sk_aucs_6_GCN  = np.mean(sk_aucs_6_GCN )
sk_average_precisions_6_GCN  = np.mean(sk_average_precisions_6_GCN )
#############################################    _6_GCN  end   ############################################################

prob_matrix_avg = np.zeros((A.shape[0], A.shape[1]))
iter_ = 0
out = []                    # 用于存储每一折的训练集和测试集索引
test_label_score = {}       # 存储测试标签和预测得分的字典

#############################################    _6_spilt  begin   ############################################################

microbiome_gcn = MultiDimensionalGCN_6_Nor(in_channels=331, hidden_channels=128, out_channels=64, num_dimensions=5).to(device)
disease_gcn = MultiDimensionalGCN_6_Nor(in_channels=331, hidden_channels=128, out_channels=64, num_dimensions=5).to(device)

len_micro_out = 64
len_disease_out = 64

# 优化器
optimizer1 = torch.optim.Adam(microbiome_gcn.parameters(), lr=0.01)
optimizer2 = torch.optim.Adam(disease_gcn.parameters(), lr=0.01)

A_tensor = torch.tensor(A.values, dtype=torch.float32).to(device)
prob_matrix_avg = np.zeros((A.shape[1], A.shape[0]))



fold_data = []
for cl, (train_index, test_index) in enumerate(kf.split(samples)):  # 循环每一折的训练和测试集

    print('############ {} fold #############'.format(cl))  # 打印当前折数
    out.append([train_index, test_index])  # 将训练和测试集索引存入列表中
    iter = iter + 1  # 迭代次数加1
    train_samples = samples[train_index, :]  # 获取当前折的训练集样本
    test_samples = samples[test_index, :]  # 获取当前折的测试集样本
    train_n = train_samples.shape[0]  # 获取训练集样本数量
    test_n = test_samples.shape[0]  # 获取测试集样本数量 len_micro

    micro_input_dim = len_micro_out
    disease_input_dim = len_disease_out

    fold_data_dict = {
        'final_matrix_train': [],
        'final_label_train': [],
        'final_matrix_test': [],
        'final_label_test': []
    }

    microbiome_gcn.train()
    disease_gcn.train()
    # vae_model.train()
    for epoch in range(50):
        loss = 0
        optimizer1.zero_grad()
        optimizer2.zero_grad()

        micro_out = microbiome_gcn(micro1, micro2,micro3,micro4,micro5,micro_edge_indices,micor_edge_weights, non_zero_micro)
        disease_out = disease_gcn(dis1,dis2,dis3,dis4,dis5,dis_edge_indices,dis_edge_weights, non_zero_disease)

        micro_out=F.normalize(micro_out, p=2, dim=1)
        disease_out = F.normalize(disease_out, p=2, dim=1)

        train_list = np.array(train_samples)
        test_list = np.array(test_samples)
        train_list_tensor = torch.tensor(train_list, dtype=torch.float32).to(device)
        test_list_tensor = torch.tensor(test_list, dtype=torch.float32).to(device)
        pre_predicted_matrix = torch.mm(micro_out,disease_out.T)
        predicted_matrix = (pre_predicted_matrix - pre_predicted_matrix.min()) / (pre_predicted_matrix.max() - pre_predicted_matrix.min()).to(torch.float32)

        train_labels = train_list_tensor[:, 2]  # 实际标签
        indices = train_list_tensor[:, :2].long()  # 确保索引为整数类型
        #train_label = predicted_matrix[indices[:, 0], indices[:, 1]]  # 使用张量索引获取预测值
        train_label = predicted_matrix[indices[:, 0], indices[:, 1]]
        train_label = train_label.double()
        train_labels = train_labels.double()
        loss_l2 = lambda_l2 * torch.norm(predicted_matrix, p='fro')
        # 现在 train_label 和 train_labels 都是张量，并且可以在计算损失时保持梯度追踪
        matrix_diff_loss = torch.mean((predicted_matrix - A_tensor.T) ** 2)
        constrate_loss = 0

        loss = lambda_mse * criterion(train_label,train_labels).to(torch.float32)  + loss_l2

        loss.backward(retain_graph=True)
        optimizer1.step()
        optimizer2.step()
        print(f'Epoch {epoch + 1}/{50}, Loss: {loss.item()}')

    microbiome_gcn.eval()
    disease_gcn.eval()

    with torch.no_grad():
        pre_prob_matrix = torch.mm(micro_out, disease_out.T)
        prob_matrix = (pre_prob_matrix - pre_prob_matrix.min()) / (pre_prob_matrix.max() - pre_prob_matrix.min())
        prob_matrix_np = prob_matrix.cpu().numpy()  # Ensure the matrix is on CPU
        prob_matrix_avg += prob_matrix_np

        unique_test_list_tensor = test_list_tensor
        test_labels = unique_test_list_tensor[:, 2].cpu().numpy()  # Move to CPU if necessary
        indices = unique_test_list_tensor[:, :2].long()  # Ensure indices are long

        perdcit_score = prob_matrix[indices[:, 0], indices[:, 1]].cpu().numpy()  # Move to CPU if necessary
        perdcit_label = [1 if prob >= 0.33 else 0 for prob in perdcit_score]

        viz = metrics.RocCurveDisplay.from_predictions(test_labels, perdcit_score,
                                                       name='ROC fold {}'.format(cl),
                                                       color=colors[cl],
                                                       alpha=0.6, lw=2, ax=ax)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)  # 对TPR进行插值
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)  # 将插值后的TPR添加到列表中
        aucs.append(viz.roc_auc)  # 将每一次交叉验证的ROC AUC值添加到aucs列表中。

        fpr_temp, tpr_temp, _ = roc_curve(test_labels, perdcit_score)
        roc_auc = auc(fpr_temp, tpr_temp)
        sk_fpr_6_spilt .append(fpr_temp)
        sk_tprs_6_spilt .append(tpr_temp)
        sk_aucs_6_spilt .append(roc_auc)

        # 计算Precision-Recall曲线和AUPR
        precision_temp, recall_temp, _ = precision_recall_curve(test_labels, perdcit_score)
        average_precision = average_precision_score(test_labels, perdcit_score)
        sk_precisions_6_spilt .append(precision_temp)
        sk_recalls_6_spilt .append(recall_temp)
        sk_average_precisions_6_spilt .append(average_precision)
        test_label_score[cl] = [test_labels, perdcit_score]
        torch.cuda.empty_cache()

prob_matrix_avg = prob_matrix_avg / k_split

mean_fpr_6_spilt  = np.linspace(0, 1, 100)
mean_recall_6_spilt  = np.linspace(0, 1, 100)
tprs = []
precisions = []
for fpr_temp, tpr_temp in zip(sk_fpr_6_spilt  , sk_tprs_6_spilt  ):
    interp_tpr = np.interp(mean_fpr_6_spilt , fpr_temp, tpr_temp)
    interp_tpr[0] = 0.0  # 确保曲线从 0 开始
    tprs.append(interp_tpr)
mean_tpr_6_spilt  = np.mean(tprs, axis=0)

mean_tpr_6_spilt [-1] = 1.0  # 确保曲线以 1 结束

for recall_temp, precision_temp in zip(sk_recalls_6_spilt  , sk_precisions_6_spilt  ):
    interp_precision = np.interp(mean_recall_6_spilt , recall_temp[::-1], precision_temp[::-1])
    precisions.append(interp_precision)
mean_precision_6_spilt  = np.mean(precisions, axis=0)
sk_aucs_6_spilt  = np.mean(sk_aucs_6_spilt )
sk_average_precisions_6_spilt  = np.mean(sk_average_precisions_6_spilt )

#############################################    _6_spilt  end   ############################################################

def compute_mean(values):
    return np.mean(values, axis=0)


fig2, axs2 = plt.subplots(1, 1, figsize=(5, 5))
model_labels = ['MGMDA', 'MGMDA_6_MGCN', 'MGMDA_3MGCN_3_GCN','MGMDA_6_GCN','MGMDA_6_spilt']
model_precisions = [mean_precision, mean_precision_6_MGCN, mean_precision_3_MGCN_3_GCN,mean_precision_6_GCN,mean_precision_6_spilt ]
model_recalls = [mean_recall, mean_recall_6_MGCN, mean_recall_3_MGCN_3_GCN,mean_recall_6_GCN,mean_recall_6_spilt]
model_auprs = [sk_average_precisions, sk_average_precisions_6_MGCN, sk_average_precisions_3_MGCN_3_GCN,sk_average_precisions_6_GCN,sk_average_precisions_6_spilt]

for precisions, recalls, auprs, label in zip(model_precisions, model_recalls, model_auprs, model_labels):

    axs2.step(recalls, precisions, where='post', label=f'{label} AUPR={auprs:.2f}')

axs2.plot([0, 1], [1, 0], '--', color='r', label='Random')
axs2.set_xlabel('Recall')
axs2.set_ylabel('Precision')
axs2.set_ylim([-0.05, 1.05])
axs2.set_xlim([-0.05, 1.05])
axs2.set_title('Precision-Recall curve')
axs2.legend(loc="best")
plt.show()


# 绘制平均ROC曲线
fig3, axs3 = plt.subplots(1, 1, figsize=(5, 5))
model_tprs = [mean_tpr, mean_tpr_6_MGCN, mean_tpr_3_MGCN_3_GCN,mean_tpr_6_GCN,mean_tpr_6_spilt]
model_fprs = [mean_fpr, mean_fpr_6_MGCN, mean_fpr_3_MGCN_3_GCN,mean_fpr_6_GCN,mean_fpr_6_spilt]
model_aucs = [sk_aucs, sk_aucs_6_MGCN, sk_aucs_3_MGCN_3_GCN,sk_aucs_6_GCN,sk_aucs_6_spilt]

for tprs, fprs, aucs, label in zip(model_tprs, model_fprs, model_aucs, model_labels):
    axs3.step(fprs, tprs, where='post', label=f'{label} AUC={aucs:.2f}')

axs3.plot([0, 1], [0, 1], '--', color='r', label='Random')
axs3.set_xlabel('FPR')
axs3.set_ylabel('TPR')
axs3.set_ylim([-0.05, 1.05])
axs3.set_xlim([-0.05, 1.05])
axs3.set_title('ROC curve')
axs3.legend(loc="best")
plt.show()