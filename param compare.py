import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import pandas as pd
from utils import *
import torch.nn.functional as F
from sklearn.model_selection import KFold
from sklearn.metrics import (roc_curve, auc, precision_recall_curve, average_precision_score,
                             f1_score, accuracy_score, recall_score, precision_score, confusion_matrix)
from model import LinearEncoder, create_multi_dimensional_data, MultiDimensionalGCN,MicroDiseaseModel

#torch.autograd.set_detect_anomaly(True)
# 检查CUDA是否可用
n = 4
out = []  # 用于存储每一折的训练集和测试集索引
k_split = 5
set_seed(123)
lambda_mse = 4
lambda_l2 = 3e-2
lambda_constrate = 5
matplotlib.use('TkAgg')
criterion = torch.nn.MSELoss()
kf = KFold(n_splits=k_split, shuffle=True, random_state=123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',  # 使用颜色编码定义颜色
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

#
# lambda_l2_list=[1e-2,2e-2,3e-2,4e-2,5e-2,6e-2,7e-2,8e-2,9e-2,1e-1]
lambda_l2_list=[1e-2,2e-2,3e-2,4e-2,5e-2,6e-2,7e-2,8e-2,9e-2,1e-1]
lambda_mse_list=[0,1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 ,9 ]
lambda_constrate_list=[0,0.1 , 0.2 , 0.3 , 0.4 , 0.5 , 0.6 , 0.7 , 0.8 ,0.9 ]
# lambda_constrate_list=[1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 ,9 , 10]
n_list=[1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 ,9 , 10]


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




# 读取数据
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

lambda_mse_AUC = []
lambda_mse_AUPR = []

lambda_constrate_AUC = []
lambda_constrate_AUPR = []

lambda_l2_AUC = []
lambda_l2_AUPR = []

n_AUC = []
n_AUPR = []

samples = get_all_the_samples_old(A.T.values)
samples = np.array(samples)



def test(lambda_l2,lambda_mse):
    # lambda_mse = 4
    # lambda_l2 = 8e-2
    lambda_constrate = 5
    lambda_l2 = lambda_l2
    lambda_mse = lambda_mse


    sk_tprs = []
    sk_aucs = []
    sk_precisions = []
    sk_recalls = []
    sk_average_precisions = []
    sk_fpr = []
    test_label_score = {}

    micro_low_features = A.T  # 微生物的低级特征是矩阵A的列向量
    disease_low_features = A  # 疾病的低级特征是矩阵A的行向量

    micro_low_features_tensor = torch.Tensor(micro_low_features.values).to(device)
    disease_low_features_tensor = torch.Tensor(disease_low_features.values).to(device)

    # 将微生物关联矩阵转换为多维图数据
    micro_data, micro1, micro2, micro3, micro4, micro5, micro_edge_indices, micor_edge_weights = create_multi_dimensional_data(
        microbiome_matrices, device, micro_low_features_tensor)
    # 将疾病关联矩阵转换为多维图数据
    disease_data, dis1, dis2, dis3, dis4, dis5, dis_edge_indices, dis_edge_weights = create_multi_dimensional_data(
        disease_matrices, device, disease_low_features_tensor)

    # HMDAD
    microbiome_gcn = MultiDimensionalGCN(in_channels=331, hidden_channels=128, out_channels=64, num_dimensions=5).to(
        device)
    disease_gcn = MultiDimensionalGCN(in_channels=331, hidden_channels=128, out_channels=64, num_dimensions=5).to(
        device)


    # 优化器
    optimizer1 = torch.optim.Adam(microbiome_gcn.parameters(), lr=0.01)
    optimizer2 = torch.optim.Adam(disease_gcn.parameters(), lr=0.01)
    # optimizer3 = torch.optim.Adam(vae_model.parameters(), lr=0.01)

    A_tensor = torch.tensor(A.values, dtype=torch.float32).to(device)
    prob_matrix_avg = np.zeros((A.shape[1], A.shape[0]))

    fold_data = []
    iter = 0
    for cl, (train_index, test_index) in enumerate(kf.split(samples)):  # 循环每一折的训练和测试集

        print('############ {} fold #############'.format(cl))  # 打印当前折数
        out.append([train_index, test_index])  # 将训练和测试集索引存入列表中
        iter = iter + 1  # 迭代次数加1
        train_samples = samples[train_index, :]  # 获取当前折的训练集样本
        test_samples = samples[test_index, :]  # 获取当前折的测试集样本



        microbiome_gcn.train()
        disease_gcn.train()
        # vae_model.train()
        for epoch in range(200):
            loss = 0
            optimizer1.zero_grad()
            optimizer2.zero_grad()

            micro_out = microbiome_gcn(micro1, micro2, micro3, micro4, micro5, micro_edge_indices, micor_edge_weights,
                                       non_zero_micro)
            disease_out = disease_gcn(dis1, dis2, dis3, dis4, dis5, dis_edge_indices, dis_edge_weights,
                                      non_zero_disease)
            micro_out = F.normalize(micro_out, p=2, dim=1)
            disease_out = F.normalize(disease_out, p=2, dim=1)
            train_list = np.array(train_samples)
            test_list = np.array(test_samples)
            train_list_tensor = torch.tensor(train_list, dtype=torch.float32).to(device)
            test_list_tensor = torch.tensor(test_list, dtype=torch.float32).to(device)
            pre_predicted_matrix = torch.mm(micro_out, disease_out.T)
            predicted_matrix = (pre_predicted_matrix - pre_predicted_matrix.min()) / (
                        pre_predicted_matrix.max() - pre_predicted_matrix.min()).to(torch.float32)

            train_labels = train_list_tensor[:, 2]  # 实际标签
            indices = train_list_tensor[:, :2].long()  # 确保索引为整数类型
            # train_label = predicted_matrix[indices[:, 0], indices[:, 1]]  # 使用张量索引获取预测值
            train_label = predicted_matrix[indices[:, 0], indices[:, 1]]
            train_label = train_label.double()
            train_labels = train_labels.double()
            loss_l2 = lambda_l2 * torch.norm(predicted_matrix, p='fro')
            # 现在 train_label 和 train_labels 都是张量，并且可以在计算损失时保持梯度追踪
            matrix_diff_loss = torch.mean((predicted_matrix - A_tensor.T) ** 2)
            # constrate_loss = constrate_loss_calculate(i_select,i_hat_select,j_select,j_hat_select)      #待修改
            constrate_loss = 0
            # constrate_loss = 0
            # print('constrate_loss=',constrate_loss)
            loss = lambda_mse * criterion(train_label, train_labels).to(
                torch.float32) + loss_l2 + lambda_constrate * constrate_loss

            # loss.backward()
            loss.backward(retain_graph=True)
            # if epoch % 2 ==0
            optimizer1.step()
            optimizer2.step()
            # optimizer3.step()

            print(f'Epoch {epoch + 1}/{200}, Loss: {loss.item()}')

        microbiome_gcn.eval()
        disease_gcn.eval()
        # vae_model.eval()

        with torch.no_grad():
            pre_prob_matrix = torch.mm(micro_out, disease_out.T)
            prob_matrix = (pre_prob_matrix - pre_prob_matrix.min()) / (pre_prob_matrix.max() - pre_prob_matrix.min())
            prob_matrix_np = prob_matrix.cpu().numpy()  # Ensure the matrix is on CPU
            prob_matrix_avg += prob_matrix_np

            # Remove duplicates in test samples
            # unique_test_list_tensor = torch.unique(test_list_tensor, dim=0)
            unique_test_list_tensor = test_list_tensor
            test_labels = unique_test_list_tensor[:, 2].cpu().numpy()  # Move to CPU if necessary
            indices = unique_test_list_tensor[:, :2].long()  # Ensure indices are long

            # Using tensor indices to get predicted scores
            perdcit_score = prob_matrix[indices[:, 0], indices[:, 1]].cpu().numpy()  # Move to CPU if necessary
            perdcit_label = [1 if prob >= 0.33 else 0 for prob in perdcit_score]

            # Visualize ROC curve
            viz = metrics.RocCurveDisplay.from_predictions(test_labels, perdcit_score,
                                                           name='ROC fold {}'.format(cl),
                                                           color=colors[cl],
                                                           alpha=0.6, lw=2, ax=ax)
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)

            # Calculate and append metrics
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

            # Calculate Precision-Recall curve and AUPR
            precision_temp, recall_temp, _ = precision_recall_curve(test_labels, perdcit_score)
            average_precision = average_precision_score(test_labels, perdcit_score)
            sk_precisions.append(precision_temp)
            sk_recalls.append(recall_temp)
            sk_average_precisions.append(average_precision)

            test_label_score[cl] = [test_labels, perdcit_score]
            # 在交叉验证的最后保存模型
            # torch.save(microbiome_gcn.state_dict(), f'microbiome_gcn_fold_{cl}.pth')
            # torch.save(disease_gcn.state_dict(), f'disease_gcn_fold_{cl}.pth')
            # Free tensors
            torch.cuda.empty_cache()

    print('############ avg score #############')
    for metric, values in metrics_summary.items():
        print(f"{metric}: {np.mean(values):.2f} ± {np.std(values):.2f}")


    print('mean AUC = ',np.mean(sk_aucs))
    print('mean AUPR = ',np.mean(sk_average_precisions))

    AUPR = np.mean(sk_average_precisions)
    AUC = np.mean(sk_aucs)
    return  AUPR,AUC





# 初始化用于保存 AUPR 和 AUC 的结果
results = []

for lambda_l2_input in lambda_l2_list:
    for lambda_mse_input in lambda_mse_list:
        AUPR,AUC = test(lambda_l2_input,lambda_mse_input)
        lambda_l2_AUPR.append(AUPR)
        lambda_l2_AUC.append(AUC)
        results.append([lambda_l2_input, lambda_mse_input, AUPR, AUC])




# 将结果保存为 CSV 文件
df = pd.DataFrame(results, columns=['lambda_l2', 'lambda_mse', 'AUPR', 'AUC'])
df.to_csv('results.csv', index=False)

# 读取 CSV 文件
df = pd.read_csv('results.csv')

# 创建数据透视表，方便绘制热图
pivot_aupr = df.pivot_table(index='lambda_l2', columns='lambda_mse', values='AUPR')
pivot_auc = df.pivot_table(index='lambda_l2', columns='lambda_mse', values='AUC')

# 绘制 AUPR 热图
plt.figure(figsize=(8, 6))
sns.heatmap(pivot_aupr, annot=True, cmap='viridis')
plt.title('AUPR Heatmap')
plt.xlabel('lambda_mse')
plt.ylabel('lambda_l2')
plt.show()

# 绘制 AUC 热图
plt.figure(figsize=(8, 6))
sns.heatmap(pivot_auc, annot=True, cmap='viridis')
plt.title('AUC Heatmap')
plt.xlabel('lambda_mse')
plt.ylabel('lambda_l2')
plt.show()




