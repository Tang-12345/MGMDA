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
import csv
#torch.autograd.set_detect_anomaly(True)
# 检查CUDA是否可用
out = []  # 用于存储每一折的训练集和测试集索引
iter = 0
n = 4
k_split = 5
set_seed(123)
lambda_mse = 4
lambda_l2 = 3e-2
lambda_constrate = 5
matplotlib.use('TkAgg')
criterion = torch.nn.MSELoss()
kf = KFold(n_splits=k_split, shuffle=True, random_state=123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']# 使用颜色编码定义颜色


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

fold_metrics = {
    'aucs': [],
    'auprs': [],
    'f1_scores': [],
    'accuracies': []
}

fold_metrics_file = 'fold_metrics.csv'
with open(fold_metrics_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Fold', 'AUC', 'AUPR', 'F1', 'Accuracy'])


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

#disbiome
# A = pd.read_csv('./dataset/disbiome/process_data/adj_matrix.csv')
# disease_chemical = pd.read_csv('./dataset/disbiome/化学-疾病/complete_disease_similarity_matrix.csv')
# disease_gene = pd.read_csv('./dataset/disbiome/基因-疾病/complete_disease_similarity_matrix.csv')
# disease_symptoms = pd.read_csv('./dataset/disbiome/疾病-症状/complete_disease_similarity_matrix.csv')
# disease_Semantics = pd.read_csv('./dataset/disbiome/疾病-语义/similarity_matrix_model2.csv', header=None)
# disease_pathway = pd.read_csv('./dataset/disbiome/疾病-通路/complete_disease_similarity_matrix.csv')
# micro_cos = pd.read_csv('./dataset/disbiome/基于关联矩阵的微生物功能/Cosine_Sim.csv')
# micro_gip = pd.read_csv('./dataset/disbiome/基于关联矩阵的微生物功能/GIP_Sim.csv')
# micro_sem = pd.read_csv('./dataset/disbiome/基于疾病语义的微生物功能/functional_similarity2_matrix.csv')
# micro_fun1 = pd.read_csv('./dataset/disbiome/微生物-功能/complete_microbe_associations_ds2_matrix.csv')
# micro_fun2 = pd.read_csv('./dataset/disbiome/微生物-功能/complete_microbe_similarities_ds2_matrix.csv')
# A = A.iloc[:, 1:]

#peryton
# A = pd.read_csv('./dataset/peryton/adjacency_matrix.csv')
# disease_chemical = pd.read_csv('./dataset/peryton/化学-疾病/complete_disease_similarity_matrix.csv')
# disease_gene = pd.read_csv('./dataset/peryton/基因-疾病/complete_disease_similarity_matrix.csv')
# disease_symptoms = pd.read_csv('./dataset/peryton/疾病-症状/complete_disease_similarity_matrix.csv')
# disease_Semantics = pd.read_csv('./dataset/peryton/疾病-语义/similarity_matrix_model2.csv', header=None)
# disease_pathway = pd.read_csv('./dataset/peryton/疾病-通路/complete_disease_similarity_matrix.csv')
# micro_cos = pd.read_csv('./dataset/peryton/基于关联矩阵的微生物功能/Cosine_Sim.csv')
# micro_gip = pd.read_csv('./dataset/peryton/基于关联矩阵的微生物功能/GIP_Sim.csv')
# micro_sem = pd.read_csv('./dataset/peryton/基于疾病语义的微生物功能/functional_similarity2_matrix.csv')
# micro_fun1 = pd.read_csv('./dataset/peryton/微生物-功能/complete_microbe_associations_ds2_matrix.csv')
# micro_fun2 = pd.read_csv('./dataset/peryton/微生物-功能/complete_microbe_similarities_ds2_matrix.csv')
# A = A.iloc[:, 1:]

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

# deep_A = calculate_metapath_optimized(mm, dd, dm, n)
samples = get_all_the_samples_old(A.T.values)
# samples = get_all_the_samples_old_22(A.T.values)


# samples = get_all_pairs_random_2(A.T.values)
# samples = get_all_pairs_random(A.values, deep_A)
samples = np.array(samples)

# # 创建线性编码器
# encoder1 = LinearEncoder(input_dim=292, output_dim=128).to(device)
# encoder2 = LinearEncoder(input_dim=39, output_dim=32).to(device)

micro_low_features = A.T  # 微生物的低级特征是矩阵A的列向量
disease_low_features = A  # 疾病的低级特征是矩阵A的行向量
# complete_micro_features = torch.cat((micro_low_features, micro_out), dim=1)
# complete_disease_features = torch.cat((disease_low_features, disease_out), dim=1)
micro_low_features_tensor = torch.Tensor(micro_low_features.values).to(device)
disease_low_features_tensor = torch.Tensor(disease_low_features.values).to(device)

# 将微生物关联矩阵转换为多维图数据
micro_data, micro1, micro2, micro3, micro4, micro5, micro_edge_indices, micor_edge_weights = create_multi_dimensional_data(microbiome_matrices, device,micro_low_features_tensor)
# 将疾病关联矩阵转换为多维图数据
disease_data, dis1, dis2, dis3, dis4, dis5, dis_edge_indices, dis_edge_weights = create_multi_dimensional_data(disease_matrices, device,disease_low_features_tensor)

# HMDAD
microbiome_gcn = MultiDimensionalGCN(in_channels=331, hidden_channels=128, out_channels=64, num_dimensions=5).to(device)
disease_gcn = MultiDimensionalGCN(in_channels=331, hidden_channels=128, out_channels=64, num_dimensions=5).to(device)

#disbiome
# microbiome_gcn = MultiDimensionalGCN(in_channels=1939, hidden_channels=256, out_channels=64, num_dimensions=5).to(device)
# disease_gcn = MultiDimensionalGCN(in_channels=1939, hidden_channels=256, out_channels=64, num_dimensions=5).to(device)

# peryton
# microbiome_gcn = MultiDimensionalGCN(in_channels=1439, hidden_channels=256, out_channels=64, num_dimensions=5).to(device)
# disease_gcn = MultiDimensionalGCN(in_channels=1439, hidden_channels=256, out_channels=64, num_dimensions=5).to(device)

len_micro_out = 64
len_disease_out = 64

# 定义整体模型，并将模型移动到GPU上
# vae_model = VAEModel(micro_input_dim=71, micro_hidden_dim=64, micro_latent_dim=32,
#                      disease_input_dim=308, disease_hidden_dim=128, disease_latent_dim=32).to(device)
# vae_model = MicroDiseaseModel(micro_input_dim=64, micro_hidden_dim=32, micro_latent_dim=16,
#                      disease_input_dim=64, disease_hidden_dim=32, disease_latent_dim=16).to(device)

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
        # optimizer3.zero_grad()

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
        # constrate_loss = constrate_loss_calculate(i_select,i_hat_select,j_select,j_hat_select)      #待修改
        constrate_loss = 0
        # constrate_loss = 0
        # print('constrate_loss=',constrate_loss)
        loss = lambda_mse * criterion(train_label,train_labels).to(torch.float32)  + loss_l2 + lambda_constrate * constrate_loss

        #loss.backward()
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


        unique_test_list_tensor = test_list_tensor
        test_labels = unique_test_list_tensor[:, 2].cpu().numpy()  # Move to CPU if necessary
        indices = unique_test_list_tensor[:, :2].long()  # Ensure indices are long

        perdcit_score = prob_matrix[indices[:, 0], indices[:, 1]].cpu().numpy()  # Move to CPU if necessary
        perdcit_label = [1 if prob >= 0.2 else 0 for prob in perdcit_score]

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

        precision, recall, _ = precision_recall_curve(test_labels, perdcit_score)
        pr_auc = auc(recall, precision)
        # 在交叉验证的最后保存模型
        # torch.save(microbiome_gcn.state_dict(), f'microbiome_gcn_fold_{cl}.pth')
        # torch.save(disease_gcn.state_dict(), f'disease_gcn_fold_{cl}.pth')
        # Free tensors
        torch.cuda.empty_cache()

        with open(fold_metrics_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([cl + 1, roc_auc, pr_auc, f1_score(test_labels, perdcit_label),
                             accuracy_score(test_labels, perdcit_label)])


prob_matrix_avg = prob_matrix_avg / k_split
np.savetxt('./result/disbiome_prob_matrix_avg.csv', prob_matrix_avg, delimiter='\t',fmt='%0.5f')  # HMDAD peryton Disbiome
print('############ avg score #############')
for metric, values in metrics_summary.items():
    print(f"{metric}: {np.mean(values):.3f} ± {np.std(values):.3f}")


print('mean AUC = ',np.mean(sk_aucs))
print('mean AUPR = ',np.mean(sk_average_precisions))
folds = test_label_score  # 将测试标签和预测概率的数据字典赋值给folds变量，以便于后续使用。


mean_tpr = np.mean(tprs, axis=0)  # 计算所有折次的真正率(TPR)的平均值。
mean_tpr[-1] = 1.0  # 确保曲线的结束点位于(1,1)
mean_auc = metrics.auc(mean_fpr, mean_tpr)  # 计算平均AUC值。
std_auc = np.std(aucs)  # 计算AUC值的标准差。
fig, ax = plt.subplots(1, 1, figsize=(5, 5))

# 计算TPR的标准差，并绘制TPR的置信区间。
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)  # 计算上界。
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)  # 计算下界。

# 左侧绘制ROC曲线
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
ax.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})', lw=2, alpha=.8)
ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.3, label=r'± 1 std. dev.')
ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title="Receiver Operating Characteristic", xlabel='False Positive Rate',
       ylabel='True Positive Rate')
ax.legend(loc="lower right")

fig2, axs2 = plt.subplots(1, 1, figsize=(5, 5))
for i in range(5):
    plt.plot(sk_fpr[i], sk_tprs[i], label=f'Fold {i + 1} AUC = {sk_aucs[i]:.2f}')
axs2.plot([0, 1], [0, 1], 'k--', label='Random',color='r')
axs2.set_xlim([-0.05, 1.05])
axs2.set_ylim([0.0, 1.05])
axs2.set_xlabel('False Positive Rate')
axs2.set_ylabel('True Positive Rate')
axs2.set_title('ROC Curves ')
axs2.legend(loc="lower right")
plt.show()


# 绘制Precision-Recall曲线
fig3, axs3 = plt.subplots(1, 1, figsize=(5, 5))
for i in range(5):
    axs3.plot(sk_recalls[i], sk_precisions[i], label=f'Fold {i + 1} AUPR = {sk_average_precisions[i]:.2f}')
axs3.plot([0, 1], [1, 0], 'k--', label='Random',color='r')
axs3.set_xlim([-0.05, 1.05])
axs3.set_ylim([0.0, 1.05])
axs3.set_xlabel('Recall')
axs3.set_ylabel('Precision')
axs3.set_title('Precision-Recall Curvesin ')
axs3.legend(loc="lower left")
plt.show()



mean_fpr = np.linspace(0, 1, 100)
mean_recall = np.linspace(0, 1, 100)

# 用于存储插值后的数据
tprs = []
precisions = []

# 插值TPR数据
for fpr_temp, tpr_temp in zip(sk_fpr, sk_tprs):
    interp_tpr = np.interp(mean_fpr, fpr_temp, tpr_temp)
    interp_tpr[0] = 0.0  # 确保曲线从0开始
    tprs.append(interp_tpr)
mean_tprs = np.mean(tprs, axis=0)
mean_tprs[-1] = 1.0  # 确保曲线以1结束
mean_auc = np.mean(sk_aucs)

# 插值Precision数据
for recall_temp, precision_temp in zip(sk_recalls, sk_precisions):
    interp_precision = np.interp(mean_recall, recall_temp[::-1], precision_temp[::-1])
    precisions.append(interp_precision)
mean_precisions = np.mean(precisions, axis=0)
mean_average_precision = np.mean(sk_average_precisions)

# 保存所有绘图所需数据
np.savez('complete_average_plot_data.npz', mean_fpr=mean_fpr, mean_tprs=mean_tprs, mean_auc=mean_auc,
         mean_recall=mean_recall, mean_precisions=mean_precisions, mean_average_precision=mean_average_precision)

# 加载数据
data = np.load('complete_average_plot_data.npz')
mean_fpr = data['mean_fpr']
mean_tprs = data['mean_tprs']
mean_auc = data['mean_auc']
mean_recall = data['mean_recall']
mean_precisions = data['mean_precisions']
mean_average_precision = data['mean_average_precision']

# 绘制平均ROC曲线
fig4, axs4 = plt.subplots(1, 1, figsize=(5, 5))
axs4.plot(mean_fpr, mean_tprs, label=f'Average AUC = {mean_auc:.2f}')
axs4.plot([0, 1], [0, 1], 'k--', label='Random', color='r')
axs4.set_xlim([-0.05, 1.05])
axs4.set_ylim([0.0, 1.05])
axs4.set_xlabel('False Positive Rate')
axs4.set_ylabel('True Positive Rate')
axs4.set_title('Average ROC Curve')
axs4.legend(loc="lower right")
plt.show()

# 绘制平均Precision-Recall曲线
fig5, axs5 = plt.subplots(1, 1, figsize=(5, 5))
axs5.plot(mean_recall, mean_precisions, label=f'Average AUPR = {mean_average_precision:.2f}')
axs5.plot([0, 1], [1, 0], 'k--', label='Random', color='r')
axs5.set_xlim([-0.05, 1.05])
axs5.set_ylim([0.0, 1.05])
axs5.set_xlabel('Recall')
axs5.set_ylabel('Precision')
axs5.set_title('Average Precision-Recall Curve')
axs5.legend(loc="lower left")
plt.show()
