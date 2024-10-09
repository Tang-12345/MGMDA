import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

# # 保存所有绘图所需数据
# np.savez('complete_average_plot_data.npz', mean_fpr=mean_fpr, mean_tprs=mean_tprs, mean_auc=mean_auc,
#          mean_recall=mean_recall, mean_precisions=mean_precisions, mean_average_precision=mean_average_precision)

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
