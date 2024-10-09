import numpy as np
import torch
import scipy.sparse as sp
import pandas as pd
import math
import random
from sklearn.preprocessing import minmax_scale, scale
import matplotlib.pyplot as plt
from sklearn import metrics
from itertools import cycle
import heapq
from torch.nn.functional import cosine_similarity
import torch.nn.functional as F

def set_seed(seed=123):
    random.seed(seed)  # Python内置随机库
    np.random.seed(seed)  # NumPy随机库
    torch.manual_seed(seed)  # PyTorch随机库
    if torch.cuda.is_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)  # PyTorch CUDA随机库
        torch.cuda.manual_seed_all(seed)  # 如果使用多个GPU
        torch.backends.cudnn.deterministic = True  # 确定性算法，可能会影响性能
        torch.backends.cudnn.benchmark = False

def calculate_combined_similarity(disease_matrices, microbe_matrices):
    """
    计算疾病和微生物的综合相似性矩阵。

    参数：
    - disease_matrices：包含5个疾病相似性矩阵的列表。
    - microbe_matrices：包含5个微生物相似性矩阵的列表。

    返回值：
    - sim_disease：疾病的综合相似性矩阵。
    - sim_microbe：微生物的综合相似性矩阵。
    """

    # 计算疾病的综合相似性矩阵
    sum_disease_similarity = np.zeros_like(disease_matrices[0])
    count_disease_nonzero = np.zeros_like(disease_matrices[0])

    for sim_matrix in disease_matrices:
        sum_disease_similarity += sim_matrix
        count_disease_nonzero += (sim_matrix != 0).astype(int)

    count_disease_nonzero[count_disease_nonzero == 0] = 1
    sim_disease = sum_disease_similarity / count_disease_nonzero

    # 计算微生物的综合相似性矩阵
    sum_microbe_similarity = np.zeros_like(microbe_matrices[0])
    count_microbe_nonzero = np.zeros_like(microbe_matrices[0])

    for sim_matrix in microbe_matrices:
        sum_microbe_similarity += sim_matrix
        count_microbe_nonzero += (sim_matrix != 0).astype(int)

    count_microbe_nonzero[count_microbe_nonzero == 0] = 1
    sim_microbe = sum_microbe_similarity / count_microbe_nonzero

    return sim_disease, sim_microbe

def matrixPow(Matrix, n):                           # 计算矩阵的n次幂
    """
    计算矩阵的n次幂。

    参数:
    - Matrix: 输入矩阵
    - n: 幂次

    返回:
    - 矩阵的n次幂
    """
    if(type(Matrix) == list):
        Matrix = np.array(Matrix)
    if(n == 1):
        return Matrix
    else:
        return np.matmul(Matrix, matrixPow(Matrix, n - 1))

def calculate_metapath_optimized(mm, dd, dm, n):        #计算n层元路径
    """
    优化版：计算微生物-疾病第n层元路径。

    参数:
    - mm: 微生物相似度矩阵
    - dd: 疾病相似度矩阵
    - md: 微生物疾病关联矩阵
    - n: 元路径的层数

    返回:
    - n层元路径矩阵
    """
    # 基本情况，如果n为1，直接计算并返回第一层元路径矩阵
    md = dm.T
    DD = dm @ mm @ md @ dd
    DM = dm @ mm
    # MM = md @ dd @ dm @ mm
    # MD = md @ dd
    if n == 1:
    #    return mm @ md @ dd
        return dd @ dm @ mm
    else:
        #k = n / 2
        k = n
        k = int(k)
        DK = matrixPow(DD, k)
        deep_A = dd @ DK @ DM
        #MK = matrixPow(MM, k)
        #deep_A = mm @ MK @ MD

        #if n % 2 ==0:
        #    deep_A = mm @ MK @ MD
        #else:
        #    deep_A = mm @ MK

    return deep_A



def get_all_the_samples_old(A):
    m, n = A.shape
    pos = []
    neg = []
    for i in range(m):
        for j in range(n):
            if A[i, j] == 1:
                pos.append([i, j, 1])
            else:
                neg.append([i, j, 0 ])
    n = len(pos)

    neg_new = random.sample(neg, n)                                     # 从neg中随机抽取n个样本，使得正负样本数量相等，存入neg_new中
    tep_samples = pos + neg_new
    samples = random.sample(tep_samples, len(tep_samples))              # 从tep_samples中随机抽取所有样本，打乱顺序
    samples = random.sample(samples, len(samples))
    samples = np.array(samples)
    return samples



def get_all_the_samples_old_22(A_in):
    A = A_in.copy()
    m, n = A.shape
    pairs = []
    for i in range(m):
        for j in range(n):
            if A[i, j] == 1:
                j_hat = np.argmin(np.where(A[i] == 0, A[i], np.inf))
                # 将找到的位置在A中置为-1
                if j_hat < n:  # 确保找到的索引在范围内
                    A[i, j_hat] = -1

                pairs.append([i, j, 1])
                pairs.append([i, j_hat, 0])
    return pairs


def get_all_pairs(A_in, deep_A):
    A = A_in.copy()
    m, n = A.shape
    pairs = []
    for i in range(m):
        for j in range(n):
            if A[i, j] == 1:
                j_hat = np.argmin(np.where(A[i] == 0, deep_A[i], np.inf))
                # 将找到的位置在A中置为-1
                if j_hat < n:  # 确保找到的索引在范围内
                    A[i, j_hat] = -1

                # 在deep_A的第j_hat列中找到最小值且A中对应位置为0的元素的行索引i_hat
                #i_hat = np.argmin(np.where(A[:, j_hat] == 0, deep_A[:, j_hat], np.inf))
                i_hat = np.argmin(np.where(A[:, j] == 0, deep_A[:, j], np.inf))
                # 将找到的位置在A中置为-1
                if i_hat < m:  # 确保找到的索引在范围内
                    #A[i_hat, j] = -1
                    pass

                #pairs.append([i, j, i_hat, j_hat])
                pairs.append([j, i, j_hat, i_hat])
    return pairs

def get_all_pairs_random(A_in, deep_A):
    A = A_in.copy()
    m, n = A.shape
    pairs = []

    for i in range(m):
        for j in range(n):
            if A[i, j] == 1:
                # 75th percentile for deep_A[i] where A[i] is 0
                valid_indices_i = np.where(A[i] == 0)[0]
                if len(valid_indices_i) > 0:
                    q75 = np.percentile(deep_A[i, valid_indices_i], 75)
                else:
                    continue  # Skip if no valid zero elements are found

                # Generate j_hat and ensure it is valid
                j_hat = -1
                while j_hat < 0 or j_hat >= n or A[i, j_hat] != 0:
                    j_hat = int(np.clip(np.random.normal(loc=q75, scale=1), 0, n - 1))

                # Mark this element to prevent reuse
                A[i, j_hat] = -1

                # 75th percentile for deep_A[:, j] where A[:, j] is 0
                valid_indices_j = np.where(A[:, j] == 0)[0]
                if len(valid_indices_j) > 0:
                    q75_col = np.percentile(deep_A[valid_indices_j, j], 75)
                else:
                    continue  # Skip if no valid zero elements are found

                # Generate i_hat and ensure it is valid
                i_hat = -1
                while i_hat < 0 or i_hat >= m or A[i_hat, j] != 0:
                    i_hat = int(np.clip(np.random.normal(loc=q75_col, scale=1), 0, m - 1))

                # Ensure the index is within range and corresponds to an element in A that is 0
                if i_hat >= m or A[i_hat, j] != 0:
                    continue

                pairs.append([j, i, j_hat, i_hat])

    return pairs


def get_all_pairs_random(A_in):
    A = A_in.copy()
    m, n = A.shape
    pairs = []
    for i in range(m):
        for j in range(n):
            if A[i, j] == 1:
                # 寻找第i行中A为0的所有列索引
                zero_cols_in_row = np.where(A[i] == 0)[0]
                if zero_cols_in_row.size > 0:  # 如果存在至少一个0
                    j_hat = np.random.choice(zero_cols_in_row)  # 随机选择一个索引
                    #A[i, j_hat] = -1  # 将找到的位置在A中置为-1

                # 寻找第j列中A为0的所有行索引
                zero_rows_in_col = np.where(A[:, j] == 0)[0]
                if zero_rows_in_col.size > 0:  # 如果存在至少一个0
                    i_hat = np.random.choice(zero_rows_in_col)  # 随机选择一个索引
                    #A[i_hat, j] = -1  # 将找到的位置在A中置为-1

                pairs.append(
                    [i, j, i_hat if zero_rows_in_col.size > 0 else None, j_hat if zero_cols_in_row.size > 0 else None])

    return pairs

def get_all_pairs_random_2(A_in):
    A = A_in.copy()
    m, n = A.shape
    pairs = []
    for i in range(m):
        for j in range(n):
            if A[i, j] == 1:
                # 寻找第i行中A为0的所有列索引
                zero_cols_in_row = np.where(A[i] == 0)[0]
                if zero_cols_in_row.size > 0:  # 如果存在至少一个0
                    j_hat = np.random.choice(zero_cols_in_row)  # 随机选择一个索引


                pairs.append([i, j, 1])
                pairs.append([i, j_hat, 0])


    return pairs


# 定义函数获取非零数量
def non_zero_count(*matrices):
    non_zero_matrices = []
    for matrix in matrices:
        features = np.stack(matrix, axis=0)  # 将矩阵堆叠在一起
        features_sum = np.sum(features, axis=-1)  # 对最后一个维度的特征进行求和
        non_zero_count = np.sum(features_sum != 0, axis=0)  # 统计每个节点在五个来源中非零特征的数量
        non_zero_count[non_zero_count == 0] = 1  # 将为 0 的值赋值为 1
        non_zero_matrices.append(torch.tensor(non_zero_count))  # 将 NumPy 数组转换为 PyTorch 张量
    return non_zero_matrices

def sim_cos( z1, z2):
    # z1_out = F.normalize(z1, p=2, dim=1)
    # z2_out = F.normalize(z2, p=2, dim=1)
    mmget=torch.mm(z1, z2.t())
    mmget = (mmget - mmget.min()) / (
                mmget.max() - mmget.min())
    return mmget

def constrate_loss_calculate(train_i_mic_feature_tensor,train_i_hat_mic_feature_tensor,
                            train_j_disease_feature_tensor,train_j_hat_disease_feature_tensor, tau=1):
    f = lambda x: torch.exp(x / tau)
    i_j_sim = f(sim_cos(train_i_mic_feature_tensor, train_j_disease_feature_tensor))
    i_j_hat_sim = f(sim_cos(train_i_mic_feature_tensor, train_j_hat_disease_feature_tensor))
    i_hat_j_sim = f(sim_cos(train_i_hat_mic_feature_tensor, train_j_disease_feature_tensor))
    diag_i_j_sim = i_j_sim.diag()
    diag_i_j_hat_sim = i_j_hat_sim.diag()
    diag_i_hat_j_sim = i_hat_j_sim.diag()

    # constrate_loss = -torch.log(diag_i_j_sim / (diag_i_j_sim + diag_i_j_hat_sim + diag_i_hat_j_sim)).mean()
    constrate_loss = ((diag_i_j_hat_sim + diag_i_hat_j_sim) / (diag_i_j_sim + diag_i_j_hat_sim + diag_i_hat_j_sim)).mean()
    # constrate_loss = (diag_i_j_sim / (diag_i_j_sim + diag_i_j_hat_sim + diag_i_hat_j_sim)).mean()
    return  constrate_loss


def select_rows(A, selected_rows):
    """
    从矩阵A中选择指定的行。

    参数:
    A (torch.Tensor): 原始矩阵 (m x n)。
    selected_rows (list): 需要保留的行的索引列表。

    返回:
    torch.Tensor: 仅保留指定行的新矩阵。
    """
    device = A.device  # 获取A所在的设备
    dtype = A.dtype  # 获取A的数据类型
    m, n = A.shape  # 获取原始矩阵的行数和列数
    k = len(selected_rows)  # 要保留的行数

    # 构造选择矩阵 M 并移动到与 A 相同的设备和数据类型
    M = torch.zeros(k, m, device=device, dtype=dtype)
    for idx, row in enumerate(selected_rows):
        M[idx, row] = 1

    # 通过矩阵乘法得到结果矩阵 B
    B = torch.mm(M, A)

    return B




def construct_selection_matrix(A_shape, selected_rows):
    """
    构造选择矩阵M，用于从原始矩阵中选择指定的行。

    参数:
    A_shape (tuple): 原始矩阵的形状 (m, n)。
    selected_rows (list): 需要保留的行的索引列表。

    返回:
    torch.Tensor: 选择矩阵 M (k x m)，用于选择原始矩阵的指定行。
    """
    m, n = A_shape  # 获取原始矩阵的行数和列数
    k = len(selected_rows)  # 要保留的行数

    # 构造选择矩阵 M
    M = torch.zeros(k, m)
    for idx, row in enumerate(selected_rows):
        M[idx, row] = 1

    return M

############################################        图的差异化学习                 ##################################################

def compute_similarity_matrix_gpu(X, sigma=1.0):
    """
    计算相似性矩阵，基于高斯核函数。

    参数:
    X: torch.Tensor, 输入数据 (n_samples, n_features)
    sigma: float, 高斯核的超参数

    返回:
    W: torch.Tensor, 相似性矩阵 (n_samples, n_samples)
    """
    # X = X.to(device)  # 确保数据被移动到 GPU 上
    dist_matrix = torch.cdist(X, X, p=2) ** 2  # 计算欧式距离的平方
    W = torch.exp(-dist_matrix / (2 * sigma ** 2))  # 使用高斯核函数计算相似性
    return W


def compute_eigen_decomposition_gpu(L, k):
    """
    计算拉普拉斯矩阵的前 k 个最小特征值和对应的特征向量。

    参数:
    L: torch.Tensor, 拉普拉斯矩阵
    k: int, 需要选择的特征向量数量

    返回:
    eigenvalues: torch.Tensor, 前 k 个最小特征值
    eigenvectors: torch.Tensor, 前 k 个最小特征向量
    """
    eigenvalues, eigenvectors = torch.linalg.eigh(L)
    return eigenvalues[:k], eigenvectors[:, :k]  # 返回前 k 个特征值和特征向量


# Step 4: 使用 Eigengap 启发选择类的范围（只输出每个类的起点和终点）
def compute_eigengap_gpu(eigenvalues, threshold=0.6):
    """
    根据特征值差距选择类的个数，并返回每个类的起点和终点特征值。

    参数:
    eigenvalues: torch.Tensor, 按升序排列的特征值
    threshold: float, 用于判断差距是否显著的阈值

    返回:
    classes_ranges: list of tuples, 每个元组包含当前类的起点和终点特征值
    """
    gaps = torch.diff(eigenvalues)  # 计算相邻特征值之间的差距

    # 初始化类的起点和终点
    classes_ranges = []
    start_value = eigenvalues[0].item()  # 第一个类从第一个特征值开始

    # 遍历差距，寻找显著增大的点作为类的边界
    for i, gap in enumerate(gaps):
        if gap > threshold:  # 如果差距超过阈值，则确定类的边界
            end_value = eigenvalues[i].item()  # 当前类的结束特征值
            classes_ranges.append((start_value, end_value))  # 添加类的范围
            start_value = eigenvalues[i + 1].item()  # 下一个类的开始特征值

    # 最后一个类的结束特征值是最后一个特征值
    classes_ranges.append((start_value, eigenvalues[-1].item()))

    return classes_ranges


def laplacian_eigen_decomposition_pytorch(L):
    """
    使用 PyTorch 对归一化的拉普拉斯矩阵进行特征分解，并按特征值大小排序。

    参数:
    L -- 归一化的拉普拉斯矩阵 (方阵)

    返回:
    sorted_eigenvalues_matrix -- 排序后的特征值对角矩阵
    sorted_indices -- 对应排序后的特征值原始索引的索引矩阵
    eigenvectors -- 未排序的特征向量矩阵
    """
    # 1. 计算特征值和特征向量（原始顺序）
    eigenvalues, eigenvectors = torch.linalg.eig(L)

    # 2. 由于 torch.linalg.eig 返回的是复数，需要取实部，因为拉普拉斯矩阵的特征值应该是实数
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real

    # 3. 对特征值从小到大排序，并记录排序的索引
    sorted_indices = torch.argsort(eigenvalues)

    # 4. 使用排序索引对特征值进行排序
    sorted_eigenvalues = eigenvalues[sorted_indices]

    # 5. 将排序后的特征值转换为对角矩阵
    sorted_eigenvalues_matrix = torch.diag(sorted_eigenvalues)

    # 6. 返回排序后的特征值对角矩阵、排序索引、以及未排序的特征向量矩阵
    return sorted_eigenvalues_matrix, sorted_indices, eigenvectors



def compute_eigengap_classes(eigenvalues_matrix, threshold, indices):
    """
    根据特征值差距选择类的个数，并返回每个类的特征标志矩阵。

    参数:
    eigenvalues: torch.Tensor, 按升序排列的特征值
    threshold: float, 用于判断差距是否显著的阈值
    indices: torch.Tensor, 对应排序后的特征值的原先的特征向量的索引矩阵

    返回:
    class_flags: torch.Tensor, 二维矩阵，其中每行表示一个类，如果节点属于该类，矩阵对应位置为1，否则为0
    """
    # 提取对角线上的特征值
    eigenvalues = torch.diagonal(eigenvalues_matrix)

    # 计算相邻特征值之间的差距
    gaps = torch.diff(eigenvalues)

    # 初始化类的起点和终点
    classes_ranges = []
    start_value = 0  # 第一个类从第一个特征值开始的索引

    # 遍历差距，寻找显著增大的点作为类的边界
    for i, gap in enumerate(gaps):
        if gap > threshold:  # 如果差距超过阈值，则确定类的边界
            end_value = i  # 当前类的结束索引
            classes_ranges.append((start_value, end_value))  # 添加类的索引范围
            start_value = i + 1  # 下一个类的开始索引

    # 最后一个类的结束索引是最后一个特征值的索引
    classes_ranges.append((start_value, len(eigenvalues) - 1))

    # 构建类标志矩阵，初始化为零矩阵，行数为类的个数，列数为节点数量
    num_classes = len(classes_ranges)
    num_nodes = indices.shape[0]
    class_flags = torch.zeros((num_classes, num_nodes), dtype=torch.int32)

    # 根据每个类的范围，填充类标志矩阵
    for class_idx, (start, end) in enumerate(classes_ranges):
        # 获取属于该类的索引
        class_indices = indices[start:end + 1]
        # 将这些索引对应的列设置为1
        class_flags[class_idx, class_indices] = 1

    return class_flags


def compute_class_average_features(M_class_flags, pre_mid2_outputs):
    """
    根据每个类中的特征标志矩阵，计算每个类的平均特征向量。

    参数:
    M_class_flags: torch.Tensor, 每行表示一个类，如果节点属于该类，矩阵对应位置为1，否则为0
    pre_mid2_outputs: torch.Tensor, 每行表示一个节点的特征向量

    返回:
    class_avg_features: torch.Tensor, 每行表示一个类的平均特征向量
    """
    num_classes = M_class_flags.shape[0]  # 类的数量
    feature_dim = pre_mid2_outputs.shape[1]  # 每个特征向量的维度

    # 初始化存放每个类的平均特征向量的矩阵
    class_avg_features = torch.zeros((num_classes, feature_dim), dtype=torch.float32)

    # 遍历每一个类，计算其平均特征向量
    for i in range(num_classes):
        # 获取属于该类的节点索引，即 M_class_flags 中为1的索引
        class_indices = torch.where(M_class_flags[i] == 1)[0]

        # 获取这些节点对应的特征向量
        class_features = pre_mid2_outputs[class_indices]

        # 计算类的平均特征向量
        class_avg_features[i] = class_features.mean(dim=0)

    return class_avg_features


def add_class_avg_to_node_features(class_avg_features, M_class_flags, pre_mid2_outputs, k_save):
    """
    将每个类的平均特征向量加到对应类的节点的特征向量上。

    参数:
    class_avg_features: torch.Tensor, 每行表示一个类的平均特征向量，形状为 (num_classes, feature_dim)
    M_class_flags: torch.Tensor, 每行表示一个类的节点索引标志矩阵，形状为 (num_classes, num_nodes)
    pre_mid2_outputs: torch.Tensor, 未排序的节点特征矩阵，形状为 (num_nodes, feature_dim)

    返回:
    updated_node_features: torch.Tensor, 处理后的节点特征向量矩阵
    """
    # 初始化处理后的节点特征矩阵，直接复制原始的节点特征矩阵
    # updated_node_features = pre_mid2_outputs.clone()
    updated_node_features =torch.zeros_like(pre_mid2_outputs)
    # 遍历每个类
    for i in range(M_class_flags.shape[0]):
        # 获取属于该类的节点索引
        class_indices = torch.where(M_class_flags[i] == 1)[0]

        # 获取当前类的平均特征向量
        class_avg_vector = class_avg_features[i]

        # 将平均特征向量加到属于该类的节点特征向量上
        updated_node_features[class_indices] += k_save*class_avg_vector

    return updated_node_features



 # linear2_outputs = [self.Linear2(mid2_out) for mid2_out in pre_mid2_outputs]
 #        dim2_Gouts = self.conv2(*linear2_outputs, edge_indices, edge_weights, non_zero)
 #        pre_dim2_outputs = [in2_layer + dim2_Gout for in2_layer, dim2_Gout in zip(linear2_outputs, dim2_Gouts)]

 #
 # linear2_outputs = [self.Linear2(mid2_out) for mid2_out in pre_mid2_outputs]
 #        dim2_Gouts = self.conv2(*linear2_outputs, edge_indices, edge_weights, non_zero)
 #        pre_dim2_outputs = [(in2_layer + dim2_Gout).to(torch.float32) for in2_layer, dim2_Gout in zip(linear2_outputs, dim2_Gouts)]
 #
 #        out = sum(pre_dim2_outputs)
 #        return out
