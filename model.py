import torch
import math
import numpy as np
import torch.nn as nn
import scipy.sparse as sp
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, MessagePassing
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.utils import from_scipy_sparse_matrix

from utils import  compute_similarity_matrix_gpu,laplacian_eigen_decomposition_pytorch,compute_eigengap_classes,compute_class_average_features, add_class_avg_to_node_features
# 检查CUDA是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, hidden_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
            nn.Sigmoid()  # Assuming input data is normalized between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        return self.encoder(x)

# 线性编码器，将矩阵的行编码成节点特征
class LinearEncoder(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearEncoder, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

# 定义一个多维图数据结构，继承自PyTorch Geometric的Data类
class MultiDimensionalData(Data):
    def __init__(self, num_dimensions, x=None, *args, **kwargs):
        super(MultiDimensionalData, self).__init__(*args, **kwargs)
        self.num_dimensions = num_dimensions
        self.x = x
        self.edge_indices = [None] * num_dimensions
        self.edge_weights = [None] * num_dimensions

    def set_dimension_data(self, dim, edge_index, edge_weight):
        self.edge_indices[dim] = edge_index
        self.edge_weights[dim] = edge_weight

# 将矩阵转换为PyTorch Geometric的数据格式 矩阵数据转化为图数据
def create_multi_dimensional_data(matrices, device, cat_matrix):
    num_nodes = matrices[0].shape[0]  # 节点数量，即矩阵的行数或列数
    num_dimensions = len(matrices)  # 维度数量，即矩阵的个数


    # 初始化多维图数据结构
    data = MultiDimensionalData(num_dimensions)

    edge_indices = []
    edge_weights = []
    node_features_list = []

    for i, matrix in enumerate(matrices):
        matrix_sparse = sp.coo_matrix(matrix)  # 将矩阵转换为稀疏矩阵格式
        edge_index, edge_weight = from_scipy_sparse_matrix(matrix_sparse)  # 从稀疏矩阵中提取边索引和边权重
        edge_index = edge_index.to(device)  # 将边索引移动到GPU
        edge_weight = edge_weight.to(device)  # 将边权重移动到GPU
        edge_indices.append(edge_index)
        edge_weights.append(edge_weight)
        data.set_dimension_data(i, edge_index, edge_weight)  # 设置每个维度的数据

        # 将原始矩阵数据转换为节点特征，将每个维度的节点特征与cat_matrix进行拼接
        node_features = torch.cat((torch.Tensor(matrix).to(device), cat_matrix), dim=1)
        node_features_list.append(node_features)

    return data, *node_features_list, edge_indices, edge_weights

class MultiDimensionalGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_dimensions,layers_pos):
        super(MultiDimensionalGCNConv, self).__init__(aggr='add')
        self.convs = torch.nn.ModuleList([
            GCNConv(in_channels, out_channels) for _ in range(num_dimensions)
        ])
        self.num_dimensions = num_dimensions
        self.out_channels = out_channels
        self.save_weight = math.exp(-layers_pos)
        # 初始化每个维度的可训练权重
        # self.dimension_weights = nn.Parameter(torch.ones(num_dimensions))


    def forward(self, dim1, dim2, dim3, dim4, dim5, edge_indices, edge_weights, non_zero,dimension_weights):
        dim1_out = self.convs[0](dim1, edge_indices[0], edge_weight = edge_weights[0])              #错误发生在当前或之前
        dim2_out = self.convs[1](dim2, edge_indices[1], edge_weight=edge_weights[1])
        dim3_out = self.convs[2](dim3, edge_indices[2], edge_weight=edge_weights[2])
        dim4_out = self.convs[3](dim4, edge_indices[3], edge_weight=edge_weights[3])
        dim5_out = self.convs[4](dim5, edge_indices[4], edge_weight=edge_weights[4])

        # norm_weights = self.dimension_weights
        norm_weights = dimension_weights
        # 使用可训练的权重计算融合后的输出
        fused_output = (
                norm_weights[0] * dim1_out +
                norm_weights[1] * dim2_out +
                norm_weights[2] * dim3_out +
                norm_weights[3] * dim4_out +
                norm_weights[4] * dim5_out
        )

        # fused_output = dim1_out + dim2_out + dim3_out + dim4_out + dim5_out
        non_zero_expand = non_zero.unsqueeze(-1).expand(-1, self.out_channels)
        # dim1_out2 = (1-self.save_weight) * dim1 + self.save_weight *fused_output / non_zero_expand
        # dim2_out2  = (1-self.save_weight) * dim2 + self.save_weight * fused_output / non_zero_expand
        # dim3_out2  = (1-self.save_weight) * dim3 + self.save_weight * fused_output / non_zero_expand
        # dim4_out2  = (1-self.save_weight) * dim4 + self.save_weight * fused_output / non_zero_expand
        # dim5_out2  = (1-self.save_weight) * dim5 + self.save_weight * fused_output / non_zero_expand
        dim1_out2 =  dim1 + self.save_weight * fused_output / non_zero_expand
        dim2_out2 =  dim2 + self.save_weight * fused_output / non_zero_expand
        dim3_out2 =  dim3 + self.save_weight * fused_output / non_zero_expand
        dim4_out2 =  dim4 + self.save_weight * fused_output / non_zero_expand
        dim5_out2 =  dim5 + self.save_weight * fused_output / non_zero_expand
        # print("norm_weights 1~5 : ",norm_weights[0],norm_weights[1],norm_weights[2],norm_weights[3],norm_weights[4])


        return dim1_out2 ,dim2_out2 ,dim3_out2 ,dim4_out2 ,dim5_out2  #


class MultiDimensionalNormalGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_dimensions):
        super(MultiDimensionalNormalGCNConv, self).__init__(aggr='add')
        self.convs = torch.nn.ModuleList([
            GCNConv(in_channels, out_channels) for _ in range(num_dimensions)
        ])
        self.num_dimensions = num_dimensions
        self.out_channels = out_channels



    def forward(self, dim1, dim2, dim3, dim4, dim5, edge_indices, edge_weights,):

        dim1_out = self.convs[0](dim1.to(torch.float32), edge_indices[0], edge_weight = edge_weights[0])              #错误发生在当前或之前
        dim2_out = self.convs[1](dim2.to(torch.float32), edge_indices[1], edge_weight=edge_weights[1])
        dim3_out = self.convs[2](dim3.to(torch.float32), edge_indices[2], edge_weight=edge_weights[2])
        dim4_out = self.convs[3](dim4.to(torch.float32), edge_indices[3], edge_weight=edge_weights[3])
        dim5_out = self.convs[4](dim5.to(torch.float32), edge_indices[4], edge_weight=edge_weights[4])

        return dim1_out ,dim2_out ,dim3_out ,dim4_out ,dim5_out  # 返回更新后的特征矩阵以及综合特征矩阵



# 定义多维图卷积网络
#3多+3区分
class MultiDimensionalGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_dimensions):
        super(MultiDimensionalGCN, self).__init__()

        self.pre_conv1 = MultiDimensionalGCNConv(hidden_channels, hidden_channels, num_dimensions, 0)
        self.pre_conv2 = MultiDimensionalGCNConv(out_channels, out_channels, num_dimensions, 1)
        self.pre_conv3 = MultiDimensionalGCNConv(out_channels, out_channels, num_dimensions, 2)

        self.beh_conv1 = MultiDimensionalNormalGCNConv(out_channels, out_channels, num_dimensions)
        self.beh_conv2 = MultiDimensionalNormalGCNConv(out_channels, out_channels, num_dimensions)
        self.beh_conv3 = MultiDimensionalNormalGCNConv(out_channels, out_channels, num_dimensions)

        self.pre_Lin1 = torch.nn.Linear(in_channels, hidden_channels)
        self.pre_Lin2 = torch.nn.Linear(hidden_channels, out_channels)
        self.pre_Lin3 = torch.nn.Linear(out_channels, out_channels)


        self.beh_Lin1 = torch.nn.Linear(out_channels, out_channels)
        self.beh_Lin2 = torch.nn.Linear(out_channels, out_channels)
        self.beh_Lin3 = torch.nn.Linear(out_channels, out_channels)

        self.dimension_weights = nn.Parameter(torch.ones(num_dimensions), requires_grad=True)
        self.k_save = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.threshold=0.5

    def forward(self, dim1, dim2, dim3, dim4, dim5, edge_indices, edge_weights, non_zero):
        input_dimension_weights = self.dimension_weights/self.dimension_weights.sum()
        sart_inputs = [dim1, dim2, dim3, dim4, dim5]

        ###########################   第 1   层    #####################################

        L1_by_Linear = [self.pre_Lin1(dim) for dim in sart_inputs]
        L1_by_MGCN = self.pre_conv1(*L1_by_Linear, edge_indices, edge_weights, non_zero, input_dimension_weights)
        L1_out = [F.relu(mid_in + mid_Gout).to(torch.float32) for mid_in, mid_Gout in zip(L1_by_Linear, L1_by_MGCN)]

        ###########################   第 2   层    #####################################

        L2_by_Linear = [self.pre_Lin2(mid_out) for mid_out in L1_out]
        L2_by_MGCN = self.pre_conv2(*L2_by_Linear, edge_indices, edge_weights, non_zero, input_dimension_weights)
        L2_out = [F.relu(in_layer + dim_Gout).to(torch.float32) for in_layer, dim_Gout in
                       zip(L2_by_Linear, L2_by_MGCN)]

        ###########################   第 3   层    #####################################

        L3_by_Linear = [self.pre_Lin3(dim_out) for dim_out in L2_out]
        L3_by_MGCN = self.pre_conv3(*L3_by_Linear, edge_indices, edge_weights, non_zero, input_dimension_weights)
        L3_out = [F.relu(mid2_in + mid2_Gout).to(torch.float32) for mid2_in, mid2_Gout in
                        zip(L3_by_Linear, L3_by_MGCN)]

        ###########################   第 4   层    #####################################

        L4_by_GCN = self.beh_conv1(*L3_out, edge_indices, edge_weights)
        L4_G_sim_matrix = [compute_similarity_matrix_gpu(mid2_out) for mid2_out in L4_by_GCN]
        # 调用函数，并分别提取函数返回的三个值
        L4_results = [laplacian_eigen_decomposition_pytorch(G_sim) for G_sim in L4_G_sim_matrix]
        # 分别提取 sorted_eigenvalues_matrix, sorted_indices 和 eigenvectors    排序后的特征值对角矩阵    对应排序后的特征值原始索引的索引矩阵    未排序的特征向量矩阵
        L4_M_sorted_eigenvalues_matrix = [res[0] for res in L4_results]
        L4_M_sorted_indices = [res[1] for res in L4_results]                                               #控制分类的阈值

        L4_M_class_flags = [compute_eigengap_classes(sort_f_value, self.threshold,  index_of_vlue) for sort_f_value, index_of_vlue in zip(L4_M_sorted_eigenvalues_matrix, L4_M_sorted_indices)]
        #求取类的平均的特征向量    待优化
        L4_M_class_avger_feature = [compute_class_average_features(class_flag, G_feature) for class_flag, G_feature in zip(L4_M_class_flags, L4_by_GCN)]
        L4_Lin_outputs = [self.beh_Lin1(mid2_out.to(device)) for mid2_out in L4_M_class_avger_feature]
        L4_Fea_outputs = [add_class_avg_to_node_features(pro_feature, pro_index, fir_feature, self.k_save )
                      for pro_feature, pro_index,fir_feature in zip(L4_Lin_outputs, L4_M_class_flags,L3_out)]
        L4_Out = [F.relu(mid2_in + mid2_Gout).to(torch.float32) for mid2_in, mid2_Gout in
                            zip(L3_out, L4_Fea_outputs)]


        ###########################   第 5   层    #####################################
        L5_by_GCN = self.beh_conv1(*L4_by_GCN, edge_indices, edge_weights)
        L5_G_sim_matrix1 = [compute_similarity_matrix_gpu(mid2_out) for mid2_out in L5_by_GCN]
        # 调用函数，并分别提取函数返回的三个值
        L5_results = [laplacian_eigen_decomposition_pytorch(G_sim) for G_sim in L5_G_sim_matrix1]
        L5_M_sorted_eigenvalues_matrix = [res[0] for res in L5_results]
        L5_M_sorted_indices = [res[1] for res in L5_results]                                               #控制分类的阈值

        L5_M_class_flags = [compute_eigengap_classes(sort_f_value, self.threshold,  index_of_vlue)
                            for sort_f_value, index_of_vlue in zip(L5_M_sorted_eigenvalues_matrix, L5_M_sorted_indices)]
        #求取类的平均的特征向量    待优化
        L5_M_class_avger_feature1 = [compute_class_average_features(class_flag, G_feature) for class_flag, G_feature in zip(L5_M_class_flags, L5_by_GCN)]
        L5_Lin_outputs= [self.beh_Lin2(mid2_out.to(device)) for mid2_out in L5_M_class_avger_feature1]
        L5_Fea_outputs = [add_class_avg_to_node_features(pro_feature, pro_index, fir_feature, self.k_save )
                       for pro_feature, pro_index,fir_feature in zip(L5_Lin_outputs, L5_M_class_flags,L4_Out)]
        L5_Out = [F.relu(mid2_in + mid2_Gout).to(torch.float32) for mid2_in, mid2_Gout in
                      zip(L4_Out, L5_Fea_outputs)]

        ###########################   第 6   层    #####################################
        L6_by_GCN = self.beh_conv1(*L5_by_GCN, edge_indices, edge_weights)
        L6_G_sim_matrix2 = [compute_similarity_matrix_gpu(mid2_out) for mid2_out in L6_by_GCN]
        # 调用函数，并分别提取函数返回的三个值
        L6_results = [laplacian_eigen_decomposition_pytorch(G_sim) for G_sim in L6_G_sim_matrix2]
        L6_M_sorted_eigenvalues_matrix = [res[0] for res in L6_results]
        L6_M_sorted_indices = [res[1] for res in L6_results]  # 控制分类的阈值

        L6_M_class_flags = [compute_eigengap_classes(sort_f_value, self.threshold, index_of_vlue) for
                         sort_f_value, index_of_vlue in zip(L6_M_sorted_eigenvalues_matrix, L6_M_sorted_indices)]
        # 求取类的平均的特征向量    待优化
        L6_M_class_avger_feature2 = [compute_class_average_features(class_flag, G_feature) for class_flag, G_feature in
                                  zip(L6_M_class_flags, L6_by_GCN)]
        L6_Lin_outputs = [self.beh_Lin3(mid2_out.to(device)) for mid2_out in L6_M_class_avger_feature2]
        L6_Fea_outputs = [add_class_avg_to_node_features(pro_feature, pro_index, fir_feature, self.k_save) for
                       pro_feature, pro_index, fir_feature in zip(L6_Lin_outputs, L6_M_class_flags, L5_Out)]
        L6_Out = [F.relu(mid2_in + mid2_Gout).to(torch.float32) for mid2_in, mid2_Gout in
                       zip(L5_Out, L6_Fea_outputs)]

        epsilon = 1e-8
        out = sum(L6_Out)+epsilon
        return out




#6多

# 定义多维图卷积网络
#6多
class MultiDimensionalGCN_6(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_dimensions):
        super(MultiDimensionalGCN_6, self).__init__()


        self.L1_conv = MultiDimensionalGCNConv(hidden_channels, hidden_channels, num_dimensions, 0)
        self.L2_conv = MultiDimensionalGCNConv(out_channels, out_channels, num_dimensions, 1)
        self.L3_conv = MultiDimensionalGCNConv(out_channels, out_channels, num_dimensions, 2)

        self.L4_conv = MultiDimensionalGCNConv(out_channels, out_channels, num_dimensions,3)
        self.L5_conv = MultiDimensionalGCNConv(out_channels, out_channels, num_dimensions,4)
        self.L6_conv = MultiDimensionalGCNConv(out_channels, out_channels, num_dimensions,5)

        self.L1_Linear = torch.nn.Linear(in_channels, hidden_channels)
        self.L2_Linear = torch.nn.Linear(hidden_channels, out_channels)
        self.L3_Linear = torch.nn.Linear(out_channels, out_channels)

        self.L4_Linear = torch.nn.Linear(out_channels, out_channels)
        self.L5_Linear = torch.nn.Linear(out_channels, out_channels)
        self.L6_Linear = torch.nn.Linear(out_channels, out_channels)

        self.dimension_weights = nn.Parameter(torch.ones(num_dimensions), requires_grad=True)

        self.k_save = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.threshold = 0.5

    def forward(self, dim1, dim2, dim3, dim4, dim5, edge_indices, edge_weights, non_zero):
        input_dimension_weights = self.dimension_weights / self.dimension_weights.sum()
        start_input = [dim1, dim2, dim3, dim4, dim5]

        ###########################   第 1   层    #####################################

        L1_inputs = [self.L1_Linear(dim) for dim in start_input]
        L1_GCN = self.L1_conv(*L1_inputs, edge_indices, edge_weights, non_zero, input_dimension_weights)
        L1_out = [F.relu(mid_in + mid_Gout).to(torch.float32) for mid_in, mid_Gout in zip(L1_inputs, L1_GCN)]

        ###########################   第 2   层    #####################################

        L2_inputs = [self.L2_Linear(mid_out) for mid_out in L1_out]
        L2_GCN = self.L2_conv(*L2_inputs, edge_indices, edge_weights, non_zero, input_dimension_weights)
        L2_out = [F.relu(in_layer + dim_Gout).to(torch.float32) for in_layer, dim_Gout in
                  zip(L2_inputs, L2_GCN)]

        ###########################   第 3   层    #####################################

        L3_inputs = [self.L3_Linear(dim_out) for dim_out in L2_out]
        L3_GCN = self.L3_conv(*L3_inputs, edge_indices, edge_weights, non_zero, input_dimension_weights)
        L3_out = [F.relu(mid2_in + mid2_Gout).to(torch.float32) for mid2_in, mid2_Gout in
                  zip(L3_inputs, L3_GCN)]
        ###########################   第 4   层    #####################################

        L4_inputs = [self.L4_Linear(dim_out) for dim_out in L3_out]
        L4_GCN = self.L4_conv(*L4_inputs, edge_indices, edge_weights,non_zero, input_dimension_weights)
        L4_out = [F.relu(mid2_in + mid2_Gout).to(torch.float32) for mid2_in, mid2_Gout in
                  zip(L4_inputs, L4_GCN)]

        ###########################   第 5  层    #####################################

        L5_inputs = [self.L5_Linear(dim_out) for dim_out in L4_out]
        L5_GCN = self.L5_conv(*L5_inputs, edge_indices, edge_weights,non_zero, input_dimension_weights)
        L5_out = [F.relu(mid2_in + mid2_Gout).to(torch.float32) for mid2_in, mid2_Gout in
                  zip(L5_inputs, L5_GCN)]

        ###########################   第 6 层    #####################################

        L6_inputs = [self.L6_Linear(dim_out) for dim_out in L5_out]
        L6_GCN = self.L6_conv(*L6_inputs, edge_indices, edge_weights,non_zero, input_dimension_weights)
        L6_out = [F.relu(mid2_in + mid2_Gout).to(torch.float32) for mid2_in, mid2_Gout in
                  zip(L6_inputs, L6_GCN)]

        epsilon = 1e-8
        out = sum(L6_out) + epsilon
        return out


class MultiDimensionalGCN_3_add_3_Nor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_dimensions):
        super(MultiDimensionalGCN_3_add_3_Nor, self).__init__()

        self.L1_conv = MultiDimensionalGCNConv(hidden_channels, hidden_channels, num_dimensions, 0)
        self.L2_conv = MultiDimensionalGCNConv(out_channels, out_channels, num_dimensions, 1)
        self.L3_conv = MultiDimensionalGCNConv(out_channels, out_channels, num_dimensions, 2)

        self.L4_conv = MultiDimensionalNormalGCNConv(out_channels, out_channels, num_dimensions)
        self.L5_conv = MultiDimensionalNormalGCNConv(out_channels, out_channels, num_dimensions)
        self.L6_conv = MultiDimensionalNormalGCNConv(out_channels, out_channels, num_dimensions)

        self.L1_Linear = torch.nn.Linear(in_channels, hidden_channels)
        self.L2_Linear = torch.nn.Linear(hidden_channels, out_channels)
        self.L3_Linear = torch.nn.Linear(out_channels, out_channels)

        self.L4_Linear = torch.nn.Linear(out_channels, out_channels)
        self.L5_Linear = torch.nn.Linear(out_channels, out_channels)
        self.L6_Linear = torch.nn.Linear(out_channels, out_channels)

        self.dimension_weights = nn.Parameter(torch.ones(num_dimensions), requires_grad=True)

        self.k_save = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.threshold=0.5

    def forward(self, dim1, dim2, dim3, dim4, dim5, edge_indices, edge_weights, non_zero):
        input_dimension_weights = self.dimension_weights/self.dimension_weights.sum()
        start_input = [dim1, dim2, dim3, dim4, dim5]

        ###########################   第 1   层    #####################################

        L1_inputs = [self.L1_Linear(dim) for dim in start_input]
        L1_GCN = self.L1_conv(*L1_inputs, edge_indices, edge_weights, non_zero, input_dimension_weights)
        L1_out = [F.relu(mid_in + mid_Gout).to(torch.float32) for mid_in, mid_Gout in zip(L1_inputs, L1_GCN)]

        ###########################   第 2   层    #####################################

        L2_inputs = [self.L2_Linear(mid_out) for mid_out in L1_out]
        L2_GCN = self.L2_conv(*L2_inputs, edge_indices, edge_weights,non_zero, input_dimension_weights)
        L2_out = [F.relu(in_layer + dim_Gout).to(torch.float32) for in_layer, dim_Gout in
                       zip(L2_inputs, L2_GCN)]

        ###########################   第 3   层    #####################################

        L3_inputs = [self.L3_Linear(dim_out) for dim_out in L2_out]
        L3_GCN = self.L3_conv(*L3_inputs, edge_indices, edge_weights, non_zero, input_dimension_weights)
        L3_out = [F.relu(mid2_in + mid2_Gout).to(torch.float32) for mid2_in, mid2_Gout in
                        zip(L3_inputs, L3_GCN)]
        ###########################   第 4   层    #####################################

        L4_inputs = [self.L4_Linear(dim_out) for dim_out in L3_out]
        L4_GCN = self.L4_conv(*L4_inputs, edge_indices, edge_weights)
        L4_out = [F.relu(mid2_in + mid2_Gout).to(torch.float32) for mid2_in, mid2_Gout in
                            zip(L4_inputs, L4_GCN)]

        ###########################   第 5  层    #####################################

        L5_inputs = [self.L5_Linear(dim_out) for dim_out in L4_out]
        L5_GCN = self.L5_conv(*L5_inputs, edge_indices, edge_weights)
        L5_out = [F.relu(mid2_in + mid2_Gout).to(torch.float32) for mid2_in, mid2_Gout in
                             zip(L5_inputs, L5_GCN)]

        ###########################   第 6 层    #####################################

        L6_inputs = [self.L6_Linear(dim_out) for dim_out in L5_out]
        L6_GCN = self.L6_conv(*L6_inputs, edge_indices, edge_weights)
        L6_out = [F.relu(mid2_in + mid2_Gout).to(torch.float32) for mid2_in, mid2_Gout in
                             zip(L6_inputs, L6_GCN)]

        epsilon = 1e-8
        out = sum(L6_out)+epsilon
        return out


class MultiDimensionalGCN_6_Nor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_dimensions):
        super(MultiDimensionalGCN_6_Nor, self).__init__()

        self.L1_conv = MultiDimensionalNormalGCNConv(hidden_channels, hidden_channels, num_dimensions)
        self.L2_conv = MultiDimensionalNormalGCNConv(out_channels, out_channels, num_dimensions)
        self.L3_conv = MultiDimensionalNormalGCNConv(out_channels, out_channels, num_dimensions)

        self.L4_conv = MultiDimensionalNormalGCNConv(out_channels, out_channels, num_dimensions)
        self.L5_conv = MultiDimensionalNormalGCNConv(out_channels, out_channels, num_dimensions)
        self.L6_conv = MultiDimensionalNormalGCNConv(out_channels, out_channels, num_dimensions)

        self.L1_Liner = torch.nn.Linear(in_channels, hidden_channels)
        self.L2_Liner = torch.nn.Linear(hidden_channels, out_channels)
        self.L3_Liner = torch.nn.Linear(out_channels, out_channels)

        self.L4_Liner = torch.nn.Linear(out_channels, out_channels)
        self.L5_Liner = torch.nn.Linear(out_channels, out_channels)
        self.L6_Liner = torch.nn.Linear(out_channels, out_channels)

        self.dimension_weights = nn.Parameter(torch.ones(num_dimensions), requires_grad=True)

        self.k_save = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.threshold=0.5

    def forward(self, dim1, dim2, dim3, dim4, dim5, edge_indices, edge_weights, non_zero):
        input_dimension_weights = self.dimension_weights/self.dimension_weights.sum()
        sart_input = [dim1, dim2, dim3, dim4, dim5]
        ###########################   第 1   层    #####################################
        L1_inputs = [self.L1_Liner(dim) for dim in sart_input]
        L1_GCN = self.L1_conv(*L1_inputs, edge_indices, edge_weights)
        L1_out = [F.relu(mid_in + mid_Gout).to(torch.float32) for mid_in, mid_Gout in zip(L1_inputs, L1_GCN)]

        ###########################   第 2   层    #####################################

        L2_inputs = [self.L2_Liner(mid_out) for mid_out in L1_out]
        L2_GCN = self.L2_conv(*L2_inputs, edge_indices, edge_weights)
        L2_out = [F.relu(in_layer + dim_Gout).to(torch.float32) for in_layer, dim_Gout in
                       zip(L2_inputs, L2_GCN)]

        ###########################   第 3   层    #####################################

        L3_inputs = [self.L3_Liner(dim_out) for dim_out in L2_out]
        L3_GCN = self.L3_conv(*L3_inputs, edge_indices, edge_weights)
        L3_out = [F.relu(mid2_in + mid2_Gout).to(torch.float32) for mid2_in, mid2_Gout in
                        zip(L3_inputs, L3_GCN)]


        ###########################   第 4   层    #####################################

        L4_inputs = [self.L4_Liner(dim_out) for dim_out in L3_out]
        L4_GCN = self.L4_conv(*L4_inputs, edge_indices, edge_weights)
        L4_out = [F.relu(mid2_in + mid2_Gout).to(torch.float32) for mid2_in, mid2_Gout in
                            zip(L4_inputs, L4_GCN)]

        ###########################   第 5  层    #####################################

        L5_inputs = [self.L5_Liner(dim_out) for dim_out in L4_out]
        L5_GCN = self.L5_conv(*L5_inputs, edge_indices, edge_weights)
        L5_out = [F.relu(mid2_in + mid2_Gout).to(torch.float32) for mid2_in, mid2_Gout in
                             zip(L5_inputs, L5_GCN)]

        ###########################   第 6 层    #####################################

        L6_inputs = [self.L6_Liner(dim_out) for dim_out in L5_out]
        L6_GCN = self.L6_conv(*L6_inputs, edge_indices, edge_weights)
        L6_out = [F.relu(mid2_in + mid2_Gout).to(torch.float32) for mid2_in, mid2_Gout in
                             zip(L6_inputs, L6_GCN)]

        epsilon = 1e-8
        out = sum(L6_out)+epsilon
        return out


class MultiDimensionalGCN_6_split(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_dimensions):
        super(MultiDimensionalGCN_6_split, self).__init__()

        self.pre_conv1 = MultiDimensionalNormalGCNConv(in_channels, hidden_channels, num_dimensions)
        self.pre_conv2 = MultiDimensionalNormalGCNConv(hidden_channels, out_channels, num_dimensions)
        self.pre_conv3 = MultiDimensionalNormalGCNConv(out_channels, out_channels, num_dimensions)


        self.beh_conv1 = MultiDimensionalNormalGCNConv(out_channels, out_channels, num_dimensions)
        self.beh_conv2 = MultiDimensionalNormalGCNConv(out_channels, out_channels, num_dimensions)
        self.beh_conv3 = MultiDimensionalNormalGCNConv(out_channels, out_channels, num_dimensions)


        self.pre_Lin1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.pre_Lin2 = torch.nn.Linear(out_channels, out_channels)
        self.pre_Lin3 = torch.nn.Linear(out_channels, out_channels)

        self.change_dim1 = torch.nn.Linear(in_channels, hidden_channels)
        self.change_dim2 = torch.nn.Linear(hidden_channels, out_channels)

        self.beh_Lin1 = torch.nn.Linear(out_channels, out_channels)
        self.beh_Lin2 = torch.nn.Linear(out_channels, out_channels)
        self.beh_Lin3 = torch.nn.Linear(out_channels, out_channels)

        self.dimension_weights = nn.Parameter(torch.ones(num_dimensions), requires_grad=True)
        self.k_save = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.threshold=0.5

    def forward(self, dim1, dim2, dim3, dim4, dim5, edge_indices, edge_weights, non_zero):
        input_dimension_weights = self.dimension_weights/self.dimension_weights.sum()
        sart_inputs = [dim1, dim2, dim3, dim4, dim5]

        ###########################   第 1   层    #####################################
        L1_by_GCN = self.pre_conv1(*sart_inputs, edge_indices, edge_weights)
        L1_G_sim_matrix = [compute_similarity_matrix_gpu(mid2_out) for mid2_out in L1_by_GCN]
        # 调用函数，并分别提取函数返回的三个值
        L1_results = [laplacian_eigen_decomposition_pytorch(G_sim) for G_sim in L1_G_sim_matrix]
        # 分别提取 sorted_eigenvalues_matrix, sorted_indices 和 eigenvectors    排序后的特征值对角矩阵    对应排序后的特征值原始索引的索引矩阵    未排序的特征向量矩阵
        L1_M_sorted_eigenvalues_matrix = [res[0] for res in L1_results]
        L1_M_sorted_indices = [res[1] for res in L1_results]  # 控制分类的阈值

        L1_M_class_flags = [compute_eigengap_classes(sort_f_value, self.threshold, index_of_vlue) for
                            sort_f_value, index_of_vlue in zip(L1_M_sorted_eigenvalues_matrix, L1_M_sorted_indices)]
        # 求取类的平均的特征向量    待优化
        L1_M_class_avger_feature = [compute_class_average_features(class_flag, G_feature) for class_flag, G_feature in
                                    zip(L1_M_class_flags, L1_by_GCN)]
        L1_Lin_outputs = [self.pre_Lin1(mid2_out.to(device)) for mid2_out in L1_M_class_avger_feature]

        L1_change_dim = [self.change_dim1(mid2_out.to(device)) for mid2_out in sart_inputs]
        L1_Fea_outputs = [add_class_avg_to_node_features(pro_feature, pro_index, fir_feature, self.k_save)
                          for pro_feature, pro_index, fir_feature in zip(L1_Lin_outputs, L1_M_class_flags, L1_change_dim)]
        L1_Out = [F.relu(mid2_in + mid2_Gout).to(torch.float32) for mid2_in, mid2_Gout in
                  zip(L1_change_dim, L1_Fea_outputs)]

        ###########################   第 2   层    #####################################


        L2_by_GCN = self.pre_conv2(*L1_by_GCN, edge_indices, edge_weights)
        L2_G_sim_matrix = [compute_similarity_matrix_gpu(mid2_out) for mid2_out in L2_by_GCN]
        # 调用函数，并分别提取函数返回的三个值
        L2_results = [laplacian_eigen_decomposition_pytorch(G_sim) for G_sim in L2_G_sim_matrix]
        # 分别提取 sorted_eigenvalues_matrix, sorted_indices 和 eigenvectors    排序后的特征值对角矩阵    对应排序后的特征值原始索引的索引矩阵    未排序的特征向量矩阵
        L2_M_sorted_eigenvalues_matrix = [res[0] for res in L2_results]
        L2_M_sorted_indices = [res[1] for res in L2_results]  # 控制分类的阈值

        L2_M_class_flags = [compute_eigengap_classes(sort_f_value, self.threshold, index_of_vlue) for
                            sort_f_value, index_of_vlue in zip(L2_M_sorted_eigenvalues_matrix, L2_M_sorted_indices)]
        # 求取类的平均的特征向量    待优化
        L2_M_class_avger_feature = [compute_class_average_features(class_flag, G_feature) for class_flag, G_feature in
                                    zip(L2_M_class_flags, L2_by_GCN)]
        L2_Lin_outputs = [self.pre_Lin2(mid2_out.to(device)) for mid2_out in L2_M_class_avger_feature]

        L2_change_dim = [self.change_dim2(mid2_out.to(device)) for mid2_out in L1_Out]
        L2_Fea_outputs = [add_class_avg_to_node_features(pro_feature, pro_index, fir_feature, self.k_save)
                          for pro_feature, pro_index, fir_feature in zip(L2_Lin_outputs, L2_M_class_flags, L2_change_dim)]
        L2_Out = [F.relu(mid2_in + mid2_Gout).to(torch.float32) for mid2_in, mid2_Gout in
                  zip(L2_change_dim, L2_Fea_outputs)]

        ###########################   第 3   层    #####################################

        L3_by_GCN = self.pre_conv3(*L2_by_GCN, edge_indices, edge_weights)
        L3_G_sim_matrix = [compute_similarity_matrix_gpu(mid2_out) for mid2_out in L3_by_GCN]
        # 调用函数，并分别提取函数返回的三个值
        L3_results = [laplacian_eigen_decomposition_pytorch(G_sim) for G_sim in L3_G_sim_matrix]
        # 分别提取 sorted_eigenvalues_matrix, sorted_indices 和 eigenvectors    排序后的特征值对角矩阵    对应排序后的特征值原始索引的索引矩阵    未排序的特征向量矩阵
        L3_M_sorted_eigenvalues_matrix = [res[0] for res in L3_results]
        L3_M_sorted_indices = [res[1] for res in L3_results]  # 控制分类的阈值

        L3_M_class_flags = [compute_eigengap_classes(sort_f_value, self.threshold, index_of_vlue) for
                            sort_f_value, index_of_vlue in zip(L3_M_sorted_eigenvalues_matrix, L3_M_sorted_indices)]
        # 求取类的平均的特征向量    待优化
        L3_M_class_avger_feature = [compute_class_average_features(class_flag, G_feature) for class_flag, G_feature in
                                    zip(L3_M_class_flags, L3_by_GCN)]
        L3_Lin_outputs = [self.pre_Lin3(mid2_out.to(device)) for mid2_out in L3_M_class_avger_feature]
        L3_Fea_outputs = [add_class_avg_to_node_features(pro_feature, pro_index, fir_feature, self.k_save)
                          for pro_feature, pro_index, fir_feature in zip(L3_Lin_outputs, L3_M_class_flags, L2_Out)]
        L3_Out = [F.relu(mid2_in + mid2_Gout).to(torch.float32) for mid2_in, mid2_Gout in
                  zip(L2_Out, L3_Fea_outputs)]

        ###########################   第 4   层    #####################################

        L4_by_GCN = self.beh_conv1(*L3_by_GCN, edge_indices, edge_weights)
        L4_G_sim_matrix = [compute_similarity_matrix_gpu(mid2_out) for mid2_out in L4_by_GCN]
        # 调用函数，并分别提取函数返回的三个值
        L4_results = [laplacian_eigen_decomposition_pytorch(G_sim) for G_sim in L4_G_sim_matrix]
        # 分别提取 sorted_eigenvalues_matrix, sorted_indices 和 eigenvectors    排序后的特征值对角矩阵    对应排序后的特征值原始索引的索引矩阵    未排序的特征向量矩阵
        L4_M_sorted_eigenvalues_matrix = [res[0] for res in L4_results]
        L4_M_sorted_indices = [res[1] for res in L4_results]                                               #控制分类的阈值

        L4_M_class_flags = [compute_eigengap_classes(sort_f_value, self.threshold,  index_of_vlue) for sort_f_value, index_of_vlue in zip(L4_M_sorted_eigenvalues_matrix, L4_M_sorted_indices)]
        #求取类的平均的特征向量    待优化
        L4_M_class_avger_feature = [compute_class_average_features(class_flag, G_feature) for class_flag, G_feature in zip(L4_M_class_flags, L4_by_GCN)]
        L4_Lin_outputs = [self.beh_Lin1(mid2_out.to(device)) for mid2_out in L4_M_class_avger_feature]
        L4_Fea_outputs = [add_class_avg_to_node_features(pro_feature, pro_index, fir_feature, self.k_save )
                      for pro_feature, pro_index,fir_feature in zip(L4_Lin_outputs, L4_M_class_flags,L3_Out)]
        L4_Out = [F.relu(mid2_in + mid2_Gout).to(torch.float32) for mid2_in, mid2_Gout in
                            zip(L3_Out, L4_Fea_outputs)]

        ###########################   第 5   层    #####################################
        L5_by_GCN = self.beh_conv2(*L4_by_GCN, edge_indices, edge_weights)
        L5_G_sim_matrix1 = [compute_similarity_matrix_gpu(mid2_out) for mid2_out in L5_by_GCN]
        # 调用函数，并分别提取函数返回的三个值
        L5_results = [laplacian_eigen_decomposition_pytorch(G_sim) for G_sim in L5_G_sim_matrix1]
        L5_M_sorted_eigenvalues_matrix = [res[0] for res in L5_results]
        L5_M_sorted_indices = [res[1] for res in L5_results]                                               #控制分类的阈值

        L5_M_class_flags = [compute_eigengap_classes(sort_f_value, self.threshold,  index_of_vlue)
                            for sort_f_value, index_of_vlue in zip(L5_M_sorted_eigenvalues_matrix, L5_M_sorted_indices)]
        #求取类的平均的特征向量    待优化
        L5_M_class_avger_feature1 = [compute_class_average_features(class_flag, G_feature) for class_flag, G_feature in zip(L5_M_class_flags, L5_by_GCN)]
        L5_Lin_outputs= [self.beh_Lin2(mid2_out.to(device)) for mid2_out in L5_M_class_avger_feature1]
        L5_Fea_outputs = [add_class_avg_to_node_features(pro_feature, pro_index, fir_feature, self.k_save )
                       for pro_feature, pro_index,fir_feature in zip(L5_Lin_outputs, L5_M_class_flags,L4_Out)]
        L5_Out = [F.relu(mid2_in + mid2_Gout).to(torch.float32) for mid2_in, mid2_Gout in
                      zip(L4_Out, L5_Fea_outputs)]

        ###########################   第 6   层    #####################################
        L6_by_GCN = self.beh_conv3(*L5_by_GCN, edge_indices, edge_weights)
        L6_G_sim_matrix2 = [compute_similarity_matrix_gpu(mid2_out) for mid2_out in L6_by_GCN]
        # 调用函数，并分别提取函数返回的三个值
        L6_results = [laplacian_eigen_decomposition_pytorch(G_sim) for G_sim in L6_G_sim_matrix2]
        L6_M_sorted_eigenvalues_matrix = [res[0] for res in L6_results]
        L6_M_sorted_indices = [res[1] for res in L6_results]  # 控制分类的阈值

        L6_M_class_flags = [compute_eigengap_classes(sort_f_value, self.threshold, index_of_vlue) for
                         sort_f_value, index_of_vlue in zip(L6_M_sorted_eigenvalues_matrix, L6_M_sorted_indices)]
        # 求取类的平均的特征向量    待优化
        L6_M_class_avger_feature2 = [compute_class_average_features(class_flag, G_feature) for class_flag, G_feature in
                                  zip(L6_M_class_flags, L6_by_GCN)]
        L6_Lin_outputs = [self.beh_Lin3(mid2_out.to(device)) for mid2_out in L6_M_class_avger_feature2]
        L6_Fea_outputs = [add_class_avg_to_node_features(pro_feature, pro_index, fir_feature, self.k_save) for
                       pro_feature, pro_index, fir_feature in zip(L6_Lin_outputs, L6_M_class_flags, L5_Out)]
        L6_Out = [F.relu(mid2_in + mid2_Gout).to(torch.float32) for mid2_in, mid2_Gout in
                       zip(L5_Out, L6_Fea_outputs)]

        epsilon = 1e-8
        out = sum(L6_Out)+epsilon
        return out

