from torch_geometric.nn import GCNConv
from torch import nn

class GCN_Net(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):      # 类的初始化方法，接收三个参数：输入维度、隐藏层维度和输出维度
        super(GCN_Net, self).__init__()
        self.GCN1=GCNConv(input_dim,hidden_dim)
        self.GCN2=GCNConv(hidden_dim,output_dim)
    def forward(self,Features,A,E):                          # 定义前向传播方法，接收三个参数：特征矩阵、邻接矩阵和边的索引
        Features=self.GCN1(Features,A)
        Features=nn.functional.relu(Features) 
        Features=nn.functional.dropout(Features,training=self.training)     # 应用dropout操作，防止过拟合，training参数指示当前是否为训练模式
        Features=self.GCN2(Features,A)   
        src=Features[E[0]]                                                  # 根据边的源节点索引E[0]，从处理后的特征矩阵中获取源节点的特征
        dst=Features[E[1]]                                                   # 根据边的目标节点索引E[1]，从处理后的特征矩阵中获取目标节点的特征
        result=(src*dst).sum(dim=-1)                                         # 计算源节点特征和目标节点特征的逐元素乘积，然后在最后一个维度上求和，得到每条边的预测结果
        result=nn.functional.sigmoid(result)                                # 对每条边的预测结果应用sigmoid函数，将其映射到(0,1)区间，表示预测的边存在的概率
        return result
    