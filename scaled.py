#用了scale的点积模型
"""
import numpy as np
import torch
import torch.nn as nn
class scaledattention(nn.Module):
    def __init__(self,scale):
        super().__init__()
        self.scale=scale
        self.softmax=nn.Softmax(dim=2)

    def forward(self,q,k,v,mask=None):
        u=torch.bmm(q,k.transpose(1,2))
        u=u/self.scale
        if mask is not None:
            u=u.masked_fill(mask,-np.inf)
        attention=self.softmax(u)
        output=torch.bmm(attention,v)

        return attention,output

if __name__=="__main__":
    n_q,n_k,n_v=2,4,4#num
    d_q,d_k,d_v=128,128,64#dim
    batch=16
    q=torch.randn(batch,n_q,d_q)
    k=torch.randn(batch,n_k,d_k)
    v=torch.randn(batch,n_v,d_v)
    mask=torch.randn(batch,n_q,n_k,).bool()

    attn=scaledattention(scale=np.power(d_k,0,5))
    attention,output=attn(q,k,v,mask=mask)
    print(attention)
    print(output)
"""
import numpy as np
import torch
import torch.nn as nn

class ScaledDotProductAttention(nn.Module):
    """
    缩放点积注意力机制的实现
    
    参数:
        scale: 缩放因子，通常为键向量维度的平方根
    """
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
        self.softmax = nn.Softmax(dim=-1)  # 在最后一个维度上应用softmax

    def forward(self, q, k, v, mask=None):
        """
        前向传播函数
        
        参数:
            q: 查询张量，形状为 [batch_size, num_queries, d_q]
            k: 键张量，形状为 [batch_size, num_keys, d_k]
            v: 值张量，形状为 [batch_size, num_values, d_v]
            mask: 可选的掩码张量，形状为 [batch_size, num_queries, num_keys]
        
        返回:
            attention: 注意力权重矩阵
            output: 注意力输出
        """
        # 验证输入维度
        assert q.size(-1) == k.size(-1), "查询和键的维度必须相同"
        assert k.size(1) == v.size(1), "键和值的数量必须相同"
        
        # 计算注意力分数
        attention_scores = torch.bmm(q, k.transpose(1, 2))  # [batch_size, num_queries, num_keys]
        attention_scores = attention_scores / self.scale  # 缩放
        
        # 应用掩码（如果提供）
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask, -np.inf)
        
        # 计算注意力权重
        attention = self.softmax(attention_scores)
        
        # 计算输出
        output = torch.bmm(attention, v)  # [batch_size, num_queries, d_v]
        
        return attention, output

if __name__ == "__main__":
    # 设置参数
    batch_size = 16
    num_queries = 2
    num_keys = 4
    num_values = 4  # 通常 num_keys == num_values
    d_q = 128  # 查询维度
    d_k = 128  # 键维度
    d_v = 64   # 值维度
    
    # 生成随机输入
    q = torch.randn(batch_size, num_queries, d_q)
    k = torch.randn(batch_size, num_keys, d_k)
    v = torch.randn(batch_size, num_values, d_v)
    
    # 生成随机掩码（可选）
    mask = torch.rand(batch_size, num_queries, num_keys) > 0.5  # 随机掩码
    
    # 创建注意力模块
    scale = d_k ** 0.5  # 正确的缩放因子计算
    attention_module = ScaledDotProductAttention(scale=scale)
    
    # 前向传播
    attention_weights, output = attention_module(q, k, v, mask=mask)
    
    # 打印结果形状
    print(f"注意力权重形状: {attention_weights.shape}")
    print(f"输出形状: {output.shape}")
    
    # 打印前几个值（可选）
    print("\n注意力权重示例:")
    print(attention_weights[0, :, :])  # 打印第一个批次的注意力权重
    print("\n输出示例:")
    print(output[0, :, :])  # 打印第一个批次的输出