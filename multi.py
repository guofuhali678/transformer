import numpy as np
import torch
import torch.nn as nn



"""
class MultiHeadAttention(nn.Module):

    def __init__(self, n_head, d_k_, d_v_, d_k, d_v, d_o):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.fc_q = nn.Linear(d_k_, n_head * d_k)
        self.fc_k = nn.Linear(d_k_, n_head * d_k)
        self.fc_v = nn.Linear(d_v_, n_head * d_v)

        self.attention = ScaledDotProductAttention(scale=np.power(d_k, 0.5))

        self.fc_o = nn.Linear(n_head * d_v, d_o)

    def forward(self, q, k, v, mask=None):

        n_head, d_q, d_k, d_v = self.n_head, self.d_k, self.d_k, self.d_v

        batch, n_q, d_q_ = q.size()
        batch, n_k, d_k_ = k.size()
        batch, n_v, d_v_ = v.size()

        q = self.fc_q(q) # 1.单头变多头
        k = self.fc_k(k)
        v = self.fc_v(v)
        q = q.view(batch, n_q, n_head, d_q).permute(2, 0, 1, 3).contiguous().view(-1, n_q, d_q)
        k = k.view(batch, n_k, n_head, d_k).permute(2, 0, 1, 3).contiguous().view(-1, n_k, d_k)
        v = v.view(batch, n_v, n_head, d_v).permute(2, 0, 1, 3).contiguous().view(-1, n_v, d_v)

        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)
        attn, output = self.attention(q, k, v, mask=mask) # 2.当成单头注意力求输出

        output = output.view(n_head, batch, n_q, d_v).permute(1, 2, 0, 3).contiguous().view(batch, n_q, -1) # 3.Concat
        output = self.fc_o(output) # 4.仿射变换得到最终输出

        return attn, output


if __name__ == "__main__":
    n_q, n_k, n_v = 2, 4, 4
    d_q_, d_k_, d_v_ = 128, 128, 64
    batch=16
    q = torch.randn(batch, n_q, d_q_)
    k = torch.randn(batch, n_k, d_k_)
    v = torch.randn(batch, n_v, d_v_)    
    mask = torch.zeros(batch, n_q, n_k).bool()

    mha = MultiHeadAttention(n_head=8, d_k_=128, d_v_=64, d_k=256, d_v=128, d_o=128)
    attn, output = mha(q, k, v, mask=mask)

    print(attn.size())
    print(output.size())
"""

import numpy as np
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    """ 多头注意力模块 """

    def __init__(self, n_head, d_model, d_k, d_v, d_o):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.scale = np.sqrt(d_k)  # 缩放因子
        
        # 线性变换层
        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        self.fc = nn.Linear(n_head * d_v, d_o)

    def forward(self, q, k, v, mask=None):
        batch_size, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        
        # 线性投影
        q = self.w_qs(q).view(batch_size, len_q, self.n_head, self.d_k)
        k = self.w_ks(k).view(batch_size, len_k, self.n_head, self.d_k)
        v = self.w_vs(v).view(batch_size, len_v, self.n_head, self.d_v)
        
        # 调整维度顺序以进行并行计算
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, self.d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, self.d_k)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, self.d_v)
        
        # 处理掩码
        if mask is not None:
            mask = mask.repeat(self.n_head, 1, 1)
        
        # 计算注意力分数
        attn = torch.bmm(q, k.transpose(1, 2)) / self.scale
        
        # 应用掩码（如果提供）
        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)
        
        # 应用softmax获取注意力权重
        attn = torch.softmax(attn, dim=-1)
        
        # 计算输出
        output = torch.bmm(attn, v)
        output = output.view(self.n_head, batch_size, len_q, self.d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(batch_size, len_q, -1)
        
        # 最终线性变换
        output = self.fc(output)
        
        return attn, output

if __name__ == "__main__":
    # 设置参数，确保维度匹配
    n_head = 8  # 头数
    d_model = 128  # 输入维度
    d_k = d_model // n_head  # 每个头的键维度
    d_v = d_model // n_head  # 每个头的值维度
    d_o = 128  # 输出维度
    
    batch_size = 16
    seq_len_q = 2  # 查询序列长度
    seq_len_k = 4  # 键/值序列长度
    
    # 创建输入张量
    q = torch.randn(batch_size, seq_len_q, d_model)
    k = torch.randn(batch_size, seq_len_k, d_model)
    v = torch.randn(batch_size, seq_len_k, d_model)  # 通常k和v长度相同
    
    # 创建掩码（可选）
    mask = torch.zeros(batch_size, seq_len_q, seq_len_k).bool()
    
    # 实例化多头注意力模块
    mha = MultiHeadAttention(n_head=n_head, d_model=d_model, d_k=d_k, d_v=d_v, d_o=d_o)
    
    # 前向传播
    attn, output = mha(q, k, v, mask=mask)
    
    # 打印结果形状
    print(f"注意力权重形状: {attn.shape}")  # 应为 [batch*n_head, seq_len_q, seq_len_k]
    print(f"输出形状: {output.shape}")     # 应为 [batch, seq_len_q, d_o]
    
    # 打印部分结果值
    print("\n注意力权重示例:")
    print(attn[0, :, :])  # 打印第一个批次的前5x5注意力权重矩阵
    print("\n输出示例:")
    print(output[0, :, :])  # 打印第一个批次第一个位置的前5个值

