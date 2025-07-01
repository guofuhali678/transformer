import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    """简化版多头自注意力模块"""
    
    def __init__(self, n_head, embedding_dim, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.embedding_dim = embedding_dim  # 输入嵌入维度
        self.head_dim = embedding_dim // n_head  # 每个头的维度
        
        # 定义QKV投影矩阵
        self.wq = nn.Linear(embedding_dim, embedding_dim)
        self.wk = nn.Linear(embedding_dim, embedding_dim)
        self.wv = nn.Linear(embedding_dim, embedding_dim)
        
        # 输出投影
        self.wo = nn.Linear(embedding_dim, embedding_dim)
        
        # Dropout和缩放因子
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** (-0.5)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        
        # 线性投影并拆分为多头
        q = self.wq(x).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数并缩放
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # 应用掩码（如果有）
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        
        # 计算注意力权重
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # 加权求和并合并多头
        output = torch.matmul(attn, v).transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.embedding_dim)
        
        # 最终线性变换
        output = self.wo(output)
        
        return output, attn


# 示例使用
if __name__ == "__main__":
    # 参数设置
    batch_size = 4
    seq_len = 10
    embedding_dim = 512  # 现在使用embedding_dim
    n_head = 8
    
    # 创建输入张量
    x = torch.randn(batch_size, seq_len, embedding_dim)
    
    # 创建掩码（可选）
    mask = torch.ones(batch_size, seq_len, seq_len).bool()
    
    # 实例化自注意力模块
    attn_module = SelfAttention(n_head=n_head, embedding_dim=embedding_dim)
    
    # 前向传播
    output, attention = attn_module(x, mask)
    
    # 打印输出形状
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {attention.shape}")
    print(f"实例：{attention[0,:,:]}")
    print(f"维数前显示：{output[0,:,:]}")
    