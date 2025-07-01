import numpy as np
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self,embedding_dim):
        #必须传入embedding_dim,类的对象声明
        super(SelfAttention,self).__init__()
        #super().__init__()
        self.embedding_dim=embedding_dim
        self.Wq=nn.Linear(embedding_dim,embedding_dim)
        self.Wk=nn.Linear(embedding_dim,embedding_dim)      
        self.Wv=nn.Linear(embedding_dim,embedding_dim)         
    def forward(self,x):
        batch_size,seq_length,embedding_dim=x.size()
        Q=self.Wq(x)
        K=self.Wk(x)
        V=self.Wv(x)

        attention_scores=torch.matual(Q,K.transpose(1,2))
        attention_scores=attention_scores/(embedding_dim**0.5)
        attention_weights=nn.functional.softmax(attention_scores,dim=2)

        O=torch.matual(attention_weights,V)
        #Q是你要查询的内容,表示你对某个位置的兴趣
        #K是提供“信息”的键,表示其他位置的信息
        #V是实际的值(数据),表示你想要提取的信息
        #每个查询(Q)会与所有的键(K)进行比较,得到一个相似度,然后这些相似度会被转化为权重,最终用这些权重对值(V)进行加权求和,得到该查询的输出
        
        return O


class  MultiHeadAttention(nn.Module):
    def __init(self,batch_size,seq_length,embedding_dim,head_num):
        super().__init__()
        self.batch_size=batch_size
        self.seq_length=seq_length
        embedding_dim=embedding_dim
        head_num=head_num

        self.head_dim=embedding_dim//head_num

        self.Wq=nn.Linear(embedding_dim,embedding_dim)
        self.Wk=nn.Linear(embedding_dim,embedding_dim)
        self.Wv=nn.Linear(embedding_dim,embedding_dim)
        self.Wo=nn.Linear(embedding_dim,embedding_dim)

    def forward(self,x):
        Q=self.Wq(x)
        K=self.Wk(x)
        V=self.Wv(x)

        Q=Q.view(self.batch_size,self.seq_length,self.head_dim,self.head_num)
        K=K.view(self.batch_size,self.seq_length,self.head_dim,self.head_num)
        V=V.view(self.batch_size,self.seq_length,self.head_dim,self.head_num)

        Q=Q.permute(0,2,1,3)
        K=K.permute(0,2,1,3)
        V=V.permute(0,2,1,3)

        attention_scores=torch.matual(Q,K.transpose(2,3))
        attention_scores=attention_scores/(self.head_dim**0.5)
        attention_weights=nn.functional.softmax(attention_scores,dim=-1)
        O=torch.matual(attention_weights,V)
        O=O.permute(0,2,1,3)
        O=O.contiguous().view(self.batch_size,self.seq_length,self.embedding_dim)
        Q=self.Wo(O)#输出再次映射回原来的空间


