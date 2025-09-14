"""
NCA-Net: N-gram通道注意力网络
============================

这个模块实现了NCA-MoE持续学习框架中的NCA-Net专家架构，
集成了通道注意力机制来提升特征表示能力。
"""

import torch
import torch.nn as nn


class SKLayer(nn.Module):
    """
    选择性核(SK)层，用于多分支特征融合
    
    这个层实现了一个简化版的SKNet架构，专门用来处理(B, C, L)格式的张量
    通过两个并行的卷积分支和自适应通道选择来提升特征表示能力
    
    参数:
        channel (int): 输入通道数 (默认5，对应n-gram维度)
        reduction_ratio (int): 注意力计算的通道压缩比例 (默认2)
    """
    
    def __init__(self, channel=5, reduction_ratio=2):
        super(SKLayer, self).__init__()
        self.channel = channel
        
        # 定义两个并行的1D卷积分支，用不同的卷积核大小来捕获不同尺度的特征
        self.branch1 = nn.Conv1d(
            in_channels=channel,
            out_channels=channel,
            kernel_size=3,
            padding=1,
            groups=channel,  # 深度可分离卷积，每个通道独立处理
            bias=False
        )
        
        self.branch2 = nn.Conv1d(
            in_channels=channel,
            out_channels=channel,
            kernel_size=5,
            padding=2,
            groups=channel,  # 深度可分离卷积，每个通道独立处理
            bias=False
        )
        
        # 通道注意力机制，类似SE-Net的设计思路
        reduced_size = max(1, channel // reduction_ratio)
        self.fc = nn.Sequential(
            nn.Linear(channel, reduced_size, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_size, channel * 2, bias=False)  # 输出2*C，分别对应两个分支的权重
        )
        
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        """
        SKLayer的前向传播
        
        参数:
            x (torch.Tensor): 输入张量，形状为(B, C, L)
            
        返回:
            torch.Tensor: 注意力加权后的输出，形状为(B, C, L)
        """
        B, C, L = x.shape
        
        # 并行卷积分支处理
        out1 = self.branch1(x)  # (B, C, L)
        out2 = self.branch2(x)  # (B, C, L)
        
        # 全局平均池化来获取通道信息
        out_sum = out1 + out2  # (B, C, L)
        U = torch.mean(out_sum, dim=-1)  # (B, C)
        
        # 生成门控权重
        gates = self.fc(U)  # (B, 2*C)
        alpha1, alpha2 = gates[:, :C], gates[:, C:]  # 分成两部分
        alpha1 = self.softmax(alpha1)  # (B, C)
        alpha2 = self.softmax(alpha2)  # (B, C)
        
        # 应用注意力权重
        alpha1 = alpha1.unsqueeze(-1)  # (B, C, 1)
        alpha2 = alpha2.unsqueeze(-1)  # (B, C, 1)
        
        # 加权融合两个分支的结果
        out = alpha1 * out1 + alpha2 * out2  # (B, C, L)
        
        return out


class AttentionBlock(nn.Module):
    """
    注意力增强的MLP块，带有残差连接
    
    这个块将基于SK的通道注意力与MLP层集成，实现动态特征重校准
    同时保持稳定的梯度流
    
    参数:
        ngram_dim (int): n-gram维度数 (默认5)
        feature_dim (int): 每个n-gram的特征维度 (默认500)
    """
    
    def __init__(self, ngram_dim=5, feature_dim=500):
        super(AttentionBlock, self).__init__()
        self.ngram_dim = ngram_dim
        self.feature_dim = feature_dim
        flattened_dim = ngram_dim * feature_dim  # 5*500 = 2500
        
        # 两层MLP，每层都有批归一化
        self.fc1 = nn.Linear(flattened_dim, flattened_dim)
        self.bn1 = nn.BatchNorm1d(flattened_dim)
        
        self.fc2 = nn.Linear(flattened_dim, flattened_dim)
        self.bn2 = nn.BatchNorm1d(flattened_dim)
        
        # 基于SK的通道注意力机制
        self.sk = SKLayer(channel=self.ngram_dim, reduction_ratio=2)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        """
        AttentionBlock的前向传播
        
        参数:
            x (torch.Tensor): 输入张量，形状为(B, ngram_dim, feature_dim)
            
        返回:
            torch.Tensor: 注意力增强后的输出，形状为(B, ngram_dim, feature_dim)
        """
        # 保存残差连接的输入
        residual = x  # (B, 5, 500)
        
        B, C, L = x.shape
        # 展平用于MLP处理
        out = x.view(B, -1)  # (B, 2500)
        
        # 第一层MLP
        out = self.fc1(out)
        out = self.bn1(out)
        out = self.relu(out)
        
        # 第二层MLP
        out = self.fc2(out)
        out = self.bn2(out)
        
        # 重塑回原始维度
        out = out.view(B, C, L)  # (B, 5, 500)
        
        # 应用SK注意力
        out = self.sk(out)  # (B, 5, 500)
        
        # 残差连接和激活
        out += residual
        out = self.relu(out)
        
        return out


class NCANet(nn.Module):
    """
    NCA-Net: N-gram通道注意力网络
    
    这是NCA-MoE框架中的核心专家架构，结合了多粒度n-gram表示
    和通道注意力机制来提升威胁检测能力
    
    参数:
        input_dim (int): 每个n-gram的输入特征维度 (默认500)
        ngram_dim (int): n-gram维度数 (默认5)
        hidden_dim (int): 隐藏层维度 (默认128)
        output_dim (int): 输出维度 (默认1，用于二分类)
    """
    
    def __init__(self, input_dim=500, ngram_dim=5, hidden_dim=128, output_dim=1):
        super(NCANet, self).__init__()
        self.input_dim = input_dim
        self.ngram_dim = ngram_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 注意力增强的MLP块
        self.attention_block = AttentionBlock(ngram_dim, input_dim)
        
        # 输出头
        self.fc1 = nn.Linear(ngram_dim * input_dim, hidden_dim)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        """
        NCA-Net的前向传播
        
        参数:
            x (torch.Tensor): 输入张量，形状为(B, ngram_dim, input_dim)
            
        返回:
            torch.Tensor: 输出logits，形状为(B, output_dim)
        """
        # 应用注意力增强处理
        x = self.attention_block(x)  # (B, 5, 500)
        
        # 展平用于最终处理
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # (B, 2500)
        
        # 输出头
        x = self.fc1(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x


def create_nca_net(input_dim=500, ngram_dim=5, hidden_dim=128, output_dim=1):
    """
    创建NCA-Net模型的工厂函数
    
    参数:
        input_dim (int): 每个n-gram的输入特征维度
        ngram_dim (int): n-gram维度数
        hidden_dim (int): 隐藏层维度
        output_dim (int): 输出维度
        
    返回:
        NCANet: 初始化好的NCA-Net模型
    """
    return NCANet(input_dim, ngram_dim, hidden_dim, output_dim)


if __name__ == "__main__":
    # 测试NCA-Net架构
    model = create_nca_net()
    
    # 创建虚拟输入
    batch_size = 32
    ngram_dim = 5
    input_dim = 500
    x = torch.randn(batch_size, ngram_dim, input_dim)
    
    # 前向传播
    output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
