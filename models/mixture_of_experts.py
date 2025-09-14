"""
专家混合(MoE)持续学习框架
========================

这个模块实现了NCA-MoE持续学习框架中的动态专家混合架构，
支持自适应专家扩展和智能路由机制。
"""

import torch
import torch.nn as nn
from .nca_net import NCANet


class ExpertNet(nn.Module):
    """
    NCA-Net的专家网络包装器
    
    这个类将NCA-Net架构包装成MoE框架中的专家网络
    
    参数:
        input_dim (int): 每个n-gram的输入特征维度
        ngram_dim (int): n-gram维度数
        hidden_dim (int): 隐藏层维度
        output_dim (int): 输出维度
    """
    
    def __init__(self, input_dim, ngram_dim, hidden_dim, output_dim):
        super(ExpertNet, self).__init__()
        self.nca_net = NCANet(input_dim, ngram_dim, hidden_dim, output_dim)
    
    def forward(self, x):
        """
        专家网络的前向传播
        
        参数:
            x (torch.Tensor): 输入张量，形状为(B, ngram_dim, input_dim)
            
        返回:
            torch.Tensor: 专家输出，形状为(B, output_dim)
        """
        return self.nca_net(x)


class GateNet(nn.Module):
    """
    专家路由的门控网络
    
    这个网络学习如何根据任务特定的模式和输入特征
    将输入路由到合适的专家网络
    
    参数:
        input_dim (int): 每个n-gram的输入特征维度
        ngram_dim (int): n-gram维度数
        num_experts (int): 专家数量 (初始为0)
    """
    
    def __init__(self, input_dim, ngram_dim, num_experts=0):
        super(GateNet, self).__init__()
        self.input_dim = input_dim
        self.ngram_dim = ngram_dim
        self.num_experts = num_experts
        
        # 如果有专家，就创建门控网络；否则设为None
        if num_experts > 0:
            self.fc = nn.Linear(ngram_dim * input_dim, num_experts)
        else:
            self.fc = None
        
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        """
        门控网络的前向传播
        
        参数:
            x (torch.Tensor): 输入张量，形状为(B, ngram_dim, input_dim)
            
        返回:
            torch.Tensor: 路由权重，形状为(B, num_experts)
        """
        if self.fc is None:
            raise ValueError("门控网络未初始化，请先调用update_num_experts方法")
        
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # 展平为(B, ngram_dim * input_dim)
        
        gate_logits = self.fc(x)
        gate_weights = self.softmax(gate_logits)
        
        return gate_weights
    
    def update_num_experts(self, num_experts):
        """
        更新专家数量并重新初始化门控网络
        
        参数:
            num_experts (int): 新的专家数量
        """
        self.num_experts = num_experts
        
        if num_experts > 0:
            # 创建新的线性层
            device = next(self.parameters()).device if self.parameters() else torch.device('cpu')
            self.fc = nn.Linear(self.ngram_dim * self.input_dim, self.num_experts).to(device)
            
            # 初始化参数
            nn.init.xavier_uniform_(self.fc.weight)
            if self.fc.bias is not None:
                nn.init.zeros_(self.fc.bias)
        else:
            self.fc = None


class NCA_MoE(nn.Module):
    """
    NCA-MoE: N-gram通道注意力专家混合框架
    
    这是主要的框架类，通过动态专家扩展和智能路由机制
    实现自适应多任务持续学习
    
    参数:
        input_dim (int): 每个n-gram的输入特征维度 (默认500)
        ngram_dim (int): n-gram维度数 (默认5)
        hidden_dim (int): 隐藏层维度 (默认128)
        output_dim (int): 输出维度 (默认1)
        num_experts (int): 初始专家数量 (默认0)
    """
    
    def __init__(self, input_dim=500, ngram_dim=5, hidden_dim=128, output_dim=1, num_experts=0):
        super(NCA_MoE, self).__init__()
        self.input_dim = input_dim
        self.ngram_dim = ngram_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        
        # 专家网络列表
        self.experts = nn.ModuleList()
        
        # 门控网络
        self.gate = GateNet(input_dim, ngram_dim, num_experts)
    
    def add_expert(self, input_dim=None, ngram_dim=None, hidden_dim=None, output_dim=None):
        """
        向混合模型中添加新专家
        
        参数:
            input_dim (int, optional): 输入维度 (为None时使用默认值)
            ngram_dim (int, optional): N-gram维度 (为None时使用默认值)
            hidden_dim (int, optional): 隐藏维度 (为None时使用默认值)
            output_dim (int, optional): 输出维度 (为None时使用默认值)
        """
        # 使用提供的值或默认值
        input_dim = input_dim if input_dim is not None else self.input_dim
        ngram_dim = ngram_dim if ngram_dim is not None else self.ngram_dim
        hidden_dim = hidden_dim if hidden_dim is not None else self.hidden_dim
        output_dim = output_dim if output_dim is not None else self.output_dim
        
        # 创建新专家
        new_expert = ExpertNet(input_dim, ngram_dim, hidden_dim, output_dim)
        
        # 移动到与现有参数相同的设备
        device = next(self.parameters()).device if self.parameters() else torch.device('cpu')
        new_expert = new_expert.to(device)
        
        # 添加到专家列表
        self.experts.append(new_expert)
        self.num_experts = len(self.experts)
        
        # 更新门控网络
        self.gate.update_num_experts(self.num_experts)
    
    def freeze_old_experts(self):
        """
        冻结除最新添加的专家外的所有专家参数
        通过保护已学习的知识来防止灾难性遗忘
        """
        if len(self.experts) > 1:
            for expert in self.experts[:-1]:  # 冻结除最后一个外的所有专家
                for param in expert.parameters():
                    param.requires_grad = False
    
    def unfreeze_all_experts(self):
        """
        解冻所有专家参数，用于微调或评估
        """
        for expert in self.experts:
            for param in expert.parameters():
                param.requires_grad = True
    
    def forward(self, x):
        """
        MoE框架的前向传播
        
        参数:
            x (torch.Tensor): 输入张量，形状为(B, ngram_dim, input_dim)
            
        返回:
            torch.Tensor: 专家输出的加权组合 (B, output_dim)
        """
        if self.num_experts == 0:
            raise ValueError("没有可用的专家，请先添加至少一个专家")
        
        # 从门控网络获取路由权重
        gate_weights = self.gate(x)  # (B, num_experts)
        
        # 获取所有专家的输出
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))  # (B, output_dim)
        
        # 堆叠专家输出
        expert_outputs = torch.stack(expert_outputs, dim=1)  # (B, num_experts, output_dim)
        
        # 应用路由权重
        gate_weights = gate_weights.unsqueeze(-1)  # (B, num_experts, 1)
        output = torch.sum(gate_weights * expert_outputs, dim=1)  # (B, output_dim)
        
        return output
    
    def get_expert_outputs(self, x):
        """
        获取各个专家的输出（不加权）
        
        参数:
            x (torch.Tensor): 输入张量，形状为(B, ngram_dim, input_dim)
            
        返回:
            tuple: (gate_weights, expert_outputs) 其中gate_weights形状为(B, num_experts)
                   expert_outputs形状为(B, num_experts, output_dim)
        """
        if self.num_experts == 0:
            raise ValueError("没有可用的专家")
        
        gate_weights = self.gate(x)
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))
        
        expert_outputs = torch.stack(expert_outputs, dim=1)
        
        return gate_weights, expert_outputs
    
    def get_model_info(self):
        """
        获取当前模型状态信息
        
        返回:
            dict: 模型信息，包括专家数量和参数数量
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'num_experts': self.num_experts,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_dim': self.input_dim,
            'ngram_dim': self.ngram_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim
        }


def create_nca_moe(input_dim=500, ngram_dim=5, hidden_dim=128, output_dim=1):
    """
    创建NCA-MoE模型的工厂函数
    
    参数:
        input_dim (int): 每个n-gram的输入特征维度
        ngram_dim (int): n-gram维度数
        hidden_dim (int): 隐藏层维度
        output_dim (int): 输出维度
        
    返回:
        NCA_MoE: 初始化好的NCA-MoE模型
    """
    return NCA_MoE(input_dim, ngram_dim, hidden_dim, output_dim)


if __name__ == "__main__":
    # 测试NCA-MoE架构
    model = create_nca_moe()
    
    # 添加一些专家
    model.add_expert()
    model.add_expert()
    
    # 创建虚拟输入
    batch_size = 32
    ngram_dim = 5
    input_dim = 500
    x = torch.randn(batch_size, ngram_dim, input_dim)
    
    # 前向传播
    output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"模型信息: {model.get_model_info()}")
