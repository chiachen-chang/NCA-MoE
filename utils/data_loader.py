"""
数据加载和预处理工具
=================

这个模块为NCA-MoE持续学习框架提供
网络安全数据集的加载和预处理工具。
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from typing import Tuple, Optional, List


class WebSecurityDataset(Dataset):
    """
    具有多实例n-gram表示的网络安全数据自定义数据集类
    
    这个数据集从numpy文件加载预处理的n-gram特征和标签，
    将它们重塑为NCA-MoE框架所需的格式。
    
    参数:
        data_path (str): numpy数据文件路径
        input_shape (tuple): 期望的输入形状 (ngram_dim, feature_dim)
    """
    
    def __init__(self, data_path: str, input_shape: Tuple[int, int] = (5, 500)):
        """
        初始化数据集
        
        参数:
            data_path (str): numpy数据文件路径
            input_shape (tuple): 期望的输入形状 (ngram_dim, feature_dim)
        """
        self.data_path = data_path
        self.input_shape = input_shape
        
        # 加载数据
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据文件未找到: {data_path}")
        
        self.data = np.load(data_path)
        print(f"从 {data_path} 加载数据: {self.data.shape}")
        
        # 提取标签和特征
        self.labels = self.data[:, -1]  # 最后一列是标签
        self.features = self.data[:, :-1]  # 除最后一列外的所有列
        
        # 重塑特征以匹配期望的输入形状
        self.features = self.features.reshape(-1, *input_shape)
        
        print(f"重塑特征为: {self.features.shape}")
        print(f"标签形状: {self.labels.shape}")
    
    def __len__(self) -> int:
        """返回数据集中的样本数量"""
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从数据集中获取样本
        
        参数:
            idx (int): 样本索引
            
        返回:
            tuple: (features, label) 其中features是(ngram_dim, feature_dim)，label是标量
        """
        sample = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return sample, label


def get_dataloaders(
    train_path: str,
    val_path: str, 
    test_path: str,
    batch_size: int = 32,
    input_shape: Tuple[int, int] = (5, 500),
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建训练、验证和测试的数据加载器
    
    参数:
        train_path (str): 训练数据路径
        val_path (str): 验证数据路径
        test_path (str): 测试数据路径
        batch_size (int): 数据加载器的批次大小
        input_shape (tuple): 期望的输入形状 (ngram_dim, feature_dim)
        num_workers (int): 数据加载的工作进程数
        pin_memory (bool): 是否固定内存以加快GPU传输
        
    返回:
        tuple: (train_loader, val_loader, test_loader)
    """
    # 创建数据集
    train_dataset = WebSecurityDataset(train_path, input_shape)
    val_dataset = WebSecurityDataset(val_path, input_shape)
    test_dataset = WebSecurityDataset(test_path, input_shape)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader


def select_replay_samples(
    dataset: WebSecurityDataset, 
    percentage: float = 0.1,
    random_seed: Optional[int] = None
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    选择用于经验回放的代表性样本
    
    这个函数实现比例采样以在回放缓冲区中保持类别平衡
    
    参数:
        dataset (WebSecurityDataset): 要采样的数据集
        percentage (float): 要选择的样本百分比 (默认: 0.1)
        random_seed (int, optional): 用于可重现性的随机种子
        
    返回:
        tuple: 用于回放的(selected_inputs, selected_labels)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # 获取每个类别的索引
    label1_indices = [i for i, (_, label) in enumerate(dataset) if label == 1]
    label0_indices = [i for i, (_, label) in enumerate(dataset) if label == 0]
    
    # 计算每个类别要选择的样本数量
    n1 = max(1, int(len(label1_indices) * percentage))
    n0 = max(1, int(len(label0_indices) * percentage))
    
    # 采样索引
    selected_label1 = np.random.choice(
        label1_indices, 
        size=min(n1, len(label1_indices)), 
        replace=False
    ).tolist()
    
    selected_label0 = np.random.choice(
        label0_indices,
        size=min(n0, len(label0_indices)),
        replace=False
    ).tolist()
    
    # 合并并打乱
    selected_indices = selected_label1 + selected_label0
    np.random.shuffle(selected_indices)
    
    # 提取选定的样本
    selected_inputs = []
    selected_labels = []
    
    for idx in selected_indices:
        input_data, label = dataset[idx]
        selected_inputs.append(input_data)
        selected_labels.append(label)
    
    return selected_inputs, selected_labels


def create_replay_buffer(
    replay_inputs: List[torch.Tensor],
    replay_labels: List[torch.Tensor]
) -> DataLoader:
    """
    从回放缓冲区样本创建数据加载器
    
    参数:
        replay_inputs (List[torch.Tensor]): 回放输入样本
        replay_labels (List[torch.Tensor]): 回放标签
        
    返回:
        DataLoader: 回放样本的数据加载器
    """
    if not replay_inputs or not replay_labels:
        return None
    
    # 堆叠张量
    replay_inputs_tensor = torch.stack(replay_inputs)
    replay_labels_tensor = torch.tensor(replay_labels)
    
    # 创建数据集
    replay_dataset = torch.utils.data.TensorDataset(
        replay_inputs_tensor, 
        replay_labels_tensor
    )
    
    # 创建数据加载器
    replay_loader = DataLoader(
        replay_dataset,
        batch_size=32,  # Default batch size
        shuffle=True
    )
    
    return replay_loader


def get_dataset_statistics(data_path: str) -> dict:
    """
    获取数据集的统计信息
    
    参数:
        data_path (str): 数据集文件路径
        
    返回:
        dict: 数据集统计信息，包括形状、类别分布等
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件未找到: {data_path}")
    
    data = np.load(data_path)
    labels = data[:, -1]
    
    stats = {
        'total_samples': len(data),
        'feature_dim': data.shape[1] - 1,
        'positive_samples': int(np.sum(labels)),
        'negative_samples': int(len(labels) - np.sum(labels)),
        'positive_ratio': float(np.mean(labels)),
        'negative_ratio': float(1 - np.mean(labels))
    }
    
    return stats


if __name__ == "__main__":
    # 测试数据加载工具
    print("测试数据加载工具...")
    
    # 示例用法（需要实际数据文件）
    # train_loader, val_loader, test_loader = get_dataloaders(
    #     "path/to/train.npy",
    #     "path/to/val.npy", 
    #     "path/to/test.npy"
    # )
    
    print("数据加载工具测试完成")
