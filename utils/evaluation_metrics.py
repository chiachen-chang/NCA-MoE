"""
持续学习评估指标
===============

这个模块实现了持续学习场景的综合评估指标，
重点关注抗遗忘能力。
"""

import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


class ContinualLearningMetrics:
    """
    持续学习场景的综合评估指标
    
    这个类提供了计算各种指标的方法，包括
    抗遗忘分数、性能退化分析和
    跨多个任务序列的稳定性测量。
    """
    
    def __init__(self):
        """初始化指标计算器"""
        self.reset()
    
    def reset(self):
        """重置所有存储的指标"""
        self.task_performances = {}  # {task_id: {epoch: performance}}
        self.final_performances = {}  # {task_id: final_performance}
        self.sequence_results = []  # 不同序列的结果列表
    
    def update_task_performance(self, task_id: int, epoch: int, performance: float):
        """
        更新特定任务在特定轮次的性能
        
        参数:
            task_id (int): 任务标识符
            epoch (int): 训练轮次
            performance (float): 性能指标（如F1分数）
        """
        if task_id not in self.task_performances:
            self.task_performances[task_id] = {}
        self.task_performances[task_id][epoch] = performance
    
    def update_final_performance(self, task_id: int, performance: float):
        """
        更新任务完成所有任务后的最终性能
        
        参数:
            task_id (int): 任务标识符
            performance (float): 最终性能指标
        """
        self.final_performances[task_id] = performance
    
    def calculate_immediate_performance(self, task_id: int) -> float:
        """
        计算学习任务后的即时性能
        
        参数:
            task_id (int): 任务标识符
            
        返回:
            float: 即时性能 (P_{t,t})
        """
        if task_id not in self.task_performances:
            return 0.0
        
        # 获取学习此任务后的即时性能
        # （即此任务为当前任务的轮次）
        task_epochs = sorted(self.task_performances[task_id].keys())
        if not task_epochs:
            return 0.0
        
        # 找到学习此任务的轮次
        # 这是简化方法 - 在实践中，您会更精确地跟踪
        return self.task_performances[task_id][task_epochs[-1]]
    
    def calculate_retained_performance(self, task_id: int) -> float:
        """
        计算完成所有任务后的保留性能
        
        参数:
            task_id (int): 任务标识符
            
        返回:
            float: 保留性能 (P_{n,t})
        """
        return self.final_performances.get(task_id, 0.0)
    
    def calculate_forgetting_rate(self, task_id: int) -> float:
        """
        计算特定任务的遗忘率
        
        参数:
            task_id (int): 任务标识符
            
        返回:
            float: 遗忘率 (P_{t,t} - P_{n,t})
        """
        immediate = self.calculate_immediate_performance(task_id)
        retained = self.calculate_retained_performance(task_id)
        return immediate - retained
    
    def calculate_average_forgetting_rate(self, task_ids: List[int]) -> float:
        """
        计算多个任务的平均遗忘率
        
        参数:
            task_ids (List[int]): 任务标识符列表
            
        返回:
            float: 平均遗忘率 (F)
        """
        if not task_ids:
            return 0.0
        
        forgetting_rates = [self.calculate_forgetting_rate(tid) for tid in task_ids]
        return np.mean(forgetting_rates)
    
    def calculate_average_initial_performance(self, task_ids: List[int]) -> float:
        """
        计算多个任务的平均初始性能
        
        参数:
            task_ids (List[int]): 任务标识符列表
            
        返回:
            float: 平均初始性能 (I)
        """
        if not task_ids:
            return 0.0
        
        initial_performances = [self.calculate_immediate_performance(tid) for tid in task_ids]
        return np.mean(initial_performances)
    
    def calculate_anti_forgetting_score(self, task_ids: List[int]) -> float:
        """
        计算任务序列的抗遗忘分数
        
        参数:
            task_ids (List[int]): 任务标识符列表
            
        返回:
            float: 抗遗忘分数 (AF-Score)
        """
        F = self.calculate_average_forgetting_rate(task_ids)
        I = self.calculate_average_initial_performance(task_ids)
        
        if I == 0:
            return 0.0
        
        return 1 - (F / I)
    
    def add_sequence_result(self, task_ids: List[int], af_score: float):
        """
        添加完整任务序列的结果
        
        参数:
            task_ids (List[int]): 此序列中的任务标识符列表
            af_score (float): 此序列的抗遗忘分数
        """
        self.sequence_results.append({
            'task_ids': task_ids,
            'af_score': af_score
        })
    
    def calculate_average_af_score(self) -> float:
        """
        计算多个序列的平均抗遗忘分数
        
        返回:
            float: 平均AF分数
        """
        if not self.sequence_results:
            return 0.0
        
        af_scores = [result['af_score'] for result in self.sequence_results]
        return np.mean(af_scores)
    
    def calculate_af_score_std(self) -> float:
        """
        计算序列间AF分数的标准差
        
        返回:
            float: AF分数的标准差
        """
        if not self.sequence_results:
            return 0.0
        
        af_scores = [result['af_score'] for result in self.sequence_results]
        return np.std(af_scores)
    
    def get_comprehensive_metrics(self, task_ids: List[int]) -> Dict[str, float]:
        """
        获取任务序列的综合指标
        
        参数:
            task_ids (List[int]): 任务标识符列表
            
        返回:
            Dict[str, float]: 包含所有指标的字典
        """
        metrics = {}
        
        # 单个任务指标
        for task_id in task_ids:
            metrics[f'task_{task_id}_immediate'] = self.calculate_immediate_performance(task_id)
            metrics[f'task_{task_id}_retained'] = self.calculate_retained_performance(task_id)
            metrics[f'task_{task_id}_forgetting'] = self.calculate_forgetting_rate(task_id)
        
        # 整体序列指标
        metrics['average_forgetting_rate'] = self.calculate_average_forgetting_rate(task_ids)
        metrics['average_initial_performance'] = self.calculate_average_initial_performance(task_ids)
        metrics['anti_forgetting_score'] = self.calculate_anti_forgetting_score(task_ids)
        
        # 多序列指标（如果可用）
        if self.sequence_results:
            metrics['average_af_score'] = self.calculate_average_af_score()
            metrics['af_score_std'] = self.calculate_af_score_std()
        
        return metrics


def calculate_classification_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """
    计算标准分类指标
    
    参数:
        y_true (List[int]): 真实标签
        y_pred (List[int]): 预测标签
        
    返回:
        Dict[str, float]: 包含精确率、召回率、f1、准确率的字典
    """
    metrics = {}
    
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    return metrics


def evaluate_model_performance(
    model,
    test_loader,
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    在测试数据集上评估模型性能
    
    参数:
        model: PyTorch模型
        test_loader: 测试数据的数据加载器
        device (str): 运行评估的设备
        
    返回:
        Dict[str, float]: 性能指标
    """
    import torch
    
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # 前向传播
            outputs = model(inputs)
            
            # 转换为概率和预测
            if outputs.dim() > 1:
                outputs = outputs.squeeze()
            
            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities > 0.5).long().cpu().numpy()
            
            all_predictions.extend(predictions)
            all_labels.extend(labels.cpu().numpy())
    
    return calculate_classification_metrics(all_labels, all_predictions)


def create_performance_matrix(
    task_orders: List[List[int]],
    performance_data: Dict[Tuple[int, int], float]
) -> np.ndarray:
    """
    创建用于可视化的性能矩阵
    
    参数:
        task_orders (List[List[int]]): 任务顺序列表
        performance_data (Dict[Tuple[int, int], float]): 性能数据
                                                       (task_order, task_id) -> performance
        
    返回:
        np.ndarray: 性能矩阵
    """
    num_orders = len(task_orders)
    num_tasks = len(task_orders[0]) if task_orders else 0
    
    matrix = np.zeros((num_orders, num_tasks))
    
    for order_idx, task_order in enumerate(task_orders):
        for task_idx, task_id in enumerate(task_order):
            key = (order_idx, task_id)
            if key in performance_data:
                matrix[order_idx, task_idx] = performance_data[key]
    
    return matrix


def print_evaluation_summary(metrics: Dict[str, float]):
    """
    打印格式化的评估指标摘要
    
    参数:
        metrics (Dict[str, float]): 指标字典
    """
    print("\n" + "="*60)
    print("评估摘要")
    print("="*60)
    
    # 抗遗忘指标
    if 'anti_forgetting_score' in metrics:
        print(f"抗遗忘分数: {metrics['anti_forgetting_score']:.4f}")
    
    if 'average_af_score' in metrics:
        print(f"平均AF分数: {metrics['average_af_score']:.4f} ± {metrics.get('af_score_std', 0):.4f}")
    
    # 性能指标
    if 'average_initial_performance' in metrics:
        print(f"平均初始性能: {metrics['average_initial_performance']:.4f}")
    
    if 'average_forgetting_rate' in metrics:
        print(f"平均遗忘率: {metrics['average_forgetting_rate']:.4f}")
    
    # 单个任务性能
    task_metrics = {k: v for k, v in metrics.items() if k.startswith('task_')}
    if task_metrics:
        print("\n单个任务性能:")
        for metric_name, value in sorted(task_metrics.items()):
            print(f"  {metric_name}: {value:.4f}")
    
    print("="*60)


if __name__ == "__main__":
    # 测试评估指标
    print("测试评估指标...")
    
    # 创建示例指标实例
    metrics = ContinualLearningMetrics()
    
    # 模拟一些任务性能
    task_ids = [1, 2, 3, 4, 5]
    
    # 添加一些示例数据
    for task_id in task_ids:
        metrics.update_task_performance(task_id, task_id, 0.9)  # 即时性能
        metrics.update_final_performance(task_id, 0.85)  # 保留性能
    
    # 计算指标
    af_score = metrics.calculate_anti_forgetting_score(task_ids)
    print(f"抗遗忘分数: {af_score:.4f}")
    
    # 测试分类指标
    y_true = [0, 1, 1, 0, 1]
    y_pred = [0, 1, 0, 0, 1]
    cls_metrics = calculate_classification_metrics(y_true, y_pred)
    print(f"分类指标: {cls_metrics}")
    
    print("评估指标测试完成")
