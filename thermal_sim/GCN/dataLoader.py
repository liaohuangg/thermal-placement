import torch
from torch.utils.data import Dataset, DataLoader
import dgl
import numpy as np
from numpy import genfromtxt

# 复用你代码中的归一化参数（需和主代码保持一致）
class ThermalGraphDataset(Dataset):
    def __init__(self, data_list, dataset_dir, 
                 Power_min, Power_max, tPower_min, tPower_max,
                 Temperature_min, Temperature_max, Conductance_min, Conductance_max):
        """
        Args:
            data_list: 训练/测试集的case_num列表（如 ['case1', 'case2', ...]）
            dataset_dir: 数据集根目录
            其余参数：归一化用的最大/最小值
        """
        self.data_list = data_list
        self.dataset_dir = dataset_dir
        self.Power_min = Power_min
        self.Power_max = Power_max
        self.tPower_min = tPower_min
        self.tPower_max = tPower_max
        self.Temperature_min = Temperature_min
        self.Temperature_max = Temperature_max
        self.Conductance_min = Conductance_min
        self.Conductance_max = Conductance_max

    def __len__(self):
        # 返回数据集总长度
        return len(self.data_list)

    def __getitem__(self, idx):
        # 单样本加载逻辑（DataLoader会并行调用此方法）
        case_num = self.data_list[idx]
        
        # 1. 读取节点特征（复用你原有的归一化逻辑）
        # 功率特征
        node_feats_read1 = genfromtxt(self.dataset_dir + f'Power_{case_num}.csv', delimiter=',')
        node_feats1 = np.zeros((len(node_feats_read1), 1), dtype=np.float32)
        for i in range(len(node_feats_read1)):
            node_feats1[i, 0] = (node_feats_read1[i,1] - self.Power_min)/(self.Power_max - self.Power_min)*2 - 1
        
        # 总功率特征
        node_feats_read2 = genfromtxt(self.dataset_dir + f'totalPower_{case_num}.csv', delimiter=',')
        node_feats2 = np.zeros((len(node_feats_read2), 1), dtype=np.float32)
        for i in range(len(node_feats_read2)):
            node_feats2[i, 0] = (node_feats_read2[i] - self.tPower_min)/(self.tPower_max - self.tPower_min)*2 - 1
        
        # 温度标签
        node_labels_read = genfromtxt(self.dataset_dir + f'Temperature_{case_num}.csv', delimiter=',')
        node_labels = np.zeros((len(node_labels_read), 1), dtype=np.float32)
        for i in range(len(node_labels_read)):
            node_labels[i, 0] = node_labels_read[i, 1]
        node_labels = (node_labels - self.Temperature_min)/(self.Temperature_max - self.Temperature_min)*2 - 1

        # 2. 读取边特征和图结构
        edge_data = genfromtxt(self.dataset_dir + f'Edge_{case_num}.csv', delimiter=',')
        # 构建DGL图（注意：DGL图不能直接序列化，需返回原始边数据，在collate_fn中构建）
        src_nodes = edge_data[:, 0].astype(np.int64)
        dst_nodes = edge_data[:, 1].astype(np.int64)
        # 边特征（电导）归一化 - 按照原始代码逻辑：edge_feats[i][0] = edge_data[i, 1]
        edge_feats = np.zeros((len(edge_data), 1), dtype=np.float32)
        for i in range(len(edge_data)):
            # 先保存原始电导值，然后归一化
            edge_feats[i, 0] = edge_data[i, 1]  # 原始代码逻辑
        edge_feats = (edge_feats - self.Conductance_min) / (self.Conductance_max - self.Conductance_min) * 2 - 1

        # 返回单样本数据（图结构拆分为src/dst，避免DGL图在多进程中出错）
        return {
            'case_num': case_num,
            'node_feats1': node_feats1,
            'node_feats2': node_feats2,
            'node_labels': node_labels,
            'src_nodes': src_nodes,
            'dst_nodes': dst_nodes,
            'edge_feats': edge_feats
        }

# 定义自定义collate_fn：将多个样本拼接为批次（处理图数据的批量合并）
def collate_fn(batch):
    """
    自定义批次拼接函数，处理图数据的批量合并（DGL.batch）
    Args:
        batch: 列表，每个元素是__getitem__返回的单样本字典
    Returns:
        批量处理后的图、特征、标签
    """
    case_nums = [item['case_num'] for item in batch]
    # 合并节点特征（批量拼接）
    node_feats1 = np.vstack([item['node_feats1'] for item in batch])
    node_feats2 = np.vstack([item['node_feats2'] for item in batch])
    node_labels = np.vstack([item['node_labels'] for item in batch])
    # 合并边特征
    edge_feats = np.vstack([item['edge_feats'] for item in batch])
    
    # 构建批量图（DGL.batch）
    g_list = []
    for item in batch:
        g = dgl.graph((item['src_nodes'], item['dst_nodes']))
        g_list.append(g)
    batched_g = dgl.batch(g_list)

    # 转换为Tensor（移到CPU，后续再推到GPU）
    return {
        'case_nums': case_nums,
        'g': batched_g,
        'node_feats1': torch.from_numpy(node_feats1),
        'node_feats2': torch.from_numpy(node_feats2),
        'node_labels': torch.from_numpy(node_labels),
        'edge_feats': torch.from_numpy(edge_feats)
    }


def create_dataloader(data_list, dataset_dir, 
                     Power_min, Power_max, tPower_min, tPower_max,
                     Temperature_min, Temperature_max, Conductance_min, Conductance_max,
                     batch_size=8, shuffle=True, num_workers=None, pin_memory=True, 
                     persistent_workers=True, drop_last=False):
    """
    创建支持CPU多进程并行加载的DataLoader
    
    Args:
        data_list: 训练/测试集的case_num列表
        dataset_dir: 数据集根目录
        Power_min, Power_max, ...: 归一化参数
        batch_size: 批次大小，默认8
        shuffle: 是否打乱数据，训练集True，测试集False
        num_workers: 并行进程数，None时自动设置为CPU核心数的1/2
        pin_memory: 是否使用锁页内存（加速CPU→GPU传输），默认True
        persistent_workers: 是否保持子进程常驻，默认True（提升多epoch训练效率）
        drop_last: 是否丢弃最后一个不完整批次，默认False
    
    Returns:
        DataLoader对象
    """
    import os
    import multiprocessing
    
    # 自动设置num_workers（CPU核心数的1/2，避免进程过多导致调度开销）
    if num_workers is None:
        cpu_count = os.cpu_count() or 4
        num_workers = max(1, cpu_count // 2)
        print(f"[DataLoader] Auto-set num_workers={num_workers} (CPU cores: {cpu_count})")
    
    # 创建Dataset
    dataset = ThermalGraphDataset(
        data_list=data_list,
        dataset_dir=dataset_dir,
        Power_min=Power_min,
        Power_max=Power_max,
        tPower_min=tPower_min,
        tPower_max=tPower_max,
        Temperature_min=Temperature_min,
        Temperature_max=Temperature_max,
        Conductance_min=Conductance_min,
        Conductance_max=Conductance_max
    )
    
    # 创建DataLoader（配置多进程并行加载）
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,  # CPU多进程并行数
        pin_memory=pin_memory,  # 锁页内存，加速CPU→GPU传输
        persistent_workers=persistent_workers if num_workers > 0 else False,  # 保持子进程常驻
        drop_last=drop_last,  # 是否丢弃最后一个不完整批次
        collate_fn=collate_fn,  # 自定义批次拼接函数
        prefetch_factor=2 if num_workers > 0 else None  # 每个worker预取2个batch
    )
    
    print(f"[DataLoader] Created with batch_size={batch_size}, num_workers={num_workers}, "
          f"pin_memory={pin_memory}, persistent_workers={persistent_workers}, "
          f"shuffle={shuffle}, drop_last={drop_last}")
    
    return dataloader