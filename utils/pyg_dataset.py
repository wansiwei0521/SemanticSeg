"""
 # Author: swan && swan_and_vansw@126.com
 # Date: 2025-02-15 11:22:49
 # LastEditors: swan && swan_and_vansw@126.com
 # LastEditTime: 2025-02-15 14:15:46
 # FilePath: //STNet//utils//pyg_dataset.py
 # Description: 
"""
import os
import torch
from torch_geometric.data import InMemoryDataset, Data, Batch
import pickle
from collections import Counter
from torch.serialization import safe_globals

# 自定义 Data 对象，用于存储一个样本中多个图以及对应的标签
class MultiGraphData(Data):
    def __init__(self, graphs=None, y=None):
        super(MultiGraphData, self).__init__()
        self.graphs = graphs  
        self.y = y

class STGraphDataset(InMemoryDataset):
    def __init__(self, root, pkl_file_path, transform=None, pre_transform=None, pre_filter=None):
        self.pkl_file_path = pkl_file_path
        super(STGraphDataset, self).__init__(root, transform, pre_transform, pre_filter)
        # 使用 safe_globals 上下文管理器来允许加载 MultiGraphData
        with safe_globals([MultiGraphData, STGraphDataset]):
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_file_names(self):
        pkl_name = os.path.basename(self.pkl_file_path)
        name_without_ext = os.path.splitext(pkl_name)[0]
        # 如果文件名以"_dataset_aug_cache"结尾，则移除该后缀
        suffix = "_dataset_aug_cache"
        if name_without_ext.endswith(suffix):
            name_without_ext = name_without_ext[:-len(suffix)]
        return [f"processed_data_{name_without_ext}.pt"]

    def process(self):
        # 从 pkl 文件中加载原始数据
        with open(self.pkl_file_path, 'rb') as f:
            raw_data = pickle.load(f)  # 期望格式为 [(window_graphs, window_label), ...]
        
        processed_list = []
        # 对于每个样本，构造一个 MultiGraphData 对象
        for window_graphs, window_label in raw_data:
            multi_graph_data = MultiGraphData(
                graphs=window_graphs,
                y=torch.tensor(window_label, dtype=torch.long)
            )
            processed_list.append(multi_graph_data)
        
        data, slices = self.collate(processed_list)
        torch.save((data, slices), self.processed_paths[0])

    def get(self, idx):
        data = super(STGraphDataset, self).get(idx)
        batch_graphs = Batch.from_data_list(data.graphs)
        return batch_graphs, data.y

    def compute_class_weights(self):
        labels = [int(self.get(i)[1].item()) for i in range(len(self))]
        label_counter = Counter(labels)
        print("每个类的数量:", dict(label_counter))
        total_samples = len(self)
        num_classes = len(label_counter)
        weights = {label: total_samples / (num_classes * count) for label, count in label_counter.items()}
        weight_list = [weights[i] for i in range(num_classes)]
        return torch.tensor(weight_list, dtype=torch.float)

if __name__ == '__main__':
    work_dir = r'D:/OneDrive - chd.edu.cn/Desktop/毕业论文数据/code/STNet'
    dataset = STGraphDataset(
        root=os.path.join(work_dir, 'dataset', 'pyg_cache'),
        pkl_file_path=os.path.join(work_dir, 'dataset', 'cache', 'motor_30_1_7_dataset_aug_cache.pkl')
    )
    
    print("数据集长度:", len(dataset))
    print("类别权重:", dataset.compute_class_weights())
    batch_graphs, label = dataset[0]
    print("第一个样本的标签:", label)
    print("第一个样本中包含的子图数量:", batch_graphs.num_graphs)


