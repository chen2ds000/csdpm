import os
import pathlib
from operator import truediv

import torch
from torch.utils.data import random_split
import torch_geometric.utils
from torch_geometric.data import InMemoryDataset, download_url

from dataset.abstract_dataset1 import AbstractDataModule, AbstractDatasetInfos


class SpectreGraphDataset(InMemoryDataset):
    def __init__(self, dataset_name, split, root, transform=None, pre_transform=None, pre_filter=None):
        self.math1 = 'math1.pt'
        self.math2 = 'math2.pt'
        self.frcsub = 'frcsub.pt'
        self.dataset_name = dataset_name
        self.split = split
        self.num_graphs = 4209*19
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data ,self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    @property
    def processed_file_names(self):
            return [self.split + '.pt']

    def download(self):

        adjs = torch.load('/root/GDPO-main/dataProsses/math1Perp.pt')
        g_cpu = torch.Generator()
        g_cpu.manual_seed(0)

        test_len = int(round(self.num_graphs * 0.2))
        train_len = int(round((self.num_graphs - test_len) * 0.8))
        val_len = self.num_graphs - train_len - test_len
        indices = torch.randperm(self.num_graphs, generator=g_cpu)
        print(f'Dataset sizes: train {train_len}, val {val_len}, test {test_len}')
        train_indices = indices[:train_len]
        val_indices = indices[train_len:train_len + val_len]
        test_indices = indices[train_len + val_len:]

        train_data = []
        val_data = []
        test_data = []

        for i, adj in enumerate(adjs):
            if i in train_indices:
                train_data.append(adj)
            elif i in val_indices:
                val_data.append(adj)
            elif i in test_indices:
                test_data.append(adj)
            else:
                raise ValueError(f'Index {i} not in any split')

        torch.save(train_data, self.raw_paths[0])
        torch.save(val_data, self.raw_paths[1])
        torch.save(test_data, self.raw_paths[2])


    def process(self):
        #  pertrain
        file_idx = {'train': 0, 'val': 1, 'test': 2}
        raw_dataset = torch.load(self.raw_paths[file_idx[self.split]])

        data_list = []


        for adj, x, record, label in raw_dataset:
            record = torch.tensor(record, dtype=torch.float32)
            adj = torch.tensor(adj, dtype=torch.float32)
            n = 11
            X = torch.tensor(x, dtype=torch.float32)
            y = torch.zeros([1, 0]).float()
            edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)
            edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
            edge_attr[:, 1] = 1
            num_nodes = n * torch.ones(1, dtype=torch.long)
            data = torch_geometric.data.Data(x=X, edge_index=edge_index, edge_attr=edge_attr,
                                             y=y, n_nodes=num_nodes, record=record, label=label)
            data_list.append(data)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            # data_list.append(data)
        torch.save(self.collate(data_list), self.processed_paths[0])
        # csg ft
        # file_idx = {'train': 0, 'val': 1, 'test': 2}
        # raw_dataset = torch.load(self.raw_paths[file_idx[self.split]])
        #
        # data_list = []
        # for adj ,x ,record ,realRE in raw_dataset:
        #
        #     StudentRecord = torch.tensor(realRE[20], dtype=torch.float16)
        #     trueRecord = []
        #     for i in range(20):
        #         trueRecord.append(realRE[i])
        #     trueRE = (StudentRecord,trueRecord)
        #     record =  torch.tensor(record,dtype=torch.float16)
        #     adj = torch.tensor(adj, dtype=torch.float16)
        #     n = 11
        #     X = torch.tensor(x,dtype=torch.float16)
        #     y = torch.zeros([1, 0]).float()
        #     edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)
        #     edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float16)
        #     edge_attr[:, 1] = 1
        #     num_nodes = n * torch.ones(1, dtype=torch.long)
        #     data = torch_geometric.data.Data(x=X, edge_index=edge_index, edge_attr=edge_attr,
        #                                      y=y, n_nodes=num_nodes,record=record,trueRE=trueRE)
        #     data_list.append(data)
        #
        #     if self.pre_filter is not None and not self.pre_filter(data):
        #         continue
        #     if self.pre_transform is not None:
        #         data = self.pre_transform(data)
        #
        #     # data_list.append(data)
        # torch.save(self.collate(data_list), self.processed_paths[0])



class SpectreGraphDataModule(AbstractDataModule):
    def __init__(self, cfg, n_graphs=200):
        self.cfg = cfg
        self.datadir = cfg.dataset.datadir
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)


        datasets = {'train': SpectreGraphDataset(dataset_name=self.cfg.dataset.name,
                                                 split='train', root=root_path),
                    'val': SpectreGraphDataset(dataset_name=self.cfg.dataset.name,
                                        split='val', root=root_path),
                    'test': SpectreGraphDataset(dataset_name=self.cfg.dataset.name,
                                        split='test', root=root_path)}
        # print(f'Dataset sizes: train {train_len}, val {val_len}, test {test_len}')

        super().__init__(cfg, datasets)
        self.inner = self.train_dataset
        print(type(self.inner))

    def __getitem__(self, item):
        print("getitem",item)
        return self.inner[item]


class SpectreDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, dataset_config):
        self.datamodule = datamodule
        self.name = 'nx_graphs'
        self.n_nodes = self.datamodule.node_counts()
        self.node_types = self.datamodule.node_types()               # There are no node types
        self.edge_types = self.datamodule.edge_counts()
        super().complete_infos(self.n_nodes, self.node_types)

class ToyDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, dataset_config):
        self.datamodule = datamodule
        self.name = 'nx_graphs'
        max_nodes = dataset_config["nodes"]
        self.n_nodes = torch.ones((max_nodes+1))/(max_nodes-1)
        self.n_nodes[0] = 0
        self.n_nodes[1] = 0
        self.node_types = torch.tensor([1])               # There are no node types
        self.edge_types = torch.tensor([0.5,0.5])
        super().complete_infos(self.n_nodes, self.node_types)
