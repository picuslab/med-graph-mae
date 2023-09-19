import dgl
import torch
import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from dgl.data import DGLDataset
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression


class MyDataset(DGLDataset):
    def __init__(self, edge_path, feature_path, labels_path):
        self.edge_path = edge_path
        self.feature_path = feature_path
        self.num_classes = None
        self.labels_path = labels_path
        super().__init__(name="my_dataset")

    def process(self):
        # Load datasets
        edges = pd.read_csv(self.edge_path, index_col = 0)
        features = pd.read_csv(self.feature_path, index_col=0)
        labels_df = pd.read_csv(self.labels_path, index_col=0)

        # Reset edge index
        edges.reset_index(drop=True, inplace=True)

        # Use the smallest possible data type for dataframe columns
        edges['source'] = edges['source'].astype('uint32')
        edges['target'] = edges['target'].astype('uint32')
        labels_df['source'] = labels_df['source'].astype('uint32')

        # Find unique values in the "source" column
        unique_sources = set(edges['source']).union(set(edges['target'])) #unique()
        
        # Filter rows by source
        features = features[features['source'].isin(unique_sources)]
        labels_df = labels_df[labels_df['source'].isin(unique_sources)]

        # Create a dictionary to map node identifiers to their unique indices
        node_ids = np.unique(np.concatenate((edges['source'], edges['target'])))
        node_id_to_idx = {nid: i for i, nid in enumerate(features['source'])}

        # Get node indices 
        feature_node_idxs = features['source'].map(node_id_to_idx)
        labels_node_idxs = labels_df['source'].map(node_id_to_idx)


        features_2 = features.iloc[feature_node_idxs, 1:2638] # Select feature columns 
        labels_df = labels_df.drop('source', axis=1)

        source_idxs = edges['source'].map(node_id_to_idx)
        target_idxs = edges['target'].map(node_id_to_idx)

        # Create a DGL graph
        g = dgl.graph((source_idxs, target_idxs))

        
        # Add labels and features to the DGL graph
        g.ndata['label'] = torch.tensor(labels_df.values, dtype=torch.float32)        
        node_features = torch.tensor(features_2.values, dtype=torch.float32)
        g.ndata['feat'] = node_features

        # Get node features and labels from the graph
        node_features = g.ndata['feat']
        node_labels = g.ndata['label']

        # Add node identifiers to the graph
        node_ids_tensor = torch.tensor(node_ids.astype(np.int64), dtype=torch.long)
        g.ndata['id'] = node_ids_tensor

        # Creation of training, validation, and test masks
        num_nodes = len(node_ids)
        train_mask, val_mask, test_mask = np.zeros(num_nodes), np.zeros(num_nodes), np.zeros(num_nodes)

        
        #test_size = 1 / num_nodes #Only for one node

        train_idx, test_idx = train_test_split(np.arange(num_nodes), test_size=0.05, random_state=42) #def:test_size = 0.05
        train_idx, val_idx = train_test_split(train_idx, test_size=0.10, random_state=42) #def: test_size = 0.10

        train_mask[train_idx] = 1
        val_mask[val_idx] = 1
        test_mask[test_idx] = 1


        #######TEST###########
        """test_node_idx = 9076

        test_mask[test_node_idx] = 1
        val_node_idx = 0
        val_mask[val_node_idx] = 1
    
        train_mask[:test_node_idx] = 1
        train_mask[test_node_idx + 1:] = 1
        train_mask[val_node_idx] = 0
        test_nodes = g.ndata['test_mask'].numpy()
        test_node_indices = np.where(test_nodes == 1)[0]
        filename_test = '/content/drive/MyDrive/Tesi_LossAcc/indice_nodo_test.csv'
        with open(filename_test, mode='w') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow(['l indice del nodo scelto risulta essere'])
            writer.writerow([test_node_indices])"""

        ######################

        g.ndata['train_mask'] = torch.BoolTensor(train_mask)
        g.ndata['val_mask'] = torch.BoolTensor(val_mask)
        g.ndata['test_mask'] = torch.BoolTensor(test_mask)

        self.graph = g
        self.num_classes = labels_df.shape[1]
        self.num_feats = features.shape[1]
    
    def get_label_mapping(self):
        return self.label_mapping

    def __getitem__(self, idx):
        return self.graph

    def __len__(self):
        return 1
    def print_info(self):
        num_nodes = self.graph.num_nodes()
        num_edges = self.graph.num_edges()
        num_training_samples = int(self.graph.ndata['train_mask'].sum())
        num_validation_samples = int(self.graph.ndata['val_mask'].sum())
        num_test_samples = int(self.graph.ndata['test_mask'].sum())

        print(f"NumNodes: {num_nodes}")
        print(f"NumEdges: {num_edges}")
        print(f"NumFeats: {self.num_feats}")
        print(f"NumClasses: {self.num_classes}")
        print(f"NumTrainingSamples: {num_training_samples}")
        print(f"NumValidationSamples: {num_validation_samples}")
        print(f"NumTestSamples: {num_test_samples}")
        print("Done loading data from cached files.")
