import logging
import numpy as np
from tqdm import tqdm
import torch
import ast
import copy
import os
import csv

from utils import (
    build_args,
    create_optimizer,
    set_random_seed,
    TBLogger,
    get_current_lr,
    load_best_configs,
)
from datasets.data_proc import load_small_dataset
from models.finetune import linear_probing_full_batch
from models import build_model
import dgl
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from dgl.data import DGLDataset
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


def pretrain(model, graph, feat, optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob, logger=None):
    logging.info("start training..")
    graph = graph.to(device)
    x = feat.to(device)

    target_nodes = torch.arange(x.shape[0], device=x.device, dtype=torch.long)

    dim_graph = graph.num_nodes(), graph.num_edges()
    dim_target = target_nodes.size()

    epoch_iter = tqdm(range(max_epoch))

    for epoch in epoch_iter:
        model.train()

        loss = model(graph, x, targets=target_nodes)

        loss_dict = {"loss": loss.item()}
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        epoch_iter.set_description(f"# Epoch {epoch}: train_loss: {loss.item():.4f}")
        if logger is not None:
            loss_dict["lr"] = get_current_lr(optimizer)
            logger.note(loss_dict, step=epoch)

        if (epoch + 1) % 3340 == 0: #def tua 400
            linear_probing_full_batch(model, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob, mute=True)

    return model


def calculate_AP_at_k(row, k):
    # Create a temporary DataFrame
    temp_df = pd.DataFrame({
        'label': row['label'],
        'predizioni': row['predizioni']
    })

    # Sort the DataFrame by score in descending order
    temp_df = temp_df.sort_values(by='predizioni', ascending=False)

    # Take the top-k
    top_k = temp_df.head(k)

    # Initialize variables
    hits = 0
    sum_precs = 0

    # Calculate the sum of precisions at each new true positive
    for i, (_, row) in enumerate(top_k.iterrows(), start=1):
        if row['label'] == 1:
            hits += 1
            sum_precs += hits / i

    # Calculate AP@k
    if hits > 0:
        AP_at_k = sum_precs / hits
    else:
        AP_at_k = 0

    return AP_at_k


def accuracy2(y_pred, y_true, k ):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    topk_indices = y_pred.argsort()[-k:]
    y_pred_labels = np.zeros_like(y_pred)
    y_pred_labels[topk_indices] = 1
    correct = (y_pred_labels == y_true).astype(float)
    accuracy = correct.sum() / correct.size
    return accuracy, y_pred_labels.tolist()

def main(args):
    device = args.device if args.device >= 0 else "cpu"
    seeds = args.seeds
    dataset_name = args.dataset
    max_epoch = args.max_epoch
    max_epoch_f = args.max_epoch_f
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    encoder_type = args.encoder
    decoder_type = args.decoder
    replace_rate = args.replace_rate

    optim_type = args.optimizer 
    loss_fn = args.loss_fn

    lr = args.lr
    weight_decay = args.weight_decay
    lr_f = args.lr_f
    weight_decay_f = args.weight_decay_f
    linear_prob = args.linear_prob
    load_model = args.load_model
    logs = args.logging
    use_scheduler = args.scheduler

    graph, (num_features, num_classes) = load_small_dataset(dataset_name)
    args.num_features = num_features

    acc_list = []
    estp_acc_list = []
    best_acc = -1
    best_loss = float('inf')
    best_result = None
    
    for i, seed in enumerate(seeds):
        print(f"####### Run {i} for seed {seed}")
        set_random_seed(seed)

        if logs:
            logger = TBLogger(name=f"{dataset_name}_loss_{loss_fn}_rpr_{replace_rate}_nh_{num_hidden}_nl_{num_layers}_lr_{lr}_mp_{max_epoch}_mpf_{max_epoch_f}_wd_{weight_decay}_wdf_{weight_decay_f}_{encoder_type}_{decoder_type}")
        else:
            logger = None

        model = build_model(args)
        model.to(device)
        optimizer = create_optimizer(optim_type, model, lr, weight_decay)

        if use_scheduler:
            logging.info("Use schedular")
            scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / max_epoch) ) * 0.5
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
        else:
            scheduler = None
            
        x = graph.ndata["feat"]
        if not load_model:
            model = pretrain(model, graph, x, optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob, logger)
            model = model.cpu()

        if load_model:
            logging.info("Loading Model ... ")
            model.load_state_dict(torch.load("checkpoint.pt"))
        
        model = model.to(device)
        model.eval()

        final_acc, estp_acc, result_prova, loss = linear_probing_full_batch(model, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob)
        current_loss = loss

        
        # Modify the code to save the best values based on loss
        acc_list.append((final_acc, estp_acc))
        estp_acc_list.append(estp_acc)
        
        if current_loss < best_loss:
            best_loss = current_loss
            best_result = result_prova

            # Save best_result as a CSV file
            output_dir = "/content/drive/MyDrive/Tesi_LossAcc/"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_file2 = os.path.join(output_dir, "best_result.csv")
            best_result.to_csv(output_file2, index=False)  

        if logger is not None:
            logger.finish()


    
    # Calculate the mean and standard deviation for the final accuracy value
    final_acc, final_acc_std = np.mean([t[0] for t in acc_list]), np.std([t[0] for t in acc_list])
    # Calculate the mean and standard deviation for the early stopping accuracy value
    estp_acc, estp_acc_std = np.mean([t[1] for t in acc_list]), np.std([t[1] for t in acc_list])

    # Define file paths for data
    edge_path = "/content/drive/MyDrive/Tesi_LossAcc/EdgeGraph_Clustered2685.csv" #def: EdgeGraph_Clustered
    feature_path = "/content/drive/MyDrive/Tesi_LossAcc/FeatLessPat.csv" #def: FeatLessPat
    labels_path = "/content/drive/MyDrive/Tesi_LossAcc/LabelsLess1364.csv" #def:LabelsLess1364

    # Load datasets
    edges = pd.read_csv(edge_path, index_col = 0)
    features = pd.read_csv(feature_path, index_col=0)
    labels_df = pd.read_csv(labels_path, index_col=0)

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

    
    # Get node indices in the feature dataframe
    feature_node_idxs = features['source'].map(node_id_to_idx)
    # Get node indices in the label dataframe
    labels_node_idxs = labels_df['source'].map(node_id_to_idx)

    
    # Filter features based on node indices
    features_2 = features.iloc[feature_node_idxs, 1:2638]  # Select feature columns
    labels_df = labels_df.drop('source', axis=1)

    source_idxs = edges['source'].map(node_id_to_idx)
    target_idxs = edges['target'].map(node_id_to_idx)

    # Create a DGL graph
    g = dgl.graph((source_idxs, target_idxs))


    # Add labels to the DGL graph
    g.ndata['label'] = torch.tensor(labels_df.values, dtype=torch.float32)

    # Add node features to the graph
    node_features = torch.tensor(features_2.values, dtype=torch.float32)
    g.ndata['feat'] = node_features

    # Get node features and labels from the graph
    node_features = g.ndata['feat']
    node_labels = g.ndata['label']

    # Add node identifiers to the graph
    node_ids_tensor = torch.tensor(node_ids.astype(np.int64), dtype=torch.long)
    g.ndata['id'] = node_ids_tensor

    # Create a DataFrame for nodes and labels
    node_labels_df = pd.DataFrame({
        'node_id': node_ids,
        'label': node_labels.tolist()
    })

    # Convert values within the "label" vector to integers
    node_labels_df['label'] = node_labels_df['label'].apply(lambda x: np.array(x).astype(int))

    # Load the predictions dataframe
    df = pd.read_csv("/content/drive/MyDrive/Tesi_LossAcc/best_result.csv")

    # Create a vector
    # Copy the df DataFrame to keep the original
    df_modified = df.copy()

    # Iterate over each row in the DataFrame
    for index, row in df_modified.iterrows():
        # Get the predictions string for the current row
        pred_str = row['predizioni']
    
        # Remove "[" and "]" characters from the string
        pred_str = pred_str.strip("[]")
    
        # Split the string into a list of values
        pred_list = pred_str.split(',')
    
        # Convert values from strings to floats
        pred_float = [float(val) for val in pred_list]
    
        # Assign the modified float list to the 'predictions' column in the current row
        df_modified.at[index, 'predizioni'] = pred_float

    # Choose the value of k
    k = 10

    # Merge the DataFrames using Node ID as the merge key
    merged_df_metrics = df_modified.merge(node_labels_df, left_on='Node ID', right_on='node_id')

    # Remove the duplicate "node_id" column
    merged_df_metrics = merged_df_metrics.drop('node_id', axis=1)

    # Apply the accuracy function to each row of DataFrame
    merged_df_metrics['new_pred'] = merged_df_metrics.apply(lambda row: accuracy2(row['predizioni'], row['label'], k)[1], axis=1)

    # Add columns for metric calculation
    merged_df = merged_df_metrics.copy()
    merged_df['predicted_vector_ones'] = merged_df['new_pred'].apply(lambda vec: sum(x == 1 for x in vec))
    merged_df['label_ones'] = merged_df['label'].apply(lambda vec: sum(x == 1 for x in vec))
    merged_df['common_ones'] = merged_df.apply(lambda row: sum(x == 1 for x, y in zip(row['new_pred'], row['label']) if y == 1), axis=1)

    # Create the "ranked" column
    merged_df['ranked'] = merged_df['predizioni'].apply(lambda x: np.argsort(x)[::-1].argsort() + 1)
    
    #MRR
    def calculate_reciprocal_rank(node_id):
        label_vector = merged_df.loc[merged_df['Node ID'] == node_id, 'label'].values[0]
        ranked_vector = merged_df.loc[merged_df['Node ID'] == node_id, 'ranked'].values[0]

        positions = [i for i, val in enumerate(label_vector) if val == 1]
        values = [ranked_vector[i] for i in positions]

        reciprocal_values = [1/val for val in values if val != 0]
        reciprocal_rank = sum(reciprocal_values)

        return reciprocal_rank
    merged_df['reciprocal_rank'] = merged_df['Node ID'].apply(calculate_reciprocal_rank)
    num_camp = merged_df['label_ones'].sum() 
    print("Numero campioni corretti:", num_camp)
    reciprocalN_True = 1/num_camp
    reciprocalRank = merged_df['reciprocal_rank'].sum()
    MRR = reciprocalN_True*reciprocalRank
    print("La metrica Mean Reciprocal Rank restituisce valori pari a:", MRR)

    #TPR
    num_camp = merged_df['Node ID'].nunique()
    print("Numero campioni:", num_camp)
    reciprocalN = 1/num_camp
    merged_df["True_positive"] = merged_df["common_ones"].apply(lambda x: 1 if x > 0 else 0)
    positive_sum = merged_df['True_positive'].sum()
    print("Numero di campioni positivi:", positive_sum)
    True_PRate = reciprocalN*positive_sum
    print("La metrica True Positive Rate restituisce valore pari a:", True_PRate)
    
    #Hits
    num_veri_positivi = merged_df['common_ones'].sum()
    print("Numero di predizioni corrette:", num_veri_positivi)
    hits2 = num_veri_positivi/k
    print("Rapporto tra positivi e k:", hits2)
    Hits = reciprocalN * hits2
    print("La metrica Hits restituisce valore pari a:", Hits)

    #Mean Recall
    merged_df["False_Negative"] = merged_df['label_ones'] - merged_df["common_ones"]
    merged_df["Recall"] = merged_df['common_ones']/(merged_df['common_ones']+merged_df['False_Negative'])
    Sum_Recall = merged_df['Recall'].sum()
    MR = reciprocalN*Sum_Recall
    print("La metrica Mean Recall restituisce valore pari a:", MR)

    #MAP
    # Calculate AP for each row and add it as a new column
    merged_df['AP'] = merged_df.apply(calculate_AP_at_k, k=k, axis=1)
    # Calculate MAP
    N = len(merged_df) # Number of samples in the dataset
    MAP = merged_df['AP'].sum() / N
    print('La metrica Mean Average Precision restituisce valore pari a:', MAP)

    # Save metrics for k
    # Create the output file name
    filename_k = '/content/drive/MyDrive/Tesi_LossAcc/metric_forK=[' + str(k) + '].csv'

    # Open the file in write mode
    with open(filename_k, mode='w') as file:

        # Create the CSV writer
        writer = csv.writer(file, delimiter=',')
        writer.writerow(['k', 'MRR', 'TPR', 'hits', 'MRec', 'MAP'])
        writer.writerow([k, MRR, True_PRate, Hits, MR,MAP])
    
    print(f"# final_acc: {final_acc:.4f}±{final_acc_std:.4f}")
    print(f"# early-stopping_acc: {estp_acc:.4f}±{estp_acc_std:.4f}")


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    args = build_args()
    if args.use_cfg:
        args = load_best_configs(args)
    print(args)
    main(args)
