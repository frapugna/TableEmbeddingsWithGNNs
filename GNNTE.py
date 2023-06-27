import torch
from tqdm import tqdm
from torch_geometric.nn.models import GIN
import pickle
import pandas as pd
from graph import *
from torch import nn
from sklearn.model_selection import train_test_split
from typing import Optional
from torch_geometric.data import Data, Dataset, Batch
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import numpy as np
import random
import time
import torch.optim.lr_scheduler as lr_scheduler
import wandb
import warnings
from my_constants import *

def set_seed() -> None:
    """This functions set the seeds for various libraries

    Args:
        seed (int): this number will be used as seed
    """
    seed = SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class GraphTriplesDataset(Dataset):
    
    def __init__(self, triples: pd.DataFrame, graphs: dict, tables: Optional[dict] = None) -> None:
        """Init function

        Args:
            triples (pd.DataFrame): Dataframe that contains triples ()'r_id','s_id','table_overlap')
            graphs (dict): a dictionary containing a graph for every key that appears in the triples dataset
            tables (Optional[dict], optional): not implemented. Defaults to None.
        """
        super(GraphTriplesDataset, self).__init__()
        self.triples = triples
        self.graphs = graphs

    def len(self) -> int:
        return len(self.triples)
    
    def get(self, idx:int) -> tuple:
        t = self.triples.iloc[idx][:]
        try:
            g1 = self.graphs[str(t[0])]
            g2 = self.graphs[str(t[1])]
        except:
            g1 = self.graphs[str(int(t[0]))]
            g2 = self.graphs[str(int(t[1]))]
        return Data(g1.X, g1.edges), Data(g2.X, g2.edges), t[2]
    
    
def train_test_valid_split(df: pd.DataFrame, ttv_ratio: set=(0.8,0.1,0.1)) -> set:
    """Given a dataframe it is divided in a train dataset, a test dataset, and a validation dataset

    Args:
        df (pd.DataFrame): the dataset to split
        ttv_ratio (set, optional): A triple that tells the function how to split the dataset (TRAIN, TEST, VALIDATE). Defaults to (0.8,0.1,0.1).

    Returns:
        set: a set of 3 elements containing the triple split of the dataset in (TRAIN, TEST, VALIDATE)
    """
    train_data, test_valid_data = train_test_split(df, test_size=ttv_ratio[1]+ttv_ratio[2], random_state=SEED)
    test_data, valid_data = train_test_split(   test_valid_data, 
                                                test_size=ttv_ratio[2]/(ttv_ratio[1]+ttv_ratio[2]), 
                                                random_state=SEED)
    return train_data, test_data, valid_data
    

class GNNTE(nn.Module):
    def __init__(self, hidden_channels: int=300, num_layers: int=3, dropout: float=0, act: str="relu", model_file: str=None) -> None:
        """_summary_

        Args:
            hidden_channels (int): size of the generated embeddings
            num_layers (int): number of layers of the network, every embedding will be generated using using his neighbours at distance num_layers
            dropout (float, optional): dropout probability for the weights. Defaults to 0.
            act (str, optional): The activation function between the layers. Defaults to "relu".
            model_file (str, optional): this parameter can be used to load a pretrained model from a model_file
        """
        super(GNNTE,self).__init__()
        if model_file != None:
            state_dict = torch.load(model_file)
            self.hidden_channels = state_dict['hidden_channels']
            self.num_layers = state_dict['num_layers']
            self.dropout = state_dict['dropout']
            self.act = state_dict['act']
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = GIN(-1, self.hidden_channels, self.num_layers, dropout=dropout, act=act).to(self.device)
            self.load_state_dict(state_dict['model_state_dict'])
        else:
            self.hidden_channels = hidden_channels
            self.num_layers = num_layers
            self.dropout = dropout
            self.act = act
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = GIN(-1, hidden_channels, num_layers, dropout=dropout, act=act).to(self.device)

    def forward(self, b: Batch) -> torch.tensor:
        """Provided a batch of graphs as input their embeddings are provided in output

        Args:
            b (Batch): a group of graphs (X tensor and edgelist tensor)

        Returns:
            torch.tensor: a tensor composed by <num_graphs_int_the_batch> rows, every row will be the embedding of the corresponding graph
        """
        #calcolo lista lunghezze
        intervals = [0]
        for i in range(b.num_graphs):
            intervals.append(b[i]['x'].shape[0]+intervals[-1])
        #gin
        out_gin = self.model(b['x'], b['edge_index'])
        #medie
        means = [torch.mean(out_gin[intervals[i]:intervals[i+1]][:], dim=0).unsqueeze(dim=0) for i in range(b.num_graphs)]
        return torch.cat(means, dim=0)

def load_model(model_file: str, hidden_channels: int, num_layers: int, dropout: int, act: str) -> GNNTE:
    """____DEPRECATED____

    The model is loaded from a checkpoint file

    Args:
        model_file (str): path to the file containing the model's backup
        hidden_channels (int): size of the generated embeddings
        num_layers (int): the number of layers of the network
        dropout (int): dropout probability
        act (str): activation function

    Returns:
        GNNTE: an istance of the GNN generated from the backup file
    """
    warnings.warn("Deprecated function, now it is in the model's constructor", category=DeprecationWarning)
    model = GNNTE(hidden_channels, num_layers, dropout=dropout, act=act)
    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict['model_state_dict'])
    return model

def load_test_training_stuff(triple_file: str, graph_file: str) -> dict:
    """Utility function to load the training stuff

    Args:
        triple_file (str): path to the triple file
        graph_file (str): path to the graph file

    Returns:
        dict: it has 2 entries: 'graphs' and 'triples', the 2 loaded files
    """
    with open(graph_file, 'rb') as f:
        gd = pickle.load(f)
    triples = pd.read_csv(triple_file)
    return {'graphs':gd, 'triples':triples}

def train_test_pipeline(triple_file: str, graph_file: str, model_file: str, hidden_channels: int, num_layers: int,
                        ttv_ratio: set=(0.8,0.1,0.1), batch_size: int=64, lr: float=0.01, dropout: float=0, 
                        num_epochs: int=100, weight_decay: float=0, act: str='relu', log_wandb: bool=False,
                        step_size: int=5, gamma: float=0.1) -> GNNTE:
    """This function performs the full train-validate-test pipeline

    Args:
        triple_file (str): path to the triple file
        graph_file (str): path to the graph file
        model_file (str): path to the backup file for the model
        hidden_channels (int): size of the generated embeddings
        num_layers (int): number of layers of the network, every embedding will be generated using using his neighbours at distance num_layers
        ttv_ratio (set, optional): a triple that tells the function how to split the dataset (TRAIN, TEST, VALIDATE). Defaults to (0.8,0.1,0.1).
        batch_size (int, optional): number of elements to put in the training batches. Defaults to 64.
        lr (float, optional): learning rate. Defaults to 0.01.
        dropout (float, optional): dropout probability. Defaults to 0.
        num_epochs (int, optional): number of training epochs. Defaults to 100.
        weight_decay (float, optional): NA. Defaults to 0.
        act (str, optional): the activation function used between the layers. Defaults to 'relu'.
        log_wandb (bool, optional): if True all the outputs of the experiments will be logged to wandb, an open session is necessary to avoid errors. Defaults to False.
        step_size (int, optional): number of epochs to wait to update the learning rate. Defaults to 5.
        gamma (float, optional): reduction factor of the learning rate. Defaults to 0.1

    Returns:
        GNNTE: the trained network
    """
    set_seed()
    # Creazione 3 datasets
    print('Loading datasets, it could take some time....')
    all_data = load_test_training_stuff(triple_file, graph_file)

    tables = train_test_valid_split(all_data['triples'], ttv_ratio)
    train_dataset = GraphTriplesDataset(tables[0], all_data['graphs'])
    test_dataset = GraphTriplesDataset(tables[1], all_data['graphs'])
    valid_dataset = GraphTriplesDataset(tables[2], all_data['graphs'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = GNNTE(hidden_channels, num_layers, dropout=dropout, act=act)
    start = time.time()

    print('Training starts')

    model = train(model, train_dataset, valid_dataset, batch_size, lr, num_epochs, device, model_file, 
                  weight_decay=weight_decay, log_wandb=log_wandb, step_size=step_size, gamma=gamma)
    end = time.time()
    t_train=end-start
    print(f'T_train: {t_train}s')

    start = time.time()
    execution_insights = test(model, test_dataset, batch_size) 
    mse = execution_insights['mse'] 
    end = time.time()
    t_test = end-start
    print(f'T_test: {t_test}s')
    print(f'MSE: {mse}')
    if log_wandb:
        wandb.run.summary["T_train"] = t_train 
        wandb.run.summary["T_test"] = t_test
        wandb.run.summary["MSE"] = mse
        wandb.run.summary["insights"] = execution_insights

    execution_insights['model'] = model

    return execution_insights

def train(model, train_dataset, valid_dataset, batch_size, lr, num_epochs, device, model_file: str, 
          shuffle: bool=False, num_workers: int=0, weight_decay: float=5e-4, log_wandb: bool=False, step_size: int=5, gamma: float=0.1) -> GNNTE:
    """This function execute the training operation

    Args:
        model (_type_): the model to train
        train_dataset (_type_): training dataset generated with train_test_valid_split
        valid_dataset (_type_): validation dataset generated with train_test_valid_split
        batch_size (_type_): size of the batches
        lr (_type_): learning rate
        num_epochs (_type_): number of training epochs
        device (_type_): the device the model is working with
        model_file (str): path to the file where to save the model's checkpoints
        shuffle (bool, optional): if set True train and validation dataset are shuffled. Defaults to False.
        num_workers (int, optional): NA. Defaults to 0.
        weight_decay (float, optional): NA. Defaults to 5e-4.
        log_wandb (bool, optional): if True all the outputs of the experiments will be logged to wandb, an open session is necessary to avoid errors. Defaults to False.
        step_size (int, optional): number of epochs to wait to update the learning rate. Defaults to 5.
        gamma (float, optional): reduction factor of the learning rate. Defaults to 0.1

    Returns:
        GNNTE: the trained network
    """
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # scheduler
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # loss
    loss_criterion = nn.MSELoss()

    best_loss = float('inf')
    for epoch in tqdm(range(num_epochs)):
        start_epoch = time.time()
        # Train step
        train_loss = train_epoch(model, train_dataloader, optimizer, loss_criterion, device)

        # Eval step
        
        val_loss = eval_epoch(model, valid_dataloader, loss_criterion, device)

        end_epoch = time.time()

        if log_wandb:
            wandb.log({"t_exec_epoch": end_epoch-start_epoch, "train_loss":train_loss, "valid_loss":val_loss})
        if val_loss < best_loss:
            best_loss = val_loss
            # save_model_checkpoint
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'hidden_channels' : model.hidden_channels,
                'num_layers' : model.num_layers,
                'dropout' : model.dropout,
                'act' : model.act,
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch
            }
            torch.save(checkpoint, model_file)
        scheduler.step()

    return GNNTE(model_file=model_file)
    # return load_model(model_file, model.hidden_channels, model.num_layers, model.dropout, model.act)

def train_epoch(model: GNNTE, train_dataloader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.MSELoss, device: str) -> torch.Tensor:
    """Operations to perform in every training epoch

    Args:
        model (GNNTE): the model to train
        train_dataloader (DataLoader): dataloader containing the training data
        optimizer (torch.optim.Optimizer): optimizer
        criterion (nn.MSELoss): loss criterion
        device (str): the device the model is working with

    Returns:
        torch.Tensor: the average loss of the step
    """
    total_loss = 0
    model.train()
    
    # For each batch of training data...
    for batch in train_dataloader:
        # Forward
        optimizer.zero_grad()
        emb_l = model(batch[0].to(device))
        emb_r = model(batch[1].to(device))

        predictions = F.cosine_similarity(emb_l, emb_r, dim=1)

        y = batch[2].to(device)

        # Loss
        loss = criterion(predictions, y)
        total_loss += loss.item()

        # Perform a backward pass to calculate gradients
        loss.backward()

        # Update parameters and the learning rate
        optimizer.step()

    # # Calculate the average loss over the entire training data
    avg_train_loss = total_loss / len(train_dataloader)

    return avg_train_loss

def eval_epoch(model: GNNTE, valid_dataloader: DataLoader, criterion: nn.MSELoss, device: str) -> float:
    """Operations to perform evaluation in every epoch

    Args:
        model (GNNTE): the model to evaluate
        valid_dataloader (DataLoader): data loader containing the validation data
        criterion (nn.MSELoss): loss criterion
        device (str): the device the model is working with

    Returns:
        torch.Tensor: the average loss on the validation set 
    """
    avg_loss = 0.0
    model.eval()

    with torch.no_grad():
        for batch in valid_dataloader:
            # to device
            emb_l = model(batch[0].to(device))
            emb_r = model(batch[1].to(device))

            predictions = F.cosine_similarity(emb_l, emb_r, dim=1)

            y = batch[2].to(device)

            # Loss
            loss = criterion(predictions, y)

            avg_loss += loss.item()

    avg_loss = avg_loss / len(valid_dataloader)

    return avg_loss

def test(model: GNNTE, test_dataset: GraphTriplesDataset, batch_size: int=64, 
         num_workers: int=0, shuffle: bool=False) -> torch.Tensor:
    """Perform the testing operation

    Args:
        model (GNNTE): the model to test
        test_dataset (GraphTriplesDataset): testing dataset generated with train_test_valid_split
        batch_size (int, optional): size of the batches. Defaults to 64.
        num_workers (int, optional): NA. Defaults to 0.
        shuffle (bool, optional): if set True train and validation dataset are shuffled. Defaults to False.

    Returns:
        torch.Tensor: torch.Tensor: the average loss on the test set 
    """
    eval_loader = DataLoader(test_dataset, 
                             batch_size=batch_size,  
                             num_workers=num_workers, 
                             shuffle=shuffle)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    # to device

    y_pred, y_true = model_inference(
        model,
        eval_loader,
        device
    )
    mse = F.mse_loss(y_pred, y_true.to(device))
    abs_diff_tensor = torch.abs(y_pred - y_true.to(device))
    mae = torch.mean(abs_diff_tensor)
    variance = torch.var(abs_diff_tensor)
    min_ae = torch.min(abs_diff_tensor)
    max_ae = torch.max(abs_diff_tensor)

    return {'mse':mse, 'mae':mae, 'variance':variance, 'min_ae':min_ae, 'max_ae':max_ae, 'abs_diff_tensor':abs_diff_tensor}

def model_inference(model: GNNTE, data_loader: DataLoader, device: str) -> set:
    """Genertion of 2 sets of labels (predicted and real)

    Args:
        model (GNNTE): the model to test
        data_loader (DataLoader): dataloader containing the test data
        device (str): the device the model is working with

    Returns:
        set: a set of 2 tensors (PREDICTIONS, LABELS)
    """
    model.eval()

    y_pred = None
    y_true = None
    with torch.no_grad():
        for batch in data_loader:
            # to device
            emb_l = model(batch[0].to(device))
            emb_r = model(batch[1].to(device))

            logits = F.cosine_similarity(emb_l, emb_r, dim=1)
            y = batch[2]
                # Save the predictions and the labels
            if y_pred is None:
                y_pred = logits  
                y_true = batch[2]
            else:
                y_pred = torch.cat((y_pred, logits))
                y_true = torch.cat((y_true, y))

    return y_pred, y_true

def run_GNNTE_experiment(project_name: str, dataset: str, lr: float, batch_size: int,
                         num_epochs: int, out_channels: int, n_layers: int, dropout: float,
                         n_sample:str, weight_decay: float, step_size: int, gamma: float
                         ) -> None:
    """Utility function to run experiments that will be logged in wandb

    Args:
        project_name (str): name of the project in wandb
        dataset (str): directory containing the stuff necessary to build the dataset
        lr (float): learning rate
        batch_size (int): size of the training batches
        num_epochs (int): number of training epochs
        out_channels (int): size of the embeddings
        n_layers (int): number of layers
        dropout (float): dropout probability
        n_sample (str): identifier of the dataset
        weight_decay (float): an L2 penalty
        step_size (int, optional): number of epochs to wait to update the learning rate. Defaults to 5.
        gamma (float, optional): reduction factor of the learning rate. Defaults to 0.1

    """
    name = f"{n_sample}_{batch_size}_{lr}_{num_epochs}_{out_channels}_{n_layers}_{dropout}_{weight_decay}_{step_size}_{gamma}"
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project=project_name,
        name=name,
        # track hyperparameters and run metadata
        config={
            "dataset": dataset,
            "batch_size": batch_size,
            "learning_rate": lr,
            "num_epochs": num_epochs,
            "out_channels": out_channels,
            "n_layers": n_layers,
            "dropout": dropout,
            "step_size": step_size,
            "gamma": gamma
        }
    )
    triples_path = dataset+"/samples.csv"
    graphs_path = dataset+"/graphs.pkl"
    checkpoint = dataset+f"/{name}.pth"

    train_test_pipeline(triples_path, graphs_path, checkpoint, out_channels, n_layers, 
                        num_epochs=num_epochs, batch_size=batch_size, lr=lr, dropout=dropout, log_wandb=True,
                        weight_decay=weight_decay, step_size=step_size, gamma=gamma)
        
    wandb.finish()

if __name__ == "__main__":
    name = 'GNNTE'
    dataset = "/home/francesco.pugnaloni/wikipedia_tables/training_data/millions_of_tables"
    lr = 0.001
    batch_size = 128
    num_epochs = 50
    out_channels = 128
    n_layers = 3
    dropout = 0
    triples_path = dataset+"/samples.csv"
    graphs_path = dataset+"/graphs.pkl"
    checkpoint = dataset+f"/{name}.pth"

    train_test_pipeline(triples_path, graphs_path, checkpoint, out_channels, n_layers, 
        num_epochs=num_epochs, batch_size=batch_size, lr=lr, dropout=dropout, log_wandb=False)