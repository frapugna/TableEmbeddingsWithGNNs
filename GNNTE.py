from collections.abc import Sequence
import torch
from tqdm import tqdm
from torch_geometric.nn.models import GIN
import pickle
import pandas as pd
from graph import *
from torch import nn
from sklearn.model_selection import train_test_split
from typing import Optional
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
from torch_geometric.data import Batch
import torch.nn.functional as F
import numpy as np
import random
import time
import torch.optim.lr_scheduler as lr_scheduler

def set_seed(seed) -> None:
    """ Set random seeds. """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class GraphTriplesDataset(Dataset):
    def __init__(self, triples: pd.DataFrame, graphs: dict, tables: Optional[dict] = None) -> None:
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
    
    
def train_test_valid_split(df: pd.DataFrame, ttv_ratio: set=(0.8,0.1,0.1)) -> None:
    train_data, test_valid_data = train_test_split(df, test_size=ttv_ratio[1]+ttv_ratio[2], random_state=42)
    test_data, valid_data = train_test_split(   test_valid_data, 
                                                test_size=ttv_ratio[2]/(ttv_ratio[1]+ttv_ratio[2]), 
                                                random_state=42)
    return train_data, test_data, valid_data
    

class GNNTE(nn.Module):
    def __init__(self, hidden_channels:int, num_layers:int, dropout: float=0) -> None:
        """_summary_

        Args:
            hidden_channels (int): size of the generated embeddings
            num_layers (int): number of layers of the network, every embedding will be generated using using his neighbours at distance num_layers
        """
        super(GNNTE,self).__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = GIN(-1,hidden_channels,num_layers, dropout=dropout).to(self.device)

    def forward(self, b: Batch) -> torch.tensor:
        #calcolo lista lunghezze
        intervals = [0]
        for i in range(b.num_graphs):
            intervals.append(b[i]['x'].shape[0]+intervals[-1])
        #gin
        out_gin = self.model(b['x'], b['edge_index'])
        #medie
        means = [torch.mean(out_gin[intervals[i]:intervals[i+1]][:], dim=0).unsqueeze(dim=0) for i in range(b.num_graphs)]
        return torch.cat(means, dim=0)

def load_model(model_file: str, hidden_channels: int, num_layers: int, dropout: int) -> GNNTE:
    model = GNNTE(hidden_channels, num_layers, dropout=dropout)
    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict['model_state_dict'])
    return model

def load_test_training_stuff(triple_file: str, graph_file: str) -> dict:
    with open(graph_file, 'rb') as f:
        gd = pickle.load(f)
    triples = pd.read_csv(triple_file)
    return {'graphs':gd, 'triples':triples}

def train_test_pipeline(triple_file: str, graph_file: str, model_file: str, hidden_channels: int, num_layers: int,
                        ttv_ratio: set=(0.8,0.1,0.1), batch_size: int=64, lr: float=0.01, dropout: float=0, 
                        num_epochs: int=100, weight_decay=0) -> GNNTE:
    set_seed(42)
    # Creazione 3 datasets
    print('Loading datasets, it could take some time....')
    all_data = load_test_training_stuff(triple_file, graph_file)

    tables = train_test_valid_split(all_data['triples'], ttv_ratio)
    train_dataset = GraphTriplesDataset(tables[0], all_data['graphs'])
    test_dataset = GraphTriplesDataset(tables[1], all_data['graphs'])
    valid_dataset = GraphTriplesDataset(tables[2], all_data['graphs'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = GNNTE(hidden_channels, num_layers, dropout=dropout)
    start = time.time()
    model = train(model, train_dataset, valid_dataset, batch_size, lr, num_epochs, device, model_file, weight_decay=weight_decay)
    end = time.time()
    print(f'T_train: {end-start}ms')

    start = time.time()
    mse = test(model, test_dataset, batch_size)
    end = time.time()
    print(f'T_test: {end-start}ms')

    return model, mse

def train(model, train_dataset, valid_dataset, batch_size, lr, num_epochs, device, model_file: str, 
          shuffle: bool=False, num_workers: int=0, weight_decay: float=5e-4) -> GNNTE:
    # train, valid dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # scheduler
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # loss
    loss_criterion = nn.MSELoss()

    best_loss = float('inf')
    for epoch in range(num_epochs):

        # Train step
        train_loss = train_epoch(model, train_dataloader, optimizer, loss_criterion, device)

        # Eval step
        
        val_loss = eval_epoch(model, valid_dataloader, loss_criterion, device)

        print(f'Epoch:{epoch}, Train loss: {train_loss}, Valid loss: {val_loss}')

        if val_loss < best_loss:
            best_loss = val_loss
            # save_model_checkpoint
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch
            }
            torch.save(checkpoint, model_file)
            # print('Checkpoint updated!')
        

    return load_model(model_file, model.hidden_channels, model.num_layers, model.dropout)

def train_epoch(model: GNNTE, train_dataloader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.MSELoss, device: str) -> torch.Tensor:
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
        #scheduler.step()

    # # Calculate the average loss over the entire training data
    avg_train_loss = total_loss / len(train_dataloader)

    return avg_train_loss

def eval_epoch(model, valid_dataloader, criterion, device):
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
    print(f'MSE: {mse}')
    return mse

def model_inference(model, data_loader, device):
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




if __name__ == "__main__":
    dir = "/home/francesco.pugnaloni/wikipedia_tables/small_tables/10000_samples"
    triples_path = dir+"/samples.csv"
    graphs_path = dir+"/graphs.pkl"


    train_test_pipeline(triples_path, graphs_path,
                            "/dati/home/francesco.pugnaloni/tmp/checkpoint.pth",
                            300, 3, num_epochs=30, batch_size=256, lr=0.001, dropout=0.2
                            )
    print('ok')