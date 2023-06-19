import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch_geometric.nn.models import GIN
from torch.utils.data import Dataset
import pickle
import pandas as pd
from graph import *
from torch import nn
from torch.utils.data import DataLoader, random_split

class GraphTriplesDataset(Dataset):
    def __init__(self, triple_file: str, graph_file: str) -> None:
        try:
            print('Loading triple file')
            self.triples = pd.read_csv(triple_file)
            print('Triple file loaded')
        except:
            raise Exception('Wrong triple file path')
        try:
            print('Loading graph file')
            with open(graph_file, 'rb') as f:
                self.graphs = pickle.load(f)
            print('Graph file loaded')
        except:
            raise Exception('Wrong graph file path')
    
    def __len__(self)->int:
        return len(self.triples)
    
    def __getitem__(self, idx:int)->tuple:
        t = self.triples.iloc[idx][:]
        return self.graphs[str(t[0])], self.graphs[str(t[1])], t[2]
    

class GNNTE(nn.Module):
    def __init__(self, hidden_channels:int, num_layers:int) -> None:
        super(GNNTE,self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = GIN(-1,hidden_channels,num_layers).to(self.device)

    def forward(self, g1: Graph, g2: Graph) -> torch.tensor:

def training_pipeline(triple_file: str, graph_file: str, ttv_ratio: set=(0.8,0.1,0.1) ) -> GNNTE:
    # Creazione 3 datasets
    data = GraphTriplesDataset(triple_file, graph_file)
    size = len(data)
    ttv = [size*ttv_ratio[0], size*ttv_ratio[1], size*ttv_ratio[2]]
    train, valid, test = random_split(train, ttv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Creazione modello
    # model to device
    # train(model, train_data, val_data, batch_size, lr, num_epochs, device)


>> train
def train(model, train_data, val_data, batch_size, lr, num_epochs, device)
# train, valid dataloader
# optimizer
# loss

best_loss = float('inf')
for epoch in range(num_epochs):

    # Train step
    train_loss = train_epoch(model, train_dataloader, optimizer, criterion, device))

    # Eval step
    val_loss = eval_epoch(model, valid_dataloader, criterion, device)


    if val_loss < best_loss:
        best_loss = val_loss
        # save_model_checkpoint
                

>> train_epoch
def train_epoch(model, train_dataloader, optimizer, criterion, device):
	total_loss = 0

	model.train()

	# For each batch of training data...
	for step, batch in enumerate(iterator):

	    # to gpu

	    # Forward
	    optimizer.zero_grad()
	    outputs = model(...)

	    # Loss
	    loss = criterion(outputs, y)
	    total_loss += loss.item()

	    # Perform a backward pass to calculate gradients
	    loss.backward()


	    # Update parameters and the learning rate
	    optimizer.step()

	# Calculate the average loss over the entire training data
	avg_train_loss = total_loss / len(iterator)

	return avg_train_loss


>> eval_epoch
def eval_epoch(model, valid_dataloader, criterion, device):
    avg_loss = 0.0
    model.eval()

    with torch.no_grad():
        for j, batch in enumerate(iterator):
            # to device

            outputs = model(...)
            loss = criterion(outputs, y)

            avg_loss += loss.item()

    avg_loss = avg_loss / len(iterator)

    return avg_loss


#_____________TEST____________________________
main
    eval_loader = DataLoader(eval_data, batch_size=args.batch_size,  num_workers=4, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    # to device

    y_pred, y_true = model_inference(
        model=model,
        iterator=eval_loader,
        device=device
    )


>> model_inference
def model_inference(model, data_loader, device):
    model.eval()

    y_pred = None
    y_true = None
    with torch.no_grad():
        for j, batch in enumerate(data_loader):
            # to device

            logits = model(x) #predictions

            # Save the predictions and the labels
            if y_pred is None:
                y_pred = logits  # FIXME: convert to predictions
                y_true = y
            else:
                y_pred = torch.cat((y_pred, logits))
                y_true = torch.cat((y_true, y))

    return y_pred, y_true





#____________________________________________________________________________________________________________________
def train_model(model: GNNTE, epoches: int=100,lr: float=0.005, weight_decay: float=5e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(1, 201):
        loss = train()
        acc = test()
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Acc: {acc:.4f}')

def train(model:GNNTE, optimizer:torch.optimizer, loader:DataLoader, device:str)->float:
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in tqdm(loader):
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def test(model:GNNTE):
    model.eval()
    z = model()
    acc = model.test(z[data.train_mask], data.y[data.train_mask],
                     z[data.test_mask], data.y[data.test_mask],
                     max_iter=150)
    return acc



if __name__ == "__main__":
    pass