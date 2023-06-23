from torch_geometric.nn.models import GIN
import torch
import torch.nn.functional as F
from graph import *
from node_embeddings import *
import time
class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, directory_name=False,save=False, load=False):
        if directory_name:
            try:
                self.load(directory_name)
            except:
                self.sample_list = []
                self.size = 0
                print('Empty sample_list created')
            self.graph_list = Graph_list(directory_name)
        else:
            self.sample_list = []
            self.size = 0
            self.graph_list = Graph_list()

    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        try:
            return (self.graph_list[self.sample_list[idx][0]], self.graph_list[self.sample_list[idx][1]], self.sample_list[idx][2])
        except IndexError:
            raise Exception("Index out of bound")
        
    def add_item(self, g):
        self.sample_list.append(g)
        self.size += 1
    
    def load(self, directory_name):
        try:
            f1 = open(directory_name+'/graph_dataset.pkl', 'rb')
            self.sample_list = pickle.load(f1)
            self.size = len(self.graph_list)
            f1.close()
        except:
            raise Exception('Read operation failed')
        self.size = len(self.graph_datasett)
    def save(self, directory_name):
        try:
            f1 = open(directory_name+'/graph_dataset.pkl', 'wb')
            pickle.dump(self.graph_dataset, f1)
            f1.close()
        except:
            raise Exception('Write operation failed')
        
def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:   
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def test(model, X_train, y_train, X_test, y_test):
    model.eval()
    z = model()
    acc = model.test(z[data.train_mask], data.y[data.train_mask],
                     z[data.test_mask], data.y[data.test_mask],
                     max_iter=150)
    return acc

class EmbeddingGenerator:
    def __init__(self, model_name='GIN', in_channels=-1, hidden_channels=128,num_layers=5, out_channels=None):
        self.model_name = model_name
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = GIN(in_channels=in_channels, hidden_channels=hidden_channels,num_layers=num_layers, out_channels=out_channels).to(device)
    
    # def __train__(self, n_epochs=100):
    #     optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005, weight_decay=5e-4)

    #     self.model.train()
    #     for epoch in range(2):
    #         self.model.train()
    #         optimizer.zero_grad()
    #         #out = self.model(data.x, data.edge_index)
    #         loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            
    #         if epoch%200 == 100:
    #             print(loss)
            
    #         loss.backward()
    #         optimizer.step()
        #_____________________________________________________________________________________________________
    #     start = time.time()

    #     device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #     t_start_epoch = 0    #debug
    #     loader = model.loader(batch_size=batch_size, shuffle=True, num_workers=num_workers) #Generate loader
    #     optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005, weight_decay=5e-4)
    #     print('Training is starting')
    #     for epoch in range(1, n_epochs):
    #         loss = train(self.model, loader, optimizer, device)
    #         acc = test()
    #     end = time.time()

    #     print(f'T_exec embedding generation: {end-start}s')

    # def __call__(self, X, edges):
    #     with torch.no_grad():
    #         return self.model(X,edges).mean(dim=0)

if __name__ == "__main__":
    df1 = pd.read_csv(r"C:\Users\frapu\Desktop\TableEmbeddingsWithGNNs\Datasets\testAB.csv")
    #df = pd.read_csv(r"C:\Users\frapu\Desktop\TableEmbeddingsWithGNNs\Datasets\walmart_amazon-tableB.csv")
    print('Starting')
    df2 = pd.read_csv(r"C:\Users\frapu\Desktop\TableEmbeddingsWithGNNs\Datasets\fodors_zagats-master.csv")
    bert_embedding_generator = Bert_Embedding_Buffer()
    string_token_preprocessor = String_token_preprocessor()
    print('Graph generation starts')
    start = time.time()
    g1 = Graph(df1, 'Table', bert_embedding_generator, string_token_preprocessor)

    end = time.time()
    print(f't_exec: {end-start}s')
    print(f'number of embeddings: {g1.X.shape[0]}')
    print('Graph generation starts')
    start = time.time()
    g2 = Graph(df2, 'Table', bert_embedding_generator, string_token_preprocessor)
    end = time.time()
    print(f't_exec: {end-start}s')
    print(f'number of embeddings: {g2.X.shape[0]}')

    gl = Graph_list()
    gl.add_item(g1)
    gl.add_item(g2)
    gl.save(r"C:\Users\frapu\Desktop\TableEmbeddingsWithGNNs\Tests")

    gd = GraphDataset(directory_name=r"C:\Users\frapu\Desktop\TableEmbeddingsWithGNNs\Tests")

    #print('Embedding generation starts')
    # e = EmbeddingGenerator()
    # start = time.time()
    # #emb = e(g.X, g.edges)
    # end = time.time()
    # print(f't_exec: {end-start}s')