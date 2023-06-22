import pandas as pd
import random
from graph import *
from trainingTablesPreprocessing import *
import pickle
import time

def generateDatasetList(n_datasets, max_n_cols, max_n_rows, sentences=None):
    if not(sentences):
        sentences = ['I have a pen and an apple','this sentence will be pretty long and hopefully it will confuse bert with a lot of useless words that are not only stop wrods but also other stuff and yws this sentence is pretty long again',
                  'cat', 'dog','Keanu Reeves is an actor 3', '4','1','2','3','4','5','6','7','8','9','0','I believe I can fly', 'The plot of the book is that it is a book', 'Leopard','JaGuar',
                  'this sentence will be pretty long and hopefully it will confuse bert with a lot of useless words that are not only stop wrods but also other stuff and yws this sentence is pretty long again',
                  'this sentence will be pretty long and hopefully it will confuse bert with a lot of useless words that are not only stop wrods but also other stuff and yws this sentence is pretty long again']
    n_sentences = len(sentences)
    tables = []
    for _ in range(n_datasets):
        r = random.randint(1,max_n_rows)
        c = random.randint(1,max_n_cols)
        rows = []
        for i in range(r):
            row = []
            for j in range(c):
                n = random.randint(0,n_sentences-1)
                row.append(sentences[n])
            rows.append(row)
        tables.append(pd.DataFrame(rows))
    return tables

def processDataset(df,embedding_buffer, string_token_preprocessor):
    start = time.time()
    Graph(df, 'Table1', embedding_buffer, string_token_preprocessor, verbose=False,token_length_limit=None)
    end = time.time()
    tot = end-start
    print(f'T_exec: {tot}s')
    return tot

def generate_test_training_stuff(num_tables: int, num_samples: int, 
                                 outdir: str="/home/francesco.pugnaloni/wikipedia_tables/small_dataset_debug",
                                 w: int=10, h: int=10) -> None:
    random.seed(42)
    dl = generateDatasetList(num_tables, h, w)
    td = {str(i):dl[i] for i in range(len(dl))}
    with open(outdir+'/tables.pkl', 'wb') as f:
        pickle.dump(td, f)
    dg = generate_graph_dictionary(outdir+'/tables.pkl', outdir+'/graphs.pkl')
    triples = []
    for _ in range(num_samples):
        a = random.randint(0,len(dg)-1)
        b = random.randint(0,len(dg)-1)
        o = random.random()
        triples.append([a,b,o])
    pd.DataFrame(triples).to_csv(outdir+"/triples.csv", index=False)

def load_test_training_stuff(filedir: str="/home/francesco.pugnaloni/wikipedia_tables/small_dataset_debug") -> dict:
    with open(filedir+'/tables.pkl', 'rb') as f:
        td = pickle.load(f)
    with open(filedir+'/graphs.pkl', 'rb') as f:
        gd = pickle.load(f)
    triples = pd.read_csv(filedir+'/triples.csv')

    return {'tables':td, 'graphs':gd, 'triples':triples}

if __name__ == '__main__':
    generate_test_training_stuff(50, 100)
