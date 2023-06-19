import pandas as pd
import random
from graph import *
from typing import Union

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

if __name__ == '__main_':
    tl = generateDatasetList(1000, 5, 50, ["ciao "*40])
    embedding_buffer_fastt = FasttextEmbeddingBuffer()
    embedding_buffer_bert = Bert_Embedding_Buffer()
    string_token_preprocessor = String_token_preprocessor()
    s = "dog "*40
    for i in range(10):
        start = time.time()
        embedding_buffer_fastt(s)
        embedding_buffer_fastt.pop_embeddings()
        end = time.time()
        print(f'Iter {i}: {end-start}s')
    
    for i in range(10):
        start = time.time()
        embedding_buffer_fastt(s)
        embedding_buffer_fastt.pop_embeddings()
        end = time.time()
        print(f'Iter {i}: {end-start}s')


if __name__ == '__main__':
    tl = generateDatasetList(10000, 5, 50, ["ciao "*40])
    print('This_test')
    embedding_buffer_fastt = FasttextEmbeddingBuffer()
    embedding_buffer_bert = Bert_Embedding_Buffer()
    string_token_preprocessor = String_token_preprocessor()
    print('FastTextEmbedding test starts______________________')
    start = time.time()
    t = 0
    for i in range(len(tl)):
        k = processDataset(tl[i], embedding_buffer_fastt, string_token_preprocessor)
        if i!=0:
            t+=k
    end = time.time()
    t_avg_fastt = t/len(tl)
    t_tot_fastt = end-start


    print('Bert test starts______________________')
    start = time.time()
    t = 0
    for i in range(len(tl)):
        k = processDataset(tl[i], embedding_buffer_bert, string_token_preprocessor)
        if i!=0:
            t+=k
    end = time.time()
    t_avg_bert = t/len(tl)
    t_tot_bert = end-start

    print(f'fastTextEmbedding___________________________\nAverage graph generation time: {t_avg_fastt}\nTotal t_exec: {t_tot_fastt}s') 
    print(f'Bert________________________________________\nAverage graph generation time: {t_avg_bert}\nTotal t_exec: {t_tot_bert}s') 
