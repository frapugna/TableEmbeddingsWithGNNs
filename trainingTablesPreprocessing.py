import pandas as pd
from tqdm import tqdm
import os
import bz2
import pickle
import _pickle as cPickle
from graph import String_token_preprocessor
from graph import Graph
from node_embeddings import *

# Load any compressed pickle file
def decompress_pickle(file:str)->list:
    data = bz2.BZ2File(file, 'rb')
    data = cPickle.load(data)
    return data

def compute_table_ids(triples_dataset_path:str, output_file:str)->set:
    df = pd.read_csv(triples_dataset_path)
    index_list = []
    for i in tqdm(range(df.shape[0])):
        # index_list.append(str(df['0'][i]))
        # index_list.append(str(df['1'][i]))

        index_list.append(str(df['r_id'][i]))
        index_list.append(str(df['s_id'][i]))

    tables_indexes = set(index_list)
    with open(output_file, 'wb') as f:
        pickle.dump(tables_indexes, f)
    return tables_indexes

def get_tables_ids(file:str)->set:
    with open(file, 'rb') as f:
        return pickle.load(f)

def process_pickle(file:str, index_set:set)->dict:
    in_list = decompress_pickle(file)
    table_dictionary = {}
    for t in in_list:
        if t['_id'] in index_set:
            try:
                table_dictionary[t['_id']] = pd.DataFrame(t['content'][t['num_header_rows']:])
            except KeyError:
                table_dictionary[t['_id']] = pd.DataFrame()
    return table_dictionary

def process_all_pickles(directory_path:str, index_path:str, out_path:str)->dict:
    ids = get_tables_ids(index_path)
    pickle_list = os.listdir(directory_path)
    out = {}
    
    print('Pickle scan starts')
    for f in tqdm(pickle_list):
        out.update(process_pickle(directory_path+'/'+f, ids))
    print('Pickle scan ends')

    print('Saving output')
    with open(out_path, 'wb') as f:
        pickle.dump(out, f)   
    print('Output saved')

    return out

def get_empty_tables_ids(dlist:dict)->list:
    count_none = 0
    out = []
    for k in tqdm(dlist.keys()):
        shape = dlist[k].shape
        if shape[0]==0 or shape[1]==0:
            count_none+=1
            out.append(k)
    print(f'Number of empty tables = {count_none}')
    return out

def drop_small_tables(table_file:str, old_triple_file:str,new_triple_file_out:str, dim_min:int=3)->pd.DataFrame:
    with open(table_file,'rb') as f:
        tables = pickle.load(f)
    to_drop_key_list = []
    for k in tqdm(tables.keys()):
        s = tables[k].shape
        if (s[0] < dim_min) or (s[1] < dim_min):
            to_drop_key_list.append(k)
    df = pd.read_csv(old_triple_file)

    to_drop_key_list = set(to_drop_key_list)

    to_drop_index_list = []
    for i in tqdm(range(df.shape[0])):
        if (str(df['r_id'][i]) in to_drop_key_list) or (str(df['s_id'][i]) in to_drop_key_list):
            to_drop_index_list.append(i)

    out = df.drop(to_drop_index_list)

    print(f'Dropped {len(to_drop_index_list)} samples')

    out.to_csv(new_triple_file_out, index=False)

    return out
    

def generate_graph_dictionary(table_dict_path:str, out_path:str)->dict:
    with open(table_dict_path,'rb') as f:
        table_dict = pickle.load(f)
    
    embedding_buffer = FasttextEmbeddingBuffer(model='fasttext-wiki-news-subwords-300')
    string_token_preprocessor = String_token_preprocessor()

    out = {}

    print('Graphs generation starts')
    for k in tqdm(table_dict.keys()):
        try:
            out[k] = Graph(table_dict[k], k, embedding_buffer, string_token_preprocessor, token_length_limit=None)
        except:
            out[k] = None
    print('Graph generation ends')

    print('Saving output')
    with open(out_path, 'wb') as f:
        pickle.dump(out, f)   
    print('Output saved')

    return out

if __name__ == '__main__':
    # ids = compute_table_ids("/dati/home/francesco.pugnaloni/wikipedia_tables/processed_tables/test_samples_base.csv", 
    #                         "/dati/home/francesco.pugnaloni/wikipedia_tables/processed_tables/table_id_set.pkl")
    # ids = get_tables_ids("/dati/home/francesco.pugnaloni/wikipedia_tables/processed_tables/table_id_set.pkl")
    # print(f'Found {len(ids)} different ids')
    # ids = get_tables_ids("/dati/home/francesco.pugnaloni/wikipedia_tables/processed_tables/table_id_set.pkl")
    # t = process_pickle("/dati/home/francesco.pugnaloni/wikipedia_tables/unprocessed_tables/wikipedia_tables_zip/enwiki-20190901-pages-meta-history7.xml-p972010p972235.output.pkl",ids)
    # process_all_pickles("/home/francesco.pugnaloni/wikipedia_tables/unprocessed_tables/wikipedia_tables_zip",
    #                     "/dati/home/francesco.pugnaloni/wikipedia_tables/processed_tables/table_id_set.pkl",
    #                     "/home/francesco.pugnaloni/wikipedia_tables/processed_tables/full_table_dict_with_id.pkl"
    #                     )
    
    gd = generate_graph_dictionary("/dati/home/francesco.pugnaloni/wikipedia_tables/processed_tables/full_table_dict_with_id.pkl", "/dati/home/francesco.pugnaloni/wikipedia_tables/processed_tables/full_graphs_dict_with_id.pkl")

    # drop_small_tables("/dati/home/francesco.pugnaloni/wikipedia_tables/processed_tables/full_table_dict_with_id.pkl",
    #                   "/dati/home/francesco.pugnaloni/wikipedia_tables/processed_tables/test_samples_no_ones.csv",
    #                   "/dati/home/francesco.pugnaloni/wikipedia_tables/processed_tables/test_samples_no_small_tables.csv")
    
