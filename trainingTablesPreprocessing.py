import pandas as pd
from tqdm import tqdm
import os
import bz2
import pickle
import _pickle as cPickle
from graph import String_token_preprocessor
from graph import Graph
from node_embeddings import *


def decompress_pickle(file: str) -> list:
    """Function to load a pickle file that is also compressed using bz2

    Args:
        file (str): path to the file

    Returns:
        list: list containing dictionary associated to wikipedia tables
    """
    data = bz2.BZ2File(file, 'rb')
    data = cPickle.load(data)
    return data

def compute_table_ids(triples_dataset_path: str, output_file: str) -> set:
    """Provided a triple file the id of the table that appear in the triples are provided

    Args:
        triples_dataset_path (str): path to the dataset containing the triples
        output_file (str): path to the file where to save the generated indexes

    Returns:
        set: set containing the indexes
    """
    df = pd.read_csv(triples_dataset_path)
    index_list = []
    for i in tqdm(range(df.shape[0])):
        index_list.append(str(df['r_id'][i]))
        index_list.append(str(df['s_id'][i]))

    tables_indexes = set(index_list)
    with open(output_file, 'wb') as f:
        pickle.dump(tables_indexes, f)
    return tables_indexes

def get_tables_ids(file: str) -> set:
    """Function to load the indexes set

    Args:
        file (str): path to the file containing the indexes

    Returns:
        set: set of the indexes
    """
    with open(file, 'rb') as f:
        return pickle.load(f)

def process_pickle(file: str, index_set: set) -> dict:
    """Function to generate from a pickle file and an index set a table dictionary 

    Args:
        file (str): path to the file containing the tables
        index_set (set): set of the desired indexes

    Returns:
        dict: dictionary containing only the tables associated to the rpovided indexes
    """
    in_list = decompress_pickle(file)
    table_dictionary = {}
    for t in in_list:
        if t['_id'] in index_set:
            try:
                table_dictionary[t['_id']] = pd.DataFrame(t['content'][t['num_header_rows']:])
            except KeyError:
                table_dictionary[t['_id']] = pd.DataFrame()
    return table_dictionary

def process_all_pickles(directory_path: str, index_path: str, out_path: str) -> dict:
    """Function to process multiple pickle files

    Args:
        directory_path (str): path to the directory containing the pickle files
        index_path (str): path to file containing the indexes
        out_path (str): path to the file where to dump the dictionary 

    Returns:
        dict: the generated dictionary of tables
    """
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

def get_empty_tables_ids(dlist: dict) -> list:
    """Function that provided a dictionary of tables returns th ids of the empty ones

    Args:
        dlist (dict): a table dictionary

    Returns:
        list: the list of the indexes of the empty tables
    """
    count_none = 0
    out = []
    for k in tqdm(dlist.keys()):
        shape = dlist[k].shape
        if shape[0]==0 or shape[1]==0:
            count_none+=1
            out.append(k)
    print(f'Number of empty tables = {count_none}')
    return out

def drop_small_tables(table_file: str, old_triple_file: str,new_triple_file_out: str, dim_min: int=3) -> pd.DataFrame:
    """Function to generate a new table dictionary from a rpovided one dropping all the "small tables"

    Args:
        table_file (str): path do the file containing the table dictionary
        old_triple_file (str): path to the old triple file
        new_triple_file_out (str): path to the new triple file
        dim_min (int, optional): lower band of the dimension of the tables to extract. Defaults to 3.

    Returns:
        pd.DataFrame: new dataset containg only table that are not small
    """
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
    

def generate_graph_dictionary(table_dict_path: str, out_path: str) -> dict:
    """Generate a graph dictionary from a table dictionary

    Args:
        table_dict_path (str): path to the table dictionary
        out_path (str): path to the file where to save the new graph dictionary

    Returns:
        dict: the generated graph dictionary
    """
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
