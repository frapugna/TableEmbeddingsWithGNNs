import pandas as pd
import random
import os
import pickle
from tqdm import tqdm
from trainingTablesPreprocessing import compute_table_ids

def generate_small_triple_dataset(n: int, 
                                  triple_file: str="/dati/home/francesco.pugnaloni/wikipedia_tables/processed_tables/test_samples_no_small_tables.csv"
                                  ) -> pd.DataFrame:
    df = pd.read_csv(triple_file)
    random.seed(42)
    random_indices = random.sample(range(df.shape[0]), min(n, df.shape[0]))
    random_dataset = df.iloc[random_indices][:]
    random_dataset.reset_index(drop=True, inplace=True)
    
    return random_dataset

def create_directory(directory_path):
    try:
        # Create a new directory at the specified path
        os.mkdir(directory_path)
        print("New directory created")
    except FileExistsError:
        pass
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def generate_small_graph_dataset(full_graph_dataset: dict, indexes: set) -> dict:
    out = {}
    for k in tqdm(indexes):
        try:
            out[str(k)] = full_graph_dataset[str(k)]
        except:
            out[str(int(k))] = full_graph_dataset[str(int(k))]
    return out

def load_small_dataset(dir: str) -> dict:
    out = {}
    out['triples'] = pd.read_csv(dir+'/samples.csv')
    with open(dir+'/graphs.pkl','rb') as f:
        out['graphs'] = pickle.load(f)
    return out

def generate_small_datasets(length_list: list, 
                            triple_file: str="/dati/home/francesco.pugnaloni/wikipedia_tables/processed_tables/test_samples_no_small_tables.csv",
                            graph_file: str="/dati/home/francesco.pugnaloni/wikipedia_tables/processed_tables/full_graphs_dict_with_id.pkl",
                            out_dir: str="/dati/home/francesco.pugnaloni/wikipedia_tables/small_tables"
                            ) -> None:
    print('Loading graph dictionary, it will take some time (~9/10 minutes)')
    with open(graph_file, 'rb') as f:
        gd = pickle.load(f)
    create_directory(out_dir)
    for n in length_list:
        newdir = out_dir+'/'+str(n)+'_samples'
        create_directory(newdir)

        print(f'Generating new dataset of length {n}_____________________')
        df = generate_small_triple_dataset(n, triple_file)
        df.to_csv(newdir+'/samples.csv', index=False)
        
        indexes = compute_table_ids(newdir+'/samples.csv', newdir+'/ids')

        print('Small graph generation starts')
        small_gd = generate_small_graph_dataset(gd, indexes)
        with open(newdir+'/graphs.pkl', 'wb') as f:
            pickle.dump(small_gd, f)
    
    print(f'{len(length_list)} new dataset successfully generated in the desired directory')


if __name__ == '__main__':
    # generate_small_datasets([10**1,10**2,10**3,10**4,10**5], 
    #                         graph_file="/dati/home/francesco.pugnaloni/wikipedia_tables/small_dataset_debug/graphs.pkl",
    #                         triple_file="/dati/home/francesco.pugnaloni/wikipedia_tables/small_dataset_debug/triples.csv"
    #                         )
    # generate_small_datasets([10**1,10**2,10**3,10**4,10**5])
    pass
