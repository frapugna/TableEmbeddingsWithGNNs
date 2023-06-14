import pickle
import pandas as pd
from tqdm import tqdm

def prepare_triple_file(input_file:str, output_file:str)->None:
    try:
        df = pd.read_csv(input_file)
    except:
        raise Exception('Wrong input file path')
    print('Input file loaded') 
    df['table_overlap'] = df['o_a'] / df[['r_a','s_a']].min(axis=1)
    df_out = df[['r_id','s_id','table_overlap']]
    print('Output file generated')
    try:
        df_out.to_csv(output_file, index=False)
    except:
        raise Exception('Write operation failed') 
    print('Write operation succeded')
def rebalance_triple_file(input_file:str, output_file:str,thresholds=None, set_thresholds=False, drop_1=True)->None:
    try:
        df = pd.read_csv(input_file)
    except:
        raise Exception('Wrong input file path')
    
    if set_thresholds:
        if len(thresholds)!=10:
            raise Exception('The thresholds set must contain exactly 10 values')
        occs = []
        for i in range(len(thresholds)):
            occs.append([])
        raise NotImplementedError
    
    if drop_1:
        df_out = df[df['table_overlap']!=1]
    show_samples_distribution(df_out)
    try:
        df_out.to_csv(output_file, index=False)
    except:
        raise Exception('Write operation failed') 
    print('Write operation succeded')

def show_samples_distribution(df:pd.DataFrame)->dict:
    d = {}
    for i in tqdm(df['table_overlap']):
        n = i//0.1/10
        if i == 1:
            n = 1
        try:
            d[n]+=1
        except:
            d[n]=1
    l=[ [k,v] for k,v in d.items()]
    df_occurrencies = pd.DataFrame(l).sort_values(0)
    ax = df_occurrencies.plot(x=0, y=1, kind='bar')
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha='center', va='bottom')
    return d

if __name__=='__main__':
    #prepare_triple_file("/home/francesco.pugnaloni/wikipedia_tables/unprocessed_tables/triples_wikipedia_tables.csv", "/home/francesco.pugnaloni/wikipedia_tables/processed_tables/test_samples.csv")
    rebalance_triple_file("/home/francesco.pugnaloni/wikipedia_tables/processed_tables/test_samples.csv","/home/francesco.pugnaloni/wikipedia_tables/processed_tables/test_samples.csv")