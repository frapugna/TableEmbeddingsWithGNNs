import pandas as pd
import pickle
import re
import nltk
from nltk.corpus import stopwords
import math
from node_embeddings import *

"""
    Assumptions:
    - all the graphs can fit in memory
    - all the columns of the dataframe have a string as identifier
    - all the columns inside the same dataframe have different names
    - the graph is undirected and unweighted
"""

def isNaN(num):
    return num != num

def get_order_of_magnitude(number):
    if number == 0:
        return 0  # Logarithm of 0 is undefined, return 0 as the order of magnitude
    else:
        return int(math.floor(math.log10(abs(number))))

def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

def preprocess_numbers(n, operations=['cast_to_float', 'discretize_strict']):
    if 'cast_to_float' in operations:
        n = float(n)
    if 'discretize_strict' in operations:
        div = 10 ** get_order_of_magnitude(n)
        n = n//div*div
    return str(n)

class String_token_preprocessor:
    def __init__(self, language='english'):
        nltk.download('stopwords')
        self.stopwords = stopwords.words(language)

    def __call__(self, s, token_length_limit=None,operations=['lowercase', 'split', 'remove_stop_words']):
        out = s
        if len(operations) == 0:
            return [out]
        
        if 'lowercase' in operations:
            out = out.lower()

        if 'split' in operations:
            out = re.split(' |_|\|', out)
        if token_length_limit:
            out = out[0:token_length_limit]
        if 'remove_stop_words' in operations:
            out = [t for t in out if not(t in self.stopwords)]

        return out

class Graph_list(torch.utils.data.Dataset):
    def __init__(self, directory_name=False,save=False, load=False):
        if directory_name:
            self.load(directory_name)
        else:
            self.graph_list = []
            self.size = 0

    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        try:
            return self.graph_list[idx]
        except IndexError:
            raise Exception("Index out of bound")
    def add_item(self, g):
        self.graph_list.append(g)
        self.size += 1
    
    def load(self, directory_name):
        try:
            f1 = open(directory_name+'/graph_list.pkl', 'rb')
            self.graph_list = pickle.load(f1)
            self.size = len(self.graph_list)
            f1.close()
        except:
            raise Exception('Read operation failed')
        self.size = len(self.graph_list)
    def save(self, directory_name):
        try:
            f1 = open(directory_name+'/graph_list.pkl', 'wb')
            pickle.dump(self.graph_list, f1)
            f1.close()
        except:
            raise Exception('Write operation failed')
class Graph:
    def get_number_of_nodes(self):
        return len(self.index_to_token)
    def __str__(self):
        return ''.join(f'{self.index_to_token[self.edges[0][i]]}<-->{self.index_to_token[self.edges[1][i]]}\n' for i in range(self.number_of_edges))

    
    def __add_edge(self, id_a, id_b):
        self.edges[0].append(id_a)
        self.edges[1].append(id_b)
        self.number_of_edges += 1
            
        self.edges[0].append(id_b)
        self.edges[1].append(id_a)
        self.number_of_edges += 1

    def __get_next_index(self, category):
        if category=='column':
            out = self.next_column_index
            self.next_column_index+=1
        elif category=='row':
            out = self.next_row_index
            self.next_row_index+=1
        elif category=='value':
            out = self.next_value_index
            self.next_value_index+=1
        else:
            raise Exception('Unexpected index format')
        return out

    def __add_value_to_index(self, value_index, column_idx, row_idx):
        self.columns_rows_to_values[column_idx].append(value_index)
        self.columns_rows_to_values[row_idx].append(value_index)

    def __generate_feature_matrix(self, embeddings):
        out = [torch.mean(embeddings[l], dim=0).reshape(1,-1) for l in self.columns_rows_to_values] 
        out = torch.cat(out, dim=0) #cat of a list
        out = torch.cat((out, embeddings), dim=0)
        return out

    def __init__(self,  df: pd.DataFrame, table_name: str, embedding_buffer: Embedding_buffer, preprocess_string_token: String_token_preprocessor,  
                token_length_limit: int=20,link_tuple_token: bool=True, link_token_attribute: bool=True, link_tuple_attribute: bool=False, 
                attribute_preprocess_operations: list=['lowercase', 'drop_numbers_from_strings'], 
                string_preprocess_operations: list=['lowercase', 'split', 'remove_stop_words'],
                number_preprocess_operations: list=['cast_to_float', 'discretize_strict'], drop_na: bool=False, verbose: bool=False) -> None:
        """
            Desc: a dataframe will be processed to generate nodes and edges to add to the graph
            Params:
            -df: the dataframe to process
            -table_name: the name of the dataframe, it will be used during the node generation
            -link_tuple_token: if true tuples and tokens will be linked by edges
            -link_token_attribute: if true tokens and attributes will be linked by edges
            -link_tuple_attribute: if true tuples and attributes will be linked by edges
        """
        self.edges = [[],[]]
        self.X = None
        self.table_name = table_name
        self.number_of_edges = 0

        link_tuple_token = link_tuple_token
        link_token_attribute = link_token_attribute
        link_tuple_attribute = link_tuple_attribute
        string_preprocess_operations = ['lowercase', 'split', 'remove_stop_words']
        number_preprocess_operations = ['cast_to_float', 'discretize_strict']
        if drop_na:
            df.dropna(axis=0,how='all', inplace=True)
            df.dropna(axis=1,how='all', inplace=True)
        
        n_columns = df.shape[1]
        n_rows = df.shape[0]

        if (n_columns == 0) or (n_rows == 0):
            raise Exception('You cannot generate a graph from an empty DataFrame')

        self.next_column_index = 0
        self.next_row_index = n_columns
        self.next_value_index = n_columns + n_rows

        self.columns_rows_to_values = [[] for _ in range(n_columns+n_rows)]  #contains for every column and row the list of the indexes of the associted values 
        index_left_shift = len(self.columns_rows_to_values)
        column_indexes = [i for i in range(n_columns)]
        value_to_index = {}
        values_count = 0
        #Tuple and token node
        for i in range(df.shape[0]):
            row_index = self.__get_next_index('row')
            if (i % 100 == 0) and verbose:
                print(f'Row: {i}/{n_rows}')
            if link_tuple_attribute:
                for id in column_indexes:
                    self.__add_edge(row_index, id)
            
            for j in range(df.shape[1]):
                t = df.iloc[i,j]
                #NaN values management
                if pd.isnull(t):
                    sentence = '#£$/'   #Random key for NaN
                    try:
                        value_index = value_to_index[sentence]
                        self.__add_value_to_index(value_index-index_left_shift, j, row_index)
                    except:
                        embedding_buffer.add_nan_embedding()
                        value_index = self.__get_next_index('value')
                        self.__add_value_to_index(values_count, j, row_index)
                        values_count += 1
                        value_to_index[sentence] = value_index

                    if link_tuple_token:
                        self.__add_edge(value_index, row_index)

                    if link_token_attribute:
                        self.__add_edge(value_index, column_indexes[j])

                    continue

                if isinstance(t, str) and not(is_float(t)):
                    token_list = preprocess_string_token(t, token_length_limit,operations=string_preprocess_operations)

                elif is_float(str(t)):
                    #Note: the string "infinity" will trigger an exception and will be skipped
                    try:
                        token_list = [preprocess_numbers(t, operations=number_preprocess_operations)]
                    except:
                        print(f'An exception occurred in the position [{i},{j}] of the table {table_name} ')
                        continue
                else:
                    raise Exception(f'The token {t} is of type {type(t)} and it is not supported')
                sentence = ' '.join(token_list)
                try:
                    value_index = value_to_index[sentence]
                    
                    self.__add_value_to_index(value_index-index_left_shift, j, row_index)   #ADDED_________________________-
                except:  
                    embedding_buffer(sentence)
                    value_index = self.__get_next_index('value')
                    self.__add_value_to_index(values_count, j, row_index)
                    values_count += 1
                    value_to_index[sentence] = value_index

                if link_tuple_token:
                    self.__add_edge(value_index, row_index)
                if link_token_attribute:
                    self.__add_edge(value_index, column_indexes[j])
        
        value_embeddings = embedding_buffer.pop_embeddings()
        self.X = self.__generate_feature_matrix(value_embeddings)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.edges = torch.tensor(self.edges, dtype=torch.long).to(device=device)
    
if __name__ == "__main__":
    from smallDatasetGenerator import *
    data = load_test_training_stuff("/home/francesco.pugnaloni/tmp/small_tables")
    t = data['tables']['25']
    t['1'] = [pd.NA, 15, pd.NA]
    #embedding_buffer = FasttextEmbeddingBuffer(model='fasttext-wiki-news-subwords-300')
    #embedding_buffer = FasttextEmbeddingBuffer()
    embedding_buffer = Bert_Embedding_Buffer()
    string_token_preprocessor = String_token_preprocessor()
    g = Graph(t, 'luca', embedding_buffer, string_token_preprocessor)
    print(f'Number of NA: {torch.sum(torch.isnan(g.X))}')
    print('ok')