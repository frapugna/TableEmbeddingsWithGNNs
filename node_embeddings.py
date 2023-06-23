import torch
from transformers import BertModel, BertTokenizer
from abc import ABC, abstractmethod
import gensim.downloader as api

class Embedding_buffer(ABC):
    @abstractmethod
    def __call__(self, sentence: str) -> NotImplemented:
        """Adds a sentence to the buffer of sentences to embed

        Args:
            sentence (str): the sentence to embed

        Returns:
            NotImplemented: up to the specific implementation
        """
        return NotImplemented
    
    @abstractmethod
    def pop_embeddings(self) -> NotImplemented:
        """The buffer is emptied and the embeddings inside it returned

        Returns:
            NotImplemented: up to the specific implementation
        """
        return NotImplemented
    
    @abstractmethod
    def add_nan_embedding(self) -> NotImplemented:
        """Special method to manage nan sentences

        Returns:
            NotImplemented: up to the specific implementation
        """
        return NotImplemented

class Bert_Embedding_Generator:
    def __init__(self, output_hidden_states: bool=False, bert_lm_name: str='bert-base-uncased') -> None:
        """The class init method

        Args:
            output_hidden_states (bool, optional): NA. Defaults to False.
            bert_lm_name (str, optional): the version of bert to use {'bert-base-uncased', 'distilbert-base-uncased'}. Defaults to 'bert-base-uncased'.
        """
        self.tokenizer = BertTokenizer.from_pretrained(bert_lm_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = BertModel.from_pretrained(bert_lm_name, output_hidden_states=output_hidden_states).to(self.device).eval()

    def encode(self, l: str) -> list:
        """Perform an encoding operation necessary to generate the embeddings

        Args:
            l (str): the str to encode

        Returns:
            list: a list of encodings
        """
        return self.tokenizer(l, padding=True)
    
    def decode(self, l: list) -> list:
        """Revert the encode operation

        Args:
            l (list): the list to decode

        Returns:
            list: the decoded tokens
        """
        return self.tokenizer.convert_ids_to_tokens(l)
    
    def __call__(self, sentence: str, strategy: str='CLS') -> torch.Tensor:
        """The class call method, it manages the emebdding generation

        Args:
            sentence (str): sentence to embed
            strategy (str, optional): do not modify. Defaults to 'CLS'.

        Returns:
            torch.Tensor: the embedding of the sentence
        """
        enc = self.encode(sentence)
        enc = {k:torch.LongTensor(v).to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.model(**enc)
        hidden_states = out['last_hidden_state']

        if strategy == 'CLS':
            sentence_embedding = hidden_states[:,0]

        elif strategy == 'average':
            sentence_embedding = torch.mean(hidden_states, dim=1)
        
        if len(sentence) == 1:
            return sentence_embedding.squeeze(0)
        else:
            return sentence_embedding

class FasttextEmbeddingBuffer(Embedding_buffer):
    def __init__(self, model: str='word2vec-google-news-300') -> None:   #'fasttext-wiki-news-subwords-300'
        """The class init method

        Args:
            model (str, optional): The version of fasttext/word2vec to use {'fasttext-wiki-news-subwords-300', 'word2vec-google-news-300'}. Defaults to 'word2vec-google-news-300'.
        """
        print('Loading fasttext model, it will take 2/3 minutes')
        self.model = api.load(model)
        print('Model loaded')
        self.vector_size = self.model.vector_size
        self.n_embeddings = 0
        self.embeddings = None

    def add_nan_embedding(self) -> None:
        """Method to manage the nan values
        """
        vector = torch.zeros(self.vector_size)
        try:
            self.embeddings = torch.cat((self.embeddings, vector.unsqueeze(0)), dim=0)
        except TypeError:
            self.embeddings = vector.unsqueeze(0)
        self.n_embeddings += 1
    
    def __get_embedding(self, word: str) -> torch.Tensor:
        """Provide the embedding of a word

        Args:
            word (str): the word to embed

        Returns:
            torch.Tensor: the embedding of the word
        """
        return self.model[word]

    def __call__(self, sentence: str) -> None:
        """Add the sentence to embed to the buffer

        Args:
            sentence (str): sentence to embed
        """
        word_list = sentence.split()
        vector = None
        n_words = 0
        for w in word_list:
            try:
                emb = torch.tensor(self.__get_embedding(w))
                try:
                    vector += emb
                    n_words += 1
                except:
                    vector = emb
                    n_words += 1
            except:
                pass
        if vector == None:
            vector = torch.rand(300)
        else:
            vector =  vector / n_words
        self.n_embeddings += 1
        try:
            self.embeddings = torch.cat((self.embeddings, vector.unsqueeze(0)), dim=0)
        except TypeError:
            self.embeddings = vector.unsqueeze(0) 

    def pop_embeddings(self) -> torch.Tensor:
        """Return all the generated embeddings and reset the buffer

        Returns:
            torch.Tensor: tensor with one row for every embedding 
        """
        out = self.embeddings
        self.embeddings = None
        self.n_embeddings = 0
        return out
    
class Bert_Embedding_Buffer(Embedding_buffer):
    def __init__(self, buffer_size: int=512, output_hidden_states: bool=False) -> None:
        """The class init method

        Args:
            buffer_size (int, optional): the size of the embedding buffer (numer of sentences to process at once). Defaults to 512.
            output_hidden_states (bool, optional): NA. Defaults to False.
        """
        self.bert_embedding_generator = Bert_Embedding_Generator(output_hidden_states=output_hidden_states)
        self.buffer_size = buffer_size
        self.n_sentences = 0
        self.buffer = []
        self.embeddings = None

    def add_nan_embedding(self) -> None:
        """Method to manage the nan values
        """
        self.__process_buffer()
        vector = torch.zeros(768)
        try:
            self.embeddings = torch.cat((self.embeddings, vector.unsqueeze(0)), dim=0)
        except:
            self.embeddings = torch.cat((self.embeddings, vector.unsqueeze(0).to('cuda')), dim=0)

    def __add_new_emb(self, new_emb: torch.Tensor) -> None:
        """Add a new embedding to the output tensor

        Args:
            new_emb (torch.Tensor): the embedding to add
        """
        if self.n_sentences == 1:
            new_emb = new_emb.reshape(1, -1)

        if self.embeddings == None:
            self.embeddings = new_emb
        else:
            try:
                self.embeddings = torch.cat((self.embeddings, new_emb), dim=0)
            except:
                self.embeddings = torch.cat((self.embeddings, new_emb.unsqueeze(0)), dim=0)

    def __process_buffer(self) -> None:
        """The buffer is processed and the embeddings are saved internally
        """
        new_emb = self.bert_embedding_generator(self.buffer)
        self.__add_new_emb(new_emb)
        self.buffer = []
        self.n_sentences = 0
        

    def __call__(self, sentence: str) -> None:
        """Add a new sentence to the buffer

        Args:
            sentence (str): the sentence to add to the buffer
        """
        self.buffer.append(sentence)
        self.n_sentences += 1
        if self.n_sentences >= self.buffer_size:
            self.__process_buffer()
        
    def pop_embeddings(self) -> torch.Tensor:
        """Obtain the generated embeddings

        Returns:
            torch.Tensor: tensor with one row for every embedding 
        """
        if self.n_sentences > 0:
            self.__process_buffer()

        out = self.embeddings
        self.embeddings = None
        return out
    