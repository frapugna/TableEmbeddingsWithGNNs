import torch
from transformers import BertModel, BertTokenizer
from abc import ABC, abstractmethod

class Embedding_buffer(ABC):
    @abstractmethod
    def __call__(self, sentence):
        return NotImplemented
    
    @abstractmethod
    def pop_embeddings(self):
        return NotImplemented
    
class w2v_embedding_buffer(Embedding_buffer):
    def __init__(self):
        pass

class Bert_Embedding_Generator:
    def __init__(self, output_hidden_states=False, bert_lm_name='bert-base-uncased'):
        """
            bert_lm_name: 'bert-base-uncased', 'distilbert-base-uncased'
        """
        self.tokenizer = BertTokenizer.from_pretrained(bert_lm_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = BertModel.from_pretrained(bert_lm_name, output_hidden_states=output_hidden_states).to(self.device).eval()

    def encode(self, l):
        #return self.tokenizer.encode(l)
        return self.tokenizer(l, padding=True)
    
    def decode(self, l):
        return self.tokenizer.convert_ids_to_tokens(l)
    
    def __call__(self, sentence, strategy='CLS'):
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

class Bert_Embedding_Buffer(Embedding_buffer):
    def __init__(self, buffer_size=512, output_hidden_states=False):
        self.bert_embedding_generator = Bert_Embedding_Generator(output_hidden_states=output_hidden_states)
        self.buffer_size = buffer_size
        self.n_sentences = 0
        self.buffer = []
        self.embeddings = None

    def __add_new_emb(self, new_emb):
        if self.n_sentences == 1:
            new_emb = new_emb.reshape(1, -1)

        if self.embeddings == None:
            self.embeddings = new_emb
        else:
            self.embeddings = torch.cat((self.embeddings, new_emb), dim=0)

    def __process_buffer(self):
        new_emb = self.bert_embedding_generator(self.buffer)
        self.__add_new_emb(new_emb)
        self.buffer = []
        self.n_sentences = 0
        

    def __call__(self, sentence):
        self.buffer.append(sentence)
        self.n_sentences += 1
        if self.n_sentences >= self.buffer_size:
            self.__process_buffer()
        
    def pop_embeddings(self):
        if self.n_sentences > 0:
            self.__process_buffer()

        out = self.embeddings
        self.embeddings = None
        return out
        
if __name__ == '__main__':
    s1 = 'i gatti sono animali'
    s2 = 'i cani sono animali'
    s3 = 'cats are animals'
    s4 = 'l\'acciaio non è un essere vivente'
    l = [s1, s2, s3, s4]
    b_buffer = Bert_Embedding_Buffer(buffer_size=3)
    for i in range(len(l)):
        b_buffer(l[i])
    emb = b_buffer.pop_embeddings()
    print('ok')

    s1 = 'i gatti sono animali'
    s2 = 'i cani sono animali'
    s3 = 'cats are animals'
    s4 = 'l\'acciaio non è un essere vivente'
    l = [s1,s2,s3,s4]
    bert = Bert_Embedding_Generator()
    emb_matrix = bert(l)


    cosi = torch.nn.CosineSimilarity(dim=0)
    distance = cosi(emb_matrix[0],emb_matrix[1])
    print(distance)
    print(emb_matrix.shape)
    s1 = 'i gatti sono animali'
    s2 = 'i cani sono animali'
    s3 = 'cats are animals'
    s4 = 'l\'acciaio non è un essere vivente'
    l1 = [s1]
    l2 = [s2]
    l3 = [s3]
    l4 = [s4]
    bert = Bert_Embedding_Generator()

    e1 = bert(l1)
    e2 = bert(l2)
    e3 = bert(l3)
    e4 = bert(l4)